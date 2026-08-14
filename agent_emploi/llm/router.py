"""Routeur LLM : une seule porte d'entrée pour tous les agents.

Les agents disent *quelle tâche* ils exécutent, jamais *quel modèle* ils veulent.
Le mappage tâche → fournisseur → modèle vit dans `config.yaml`, ce qui permet de
déplacer une tâche du tier gratuit au tier payant (ou l'inverse) sans toucher au
code d'aucun agent.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from agent_emploi.config import Config, ModelRef, api_key
from agent_emploi.llm.budget import BudgetTracker, estimate_cost
from agent_emploi.llm.providers import (
    REGISTRY,
    Completion,
    Provider,
    ProviderError,
    RateLimited,
    TransientError,
)
from agent_emploi.models import LlmUsage

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

#: Bloc de code Markdown éventuellement enroulé autour du JSON par le modèle.
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

#: Nombre d'attentes consenties sur un même modèle avant de passer au repli.
MAX_RATE_LIMIT_RETRIES = 2

#: Au-delà, l'attente n'est plus une cadence mais un quota épuisé (souvent
#: journalier) : mieux vaut tenter le repli que bloquer la passe.
MAX_RATE_LIMIT_WAIT = 30.0

#: Attente retenue quand le fournisseur signale la saturation sans chiffrer le
#: délai. Les fenêtres de cadence des tiers gratuits se comptent en minutes ;
#: dix secondes suffisent le plus souvent à en sortir.
DEFAULT_RATE_LIMIT_WAIT = 10.0

#: Reprises consenties sur une panne de transport (DNS, connexion, délai).
#: Une coupure de quelques secondes ne doit pas coûter une offre, encore moins
#: une passe entière.
MAX_TRANSIENT_RETRIES = 3

#: Première attente avant reprise, en secondes. Doublée à chaque essai : 2, 4,
#: 8 — de quoi traverser une bascule de résolveur sans faire patienter
#: longtemps quand le réseau est vraiment coupé.
TRANSIENT_BACKOFF = 2.0


class LlmError(RuntimeError):
    """Échec définitif : le modèle principal et son repli ont tous deux échoué.

    `transient` distingue les deux causes que l'appelant ne doit pas traiter de
    la même façon : une route cassée (clé absente, modèle inconnu) ne guérira
    pas toute seule, alors qu'un réseau coupé n'apprend rien sur la route.
    """

    def __init__(self, message: str, *, transient: bool = False) -> None:
        super().__init__(message)
        self.transient = transient


def extract_json(text: str) -> str:
    """Isole le JSON d'une réponse, même enrobée de texte ou de balises Markdown.

    Les modèles du tier gratuit ajoutent volontiers « Voici le résultat : » ou
    des balises de code autour du JSON demandé.
    """
    cleaned = _FENCE.sub("", text).strip()
    if cleaned.startswith("{") or cleaned.startswith("["):
        return cleaned
    # Repli : le plus grand fragment entre la première accolade et la dernière.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned


class Router:
    """Résout la tâche vers un modèle, appelle, journalise, replie si besoin."""

    def __init__(self, config: Config, budget: BudgetTracker | None = None) -> None:
        self.config = config
        self.budget = budget or BudgetTracker(
            config.paths.usage_file, config.budget
        )
        self._providers: dict[str, Provider] = {}

    def _provider(self, name: str) -> Provider:
        """Instancie un fournisseur à la demande, puis le réutilise.

        L'instanciation paresseuse évite d'exiger une clé d'API pour un
        fournisseur que la passe courante n'utilisera jamais.
        """
        if name not in self._providers:
            try:
                cls = REGISTRY[name]
            except KeyError:
                known = ", ".join(sorted(REGISTRY))
                raise LlmError(
                    f"fournisseur inconnu: {name!r} (connus: {known})"
                ) from None
            try:
                key = api_key(name)
            except RuntimeError as exc:
                # Clé manquante : c'est une erreur de route, pas une panne du
                # processus. Convertie en LlmError, elle laisse sa chance au
                # repli plutôt que d'interrompre toute la passe.
                raise LlmError(str(exc)) from None
            self._providers[name] = cls(key)
        return self._providers[name]

    # ------------------------------------------------------------------- appels

    def complete(
        self,
        task: str,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 2048,
        json_mode: bool = False,
        job_id: str | None = None,
    ) -> str:
        """Exécute une tâche et retourne le texte brut de la réponse."""
        return self.completion(
            task,
            prompt,
            system=system,
            max_tokens=max_tokens,
            json_mode=json_mode,
            job_id=job_id,
        ).text

    def completion(
        self,
        task: str,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 2048,
        json_mode: bool = False,
        job_id: str | None = None,
    ) -> Completion:
        """Exécute une tâche et retourne la réponse complète, métadonnées incluses.

        Lève `BudgetExceeded` si un plafond du jour est atteint (la boucle doit
        alors s'arrêter, pas réessayer), ou `LlmError` si tous les modèles de la
        route ont échoué.
        """
        self.budget.check()

        route = self.config.llm.route(task)
        candidates: list[ModelRef] = [route]
        if route.fallback is not None:
            candidates.append(route.fallback)

        errors: list[str] = []
        #: Vrai tant que tous les échecs rencontrés sont des pannes de
        #: transport : la route, elle, n'a rien démontré.
        only_transient = True
        for candidate in candidates:
            try:
                provider = self._provider(candidate.provider)
            except LlmError as exc:
                # Clé absente ou fournisseur inconnu : on tente le repli.
                errors.append(str(exc))
                only_transient = False
                continue

            completion = None
            rate_limited = 0
            transient = 0
            while True:
                try:
                    completion = provider.complete(
                        model=candidate.model,
                        prompt=prompt,
                        system=system,
                        max_tokens=max_tokens,
                        json_mode=json_mode,
                    )
                    break
                except TransientError as exc:
                    # Le réseau, pas le modèle. Le repli emprunte le même
                    # réseau : y basculer tout de suite ne réglerait rien, et
                    # gaspillerait un quota. On attend et on recommence.
                    transient += 1
                    if transient > MAX_TRANSIENT_RETRIES:
                        errors.append(f"{candidate.provider}/{candidate.model}: {exc}")
                        self._log(task, candidate, job_id, ok=False, error=str(exc))
                        logger.warning(
                            "tâche %s: réseau indisponible pour %s/%s après %d "
                            "reprises — %s",
                            task,
                            candidate.provider,
                            candidate.model,
                            MAX_TRANSIENT_RETRIES,
                            exc,
                        )
                        break
                    delay = TRANSIENT_BACKOFF * 2 ** (transient - 1)
                    logger.info(
                        "tâche %s: %s injoignable (%s), reprise dans %.0fs (%d/%d)",
                        task,
                        candidate.provider,
                        exc,
                        delay,
                        transient,
                        MAX_TRANSIENT_RETRIES,
                    )
                    time.sleep(delay)
                except RateLimited as exc:
                    # La cadence n'est pas une panne : le même appel passera
                    # dans quelques secondes. Basculer tout de suite sur le
                    # repli dépenserait le seul autre quota gratuit pour rien,
                    # et le laisse indisponible quand la panne est réelle.
                    rate_limited += 1
                    delay = exc.retry_after or DEFAULT_RATE_LIMIT_WAIT
                    if (
                        rate_limited > MAX_RATE_LIMIT_RETRIES
                        or delay > MAX_RATE_LIMIT_WAIT
                    ):
                        errors.append(f"{candidate.provider}/{candidate.model}: {exc}")
                        only_transient = False
                        self._log(task, candidate, job_id, ok=False, error=str(exc))
                        logger.warning(
                            "tâche %s: cadence dépassée sur %s/%s, repli éventuel",
                            task,
                            candidate.provider,
                            candidate.model,
                        )
                        break
                    logger.info(
                        "tâche %s: cadence %s/%s atteinte, reprise dans %.1fs",
                        task,
                        candidate.provider,
                        candidate.model,
                        delay,
                    )
                    time.sleep(delay)
                except ProviderError as exc:
                    errors.append(f"{candidate.provider}/{candidate.model}: {exc}")
                    only_transient = False
                    self._log(task, candidate, job_id, ok=False, error=str(exc))
                    logger.warning(
                        "tâche %s: échec %s/%s, repli éventuel",
                        task,
                        candidate.provider,
                        candidate.model,
                    )
                    break

            if completion is None:
                continue

            self._log(
                task,
                candidate,
                job_id,
                tokens_in=completion.tokens_in,
                tokens_out=completion.tokens_out,
            )
            return completion

        cause = (
            "réseau injoignable" if only_transient else "tous les modèles ont échoué"
        )
        raise LlmError(
            f"tâche {task!r}: {cause} — " + " | ".join(errors),
            transient=only_transient,
        )

    def structured(
        self,
        task: str,
        prompt: str,
        schema: type[T],
        *,
        system: str | None = None,
        max_tokens: int = 2048,
        job_id: str | None = None,
    ) -> T:
        """Exécute une tâche et valide la réponse contre un modèle Pydantic.

        Le schéma est décrit dans le prompt et vérifié côté client : cette
        approche fonctionne chez tous les fournisseurs, alors que les modes
        « schéma natif » diffèrent et refusent certaines constructions. En cas de
        sortie invalide, une seule relance est tentée en indiquant l'erreur.
        """
        instructions = (
            "\n\nRéponds UNIQUEMENT avec un objet JSON valide respectant "
            "exactement ce schéma JSON Schema, sans texte autour ni balise de code :\n"
            f"{json.dumps(schema.model_json_schema(), ensure_ascii=False)}"
        )
        attempt_prompt = prompt + instructions
        attempt_tokens = max_tokens
        last_error = ""

        for attempt in range(2):
            completion = self.completion(
                task,
                attempt_prompt,
                system=system,
                max_tokens=attempt_tokens,
                json_mode=True,
                job_id=job_id,
            )
            try:
                return schema.model_validate_json(extract_json(completion.text))
            except (ValidationError, ValueError) as exc:
                last_error = str(exc)
                if completion.truncated:
                    # Le JSON n'est pas fautif, il est coupé : lui redemander de
                    # « corriger son erreur » produirait la même coupure. Seule
                    # la place manque, on la double.
                    last_error = (
                        f"réponse tronquée à {attempt_tokens} tokens — {last_error}"
                    )
                    logger.warning(
                        "tâche %s: réponse tronquée à %d tokens (tentative %d/2)",
                        task,
                        attempt_tokens,
                        attempt + 1,
                    )
                    attempt_tokens *= 2
                    attempt_prompt = (
                        prompt
                        + instructions
                        + "\n\nTa réponse précédente a été coupée avant la fin. "
                        "Sois plus concis : une phrase pour `reason`, au plus "
                        "cinq entrées dans `matched` et `gaps`."
                    )
                    continue
                logger.warning(
                    "tâche %s: sortie JSON invalide (tentative %d/2)", task, attempt + 1
                )
                attempt_prompt = (
                    prompt
                    + instructions
                    + f"\n\nTa réponse précédente était invalide :\n{last_error}\n"
                    "Corrige-la et renvoie uniquement le JSON."
                )

        raise LlmError(
            f"tâche {task!r}: sortie JSON invalide après 2 tentatives — {last_error}"
        )

    # -------------------------------------------------------------------- trace

    def _log(
        self,
        task: str,
        candidate: ModelRef,
        job_id: str | None,
        *,
        tokens_in: int = 0,
        tokens_out: int = 0,
        ok: bool = True,
        error: str | None = None,
    ) -> None:
        cost = estimate_cost(
            self.config.llm.price(candidate.model), tokens_in, tokens_out
        )
        self.budget.record(
            LlmUsage(
                task=task,
                provider=candidate.provider,
                model=candidate.model,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_est=cost,
                job_id=job_id,
                ok=ok,
                error=error,
            )
        )
