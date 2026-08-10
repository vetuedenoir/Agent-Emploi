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
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from agent_emploi.config import Config, ModelRef, api_key
from agent_emploi.llm.budget import BudgetTracker, estimate_cost
from agent_emploi.llm.providers import REGISTRY, Provider, ProviderError
from agent_emploi.models import LlmUsage

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

#: Bloc de code Markdown éventuellement enroulé autour du JSON par le modèle.
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class LlmError(RuntimeError):
    """Échec définitif : le modèle principal et son repli ont tous deux échoué."""


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
        """Exécute une tâche et retourne le texte brut de la réponse.

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
        for candidate in candidates:
            try:
                provider = self._provider(candidate.provider)
            except LlmError as exc:
                # Clé absente ou fournisseur inconnu : on tente le repli.
                errors.append(str(exc))
                continue

            try:
                completion = provider.complete(
                    model=candidate.model,
                    prompt=prompt,
                    system=system,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                )
            except ProviderError as exc:
                errors.append(f"{candidate.provider}/{candidate.model}: {exc}")
                self._log(task, candidate, job_id, ok=False, error=str(exc))
                logger.warning(
                    "tâche %s: échec %s/%s, repli éventuel",
                    task,
                    candidate.provider,
                    candidate.model,
                )
                continue

            self._log(
                task,
                candidate,
                job_id,
                tokens_in=completion.tokens_in,
                tokens_out=completion.tokens_out,
            )
            return completion.text

        raise LlmError(f"tâche {task!r}: tous les modèles ont échoué — " + " | ".join(errors))

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
        last_error = ""

        for attempt in range(2):
            raw = self.complete(
                task,
                attempt_prompt,
                system=system,
                max_tokens=max_tokens,
                json_mode=True,
                job_id=job_id,
            )
            try:
                return schema.model_validate_json(extract_json(raw))
            except (ValidationError, ValueError) as exc:
                last_error = str(exc)
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
