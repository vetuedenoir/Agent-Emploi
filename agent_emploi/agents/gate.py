"""Porte Jev — l'étage qui décide si une offre mérite le fit-check LLM.

Mesuré sur les 202 premières offres notées : le score lexical moyen des offres
retenues par le fit-check (0,101) ne se distingue pas de celui des offres
refusées (0,105), et 9 offres sur 10 envoyées au LLM en revenaient refusées —
pour l'essentiel sur l'expérience exigée ou le contrat, pas sur le sujet. Le
titre ayant déjà été filtré, toutes ces offres parlent d'IA : un score de
proximité thématique, lexical ou par embeddings, ne peut pas les départager.

Jev répond à des questions typées, ce qui est exactement ce qu'il faut ici :
une probabilité que l'expérience exigée soit bloquante, une autre que le
contrat soit compatible, un score d'adéquation. Un seul appel pour les trois,
environ 0,0001 $ par offre.

La porte ne rédige rien : `matched`, `gaps` et `reason` restent au fit-check,
qui ne voit plus que les offres qu'elle a laissées passer.
"""

from __future__ import annotations

import hashlib
import logging
import os

from agent_emploi.config import Config, GateConfig, Pricing
from agent_emploi.filters import LexicalScorer
from agent_emploi.llm.budget import BudgetTracker, estimate_cost
from agent_emploi.llm.jev import (
    MAX_STATE_CHARS,
    JevClient,
    JevError,
    JevUnavailable,
    serialized_length,
)
from agent_emploi.models import GateVerdict, Job, LlmUsage

logger = logging.getLogger(__name__)

TASK = "gate"
PROVIDER = "jev"
API_KEY_ENV = "JEVMODEL_API_KEY"

#: Le CV partage l'état avec l'annonce. Au-delà, il mangerait la place de la
#: description, qui est ce que la porte doit lire.
MAX_CV_CHARS = 4000

#: Échelle d'adéquation, calquée sur le barème du fit-check (`agents/fit.py`).
#: Jev renvoie une position fractionnaire : 0 = premier niveau.
ADEQUATION_LEVELS = [
    "hors périmètre",
    "recouvrement partiel : exigences structurantes manquantes",
    "candidature défendable : une ou deux exigences importantes non couvertes",
    "adéquation forte : l'essentiel est couvert, niveau d'expérience compris",
]

ADEQUATION = (
    "Évalue l'adéquation entre le candidat (`cv`) et l'offre (`offre`), avec "
    "exigence : il s'agit d'éviter au candidat des candidatures perdues "
    "d'avance, pas de l'encourager. Juge uniquement sur ce qui figure dans le CV "
    "et dans l'annonce, sans rien supposer."
)

EXPERIENCE = (
    "L'annonce exige-t-elle une expérience professionnelle nettement supérieure "
    "à celle que montre le CV (nombre d'années, niveau confirmé ou senior, poste "
    "équivalent déjà occupé) ? Un stage ou une alternance ouverts aux étudiants "
    "n'exigent pas d'expérience ; des compétences techniques demandées ne sont "
    "pas une exigence d'expérience."
)

__all__ = ["GateAgent", "JevError", "JevUnavailable", "build_state", "decide"]


def build_state(job: Job, cv_text: str) -> dict:
    """L'état envoyé à Jev : CV et offre, description tronquée pour tenir.

    La limite porte sur la sérialisation JSON, où un retour à la ligne compte
    deux caractères : on mesure donc l'état réel plutôt que la description.
    """
    offre = {"titre": job.title, "entreprise": job.company}
    for key, value in (
        ("contrat", job.contract),
        ("lieu", job.location),
        ("télétravail", job.remote),
    ):
        if value:
            offre[key] = value
    offre["description"] = ""
    state = {"cv": cv_text[:MAX_CV_CHARS], "offre": offre}

    room = MAX_STATE_CHARS - serialized_length(state)
    description = job.description[: max(room, 0)]
    offre["description"] = description
    while serialized_length(state) > MAX_STATE_CHARS and description:
        overflow = serialized_length(state) - MAX_STATE_CHARS
        description = description[: -max(overflow, 1)]
        offre["description"] = description
    return state


def questions(contracts: list[str]) -> dict[str, dict]:
    """Les questions posées, une seule requête pour toutes.

    La question du contrat n'est posée que si la recherche en vise : sans liste,
    tout contrat convient et la question ne mesurerait rien.
    """
    asked: dict[str, dict] = {
        "adequation": {
            "type": "score",
            "instructions": ADEQUATION,
            "criteria": ADEQUATION_LEVELS,
        },
        "experience_bloquante": {"type": "noul", "instructions": EXPERIENCE},
    }
    if contracts:
        asked["contrat_compatible"] = {
            "type": "noul",
            "instructions": (
                "Le contrat réellement proposé par l'annonce est-il de l'un de ces "
                f"types : {', '.join(contracts)} ? Juge sur ce que l'annonce "
                "propose, pas sur ce que cherche le candidat."
            ),
        }
    return asked


def decide(
    config: GateConfig,
    adequation: float,
    experience_blocking: float,
    contract_ok: float | None,
) -> GateVerdict:
    """Applique les seuils. Le motif retenu est le plus décisif des trois.

    Contrat, puis expérience, puis adéquation : les deux premiers sont des
    exigences binaires qu'aucune compétence ne compense, le troisième un
    jugement d'ensemble.
    """
    reason = None
    if contract_ok is not None and contract_ok < config.min_contract:
        reason = f"jev:contrat({contract_ok:.2f})"
    elif experience_blocking >= config.max_blocking:
        reason = f"jev:experience({experience_blocking:.2f})"
    elif adequation < config.min_score:
        reason = f"jev:adequation {adequation:.1f}<{config.min_score:.1f}"
    return GateVerdict(
        adequation=adequation,
        experience_blocking=experience_blocking,
        contract_ok=contract_ok,
        passed=reason is None,
        reason=reason,
    )


class GateAgent:
    """Pose les questions à Jev, journalise le coût, tranche selon les seuils."""

    def __init__(
        self,
        client: JevClient,
        budget: BudgetTracker,
        cv_text: str | list[str],
        config: GateConfig,
        *,
        contracts: list[str],
        pricing: Pricing,
    ) -> None:
        self.client = client
        self.budget = budget
        self.config = config
        self.contracts = contracts
        self.pricing = pricing
        self.cv_texts = [cv_text] if isinstance(cv_text, str) else list(cv_text)
        for text in self.cv_texts:
            if len(text) > MAX_CV_CHARS:
                logger.warning(
                    "CV de %d caractères tronqué à %d pour la porte Jev",
                    len(text),
                    MAX_CV_CHARS,
                )
        #: Avec plusieurs variantes, la porte juge avec la plus proche de
        #: l'offre au sens lexical : elle écarte l'inaccessible, le choix fin
        #: de la variante revient au fit-check.
        self._scorer = LexicalScorer(self.cv_texts) if len(self.cv_texts) > 1 else None

    def cv_for(self, job: Job) -> str:
        """Le CV présenté à Jev pour cette offre."""
        if self._scorer is None:
            return self.cv_texts[0]
        return self.cv_texts[self._scorer.best(f"{job.title}\n{job.description}")]

    @classmethod
    def from_config(
        cls,
        config: Config,
        cv_text: str | list[str],
        budget: BudgetTracker,
        *,
        force: bool = False,
    ) -> GateAgent | None:
        """La porte si elle est activée et sa clé présente, sinon `None`.

        Une clé absente n'arrête rien : l'étage est simplement sauté, et le
        fit-check tranche seul, comme avant la porte. `force` ignore
        `gate.enabled`, pour calibrer la porte avant de l'activer.
        """
        if not (config.gate.enabled or force):
            return None
        key = os.environ.get(API_KEY_ENV)
        if not key:
            logger.warning("%s absente — porte Jev ignorée", API_KEY_ENV)
            return None
        client = JevClient(key)
        return cls(
            client,
            budget,
            cv_text,
            config.gate,
            contracts=config.search.contracts,
            pricing=config.llm.price(client.model),
        )

    def evaluate(self, job: Job) -> GateVerdict:
        """Un appel Jev. Propage `BudgetExceeded`, `JevUnavailable` et `JevError`.

        Comme pour le fit-check, la distinction est laissée à l'orchestrateur :
        la première arrête la passe, la deuxième la porte, la troisième ne
        concerne que cette offre.
        """
        self.budget.check()
        state = build_state(job, self.cv_for(job))
        # Même offre, même état : même clé. Une reprise après une coupure n'est
        # pas facturée deux fois ; un CV modifié, lui, donne une nouvelle clé.
        digest = hashlib.sha256(repr(state).encode()).hexdigest()[:12]
        try:
            response = self.client.ask(
                state,
                questions(self.contracts),
                idempotency_key=f"gate-{job.id}-{digest}",
            )
        except (JevError, JevUnavailable) as exc:
            self._log(job, ok=False, error=str(exc))
            raise
        self._log(job, tokens_in=response.tokens_in)

        try:
            answers = response.answers
            adequation = float(answers["adequation"]["score"])
            experience = float(answers["experience_bloquante"]["noul"])
            contract = (
                float(answers["contrat_compatible"]["noul"])
                if "contrat_compatible" in answers
                else None
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise JevError(f"jev: réponse incomplète — {response.answers!r}") from exc

        verdict = decide(self.config, adequation, experience, contract)
        logger.info(
            "porte %s — %s (adéquation %.1f, expérience %.2f, contrat %s)",
            job.title,
            "passe" if verdict.passed else verdict.reason,
            adequation,
            experience,
            "—" if contract is None else f"{contract:.2f}",
        )
        return verdict

    def _log(
        self,
        job: Job,
        *,
        tokens_in: int = 0,
        ok: bool = True,
        error: str | None = None,
    ) -> None:
        self.budget.record(
            LlmUsage(
                task=TASK,
                provider=PROVIDER,
                model=self.client.model,
                tokens_in=tokens_in,
                cost_est=estimate_cost(self.pricing, tokens_in, 0),
                job_id=job.id,
                ok=ok,
                error=error,
            )
        )
