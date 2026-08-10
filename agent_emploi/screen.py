"""Passe de filtrage : des offres découvertes aux offres notées.

Enchaîne les trois étages dans l'ordre du moins cher au plus cher, chacun ne
voyant que ce que le précédent a laissé passer :

1. `screen_metadata` — gratuit, aucune requête.
2. `enrich` puis `screen_content` — une requête HTTP par offre.
3. `FitAgent.evaluate` — un appel LLM par offre.

Chaque offre est transitionnée dans `seen.jsonl` avec son motif, et les offres
retenues sont conservées en entier dans `jobs.jsonl` pour l'étape de rédaction.
La passe est donc reprenable : relancée, elle ne repasse pas sur ce qui a déjà
été tranché.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Mapping

from agent_emploi.agents.fit import FitAgent
from agent_emploi.config import Config
from agent_emploi.filters import LexicalScorer, screen_content, screen_metadata
from agent_emploi.llm.budget import BudgetExceeded
from agent_emploi.llm.router import LlmError
from agent_emploi.models import FitVerdict, Job, JobState
from agent_emploi.sources import REGISTRY, JobSource
from agent_emploi.store.jobs import JobStore
from agent_emploi.store.seen import SeenStore

logger = logging.getLogger(__name__)

#: Au-delà de tant d'échecs LLM d'affilée, c'est la route qui est en cause et non
#: les offres : la passe s'arrête au lieu de brûler un appel par offre.
MAX_CONSECUTIVE_FAILURES = 3


@dataclass
class ScreenReport:
    """Bilan d'une passe de filtrage, pour l'affichage et le réglage des seuils."""

    examined: int = 0
    #: Nombre d'offres ayant demandé une requête de détail.
    enriched: int = 0
    #: Offres ayant franchi les deux étages déterministes.
    prescreened: int = 0
    #: Offres réellement soumises au LLM.
    evaluated: int = 0
    accepted: list[tuple[Job, FitVerdict]] = field(default_factory=list)
    refused: list[tuple[Job, FitVerdict]] = field(default_factory=list)
    #: Motifs de rejet agrégés par catégorie — la matière première du réglage.
    reasons: Counter[str] = field(default_factory=Counter)
    errors: list[str] = field(default_factory=list)
    #: Motif d'interruption anticipée (plafond de budget), le cas échéant.
    stopped: str | None = None

    def note_rejection(self, reason: str) -> None:
        self.reasons[reason.split(":", 1)[0]] += 1


def load_cv_text(config: Config) -> str:
    """Lit le CV texte, seule entrée du scoring lexical et des prompts."""
    path = config.profile.cv_markdown
    if not path.exists():
        raise FileNotFoundError(
            f"CV texte introuvable: {path} — requis pour le filtrage lexical"
        )
    return path.read_text(encoding="utf-8")


def run_screen(
    config: Config,
    jobs: list[Job],
    *,
    seen: SeenStore,
    job_store: JobStore,
    fit_agent: FitAgent,
    cv_text: str,
    sources: Mapping[str, JobSource] | None = None,
    record: bool = True,
) -> ScreenReport:
    """Filtre, enrichit puis note une liste d'offres.

    `sources` permet d'injecter des sources déjà ouvertes (et de tester sans
    réseau) ; sinon elles sont instanciées à la demande, une par nom rencontré,
    et refermées à la fin.
    """
    report = ScreenReport(examined=len(jobs))
    scorer = LexicalScorer(cv_text)
    opened: dict[str, JobSource] = {}
    consecutive_failures = 0

    def resolve(name: str) -> JobSource | None:
        """Source pour un nom donné, injectée ou instanciée une seule fois."""
        if sources is not None:
            return sources.get(name)
        if name not in opened:
            source_cls = REGISTRY.get(name)
            if source_cls is None:
                return None
            opened[name] = source_cls()
        return opened[name]

    def reject(job: Job, reason: str) -> None:
        report.note_rejection(reason)
        _transition(seen, job, JobState.REJECTED, reason, record=record)

    try:
        for job in jobs:
            screening = screen_metadata(job, config.filters, config.search)
            if not screening.passed:
                reject(job, screening.reason)
                continue

            if not job.is_enriched:
                source = resolve(job.source)
                if source is None:
                    report.errors.append(f"source inconnue: {job.source}")
                    continue
                report.enriched += 1
                try:
                    job = source.enrich(job)
                except Exception as exc:  # noqa: BLE001 — une source ne doit pas tuer la passe
                    report.errors.append(f"enrichissement {job.url}: {exc}")
                    logger.warning("enrichissement échoué (%s): %s", job.url, exc)
                    continue

            screening = screen_content(job, config.filters, scorer)
            if not screening.passed:
                reject(job, screening.reason)
                continue

            report.prescreened += 1
            _transition(seen, job, JobState.PRESCREENED, screening.reason, record=record)
            if record:
                job_store.save(job, lexical_score=screening.lexical_score)

            try:
                verdict = fit_agent.evaluate(job)
            except BudgetExceeded as exc:
                # Plafond atteint : on arrête la passe proprement. Les offres
                # déjà en PRESCREENED seront reprises à la prochaine.
                report.stopped = str(exc)
                logger.warning("passe interrompue: %s", exc)
                break
            except LlmError as exc:
                # Échec sur une offre seulement : elle reste en PRESCREENED et
                # sera réévaluée plus tard.
                report.errors.append(f"fit-check {job.title}: {exc}")
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    # Toutes les offres échouent d'affilée : c'est la route qui
                    # est cassée (clé absente, fournisseur en panne), pas les
                    # offres. Insister coûterait un appel par offre pour rien.
                    report.stopped = (
                        f"{consecutive_failures} échecs LLM consécutifs — "
                        "route fit_check inutilisable"
                    )
                    logger.warning("passe interrompue: %s", report.stopped)
                    break
                continue

            consecutive_failures = 0

            report.evaluated += 1
            if record:
                job_store.save(job, fit=verdict, lexical_score=screening.lexical_score)

            if fit_agent.accepts(verdict):
                report.accepted.append((job, verdict))
                _transition(
                    seen,
                    job,
                    JobState.FIT_OK,
                    f"fit:{verdict.verdict}({verdict.score})",
                    record=record,
                )
            else:
                reason = fit_agent.rejection_reason(verdict)
                report.refused.append((job, verdict))
                report.note_rejection(reason)
                _transition(seen, job, JobState.REJECTED, reason, record=record)
    finally:
        for source in opened.values():
            close = getattr(source, "close", None)
            if close is not None:
                close()

    report.accepted.sort(key=lambda pair: pair[1].score, reverse=True)
    return report


def _transition(
    seen: SeenStore, job: Job, target: JobState, reason: str, *, record: bool
) -> None:
    """Persiste un changement d'état, sauf en passe à blanc.

    `SeenStore.advance` absorbe les transitions impossibles : le filtrage est
    rejouable, et la mémoire fait foi.
    """
    if not record:
        return
    seen.advance(job.id, target, reason or None)
