"""Passe de rédaction : des offres retenues aux dossiers prêts à valider.

Pour chaque offre en `fit_ok`, dans l'ordre décroissant du score :

1. rédaction de la lettre (LLM payant) ;
2. contrôle déterministe — formules interdites, longueur ;
3. revue LLM (tier gratuit) : cohérence avec le CV, spécificité, ton ;
4. choix du CV à joindre (Python pur, d'après la langue de l'annonce) ;
5. écriture du dossier `outbox/` et passage en `awaiting_user`.

**La passe s'arrête là.** Rien n'est envoyé, aucun navigateur n'est ouvert : la
suite est l'affaire de l'étape 5.

Un seul compteur de reprises, `letter.max_regenerations`, couvre les deux motifs
de réécriture (formule interdite ou revue négative). C'est l'étape payante :
mieux vaut un défaut signalé à l'utilisateur qu'une troisième facture. Une
lettre livrée avec des réserves l'est toujours avec ses réserves visibles —
dans le rapport, dans `preview.html` et dans `review.json`.

La passe est reprenable : la lettre est écrite dans `jobs.jsonl` dès sa
génération, donc une interruption après l'appel payant ne le fait pas repayer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from agent_emploi.agents.letter import LetterAgent
from agent_emploi.agents.review import ReviewAgent
from agent_emploi.config import Config
from agent_emploi.llm.budget import BudgetExceeded
from agent_emploi.llm.router import LlmError
from agent_emploi.models import JobRecord, JobState, Letter, ReviewVerdict
from agent_emploi.outbox import bundle_path, write_bundle
from agent_emploi.profile import Profile
from agent_emploi.store.jobs import JobStore
from agent_emploi.store.seen import SeenStore

logger = logging.getLogger(__name__)

#: États depuis lesquels une offre peut (re)partir en rédaction. `DRAFTED` et
#: `REVIEWED` sont des reprises : une passe interrompue avant l'écriture du
#: dossier a laissé l'offre là, avec sa lettre déjà payée.
DRAFTABLE = frozenset({JobState.FIT_OK, JobState.DRAFTED, JobState.REVIEWED})

#: Au-delà de tant d'échecs LLM d'affilée, c'est la route qui est en cause et
#: non les offres : la passe s'arrête au lieu de brûler un appel par offre.
MAX_CONSECUTIVE_FAILURES = 2


@dataclass
class DraftedApplication:
    """Une candidature préparée, telle qu'elle est remise à l'utilisateur."""

    record: JobRecord
    directory: Path
    #: Réserves restantes : formules interdites, longueur, revue négative. Une
    #: liste non vide ne bloque pas la livraison — elle appelle une relecture.
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.warnings


@dataclass
class DraftReport:
    """Bilan d'une passe de rédaction."""

    candidates: int = 0
    #: Lettres réellement générées (appels payants), reprises comprises.
    generated: int = 0
    #: Lettres reprises d'une passe précédente, donc gratuites.
    resumed: int = 0
    regenerated: int = 0
    reviewed: int = 0
    prepared: list[DraftedApplication] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    #: Motif d'interruption anticipée (plafond de budget, route en panne).
    stopped: str | None = None

    @property
    def flagged(self) -> list[DraftedApplication]:
        """Dossiers livrés avec des réserves — à relire en priorité."""
        return [item for item in self.prepared if not item.ok]


def select_candidates(
    job_store: JobStore, seen: SeenStore, *, limit: int | None = None
) -> list[JobRecord]:
    """Offres prêtes à être rédigées, les mieux notées d'abord.

    La mémoire fait foi sur l'état : `jobs.jsonl` garde toutes les offres
    pré-retenues, y compris celles que le fit-check a ensuite écartées ou que
    l'utilisateur a déjà en main.
    """
    ready = [
        record
        for record in job_store.records()
        if record.fit is not None
        and (entry := seen.get(record.job.id)) is not None
        and entry.state in DRAFTABLE
    ]
    ready.sort(key=lambda record: record.fit.score, reverse=True)
    return ready[:limit] if limit is not None else ready


def compose(
    record: JobRecord,
    *,
    letter_agent: LetterAgent,
    review_agent: ReviewAgent,
    max_regenerations: int,
    report: DraftReport,
) -> tuple[Letter, ReviewVerdict | None, list[str]]:
    """Rédige, contrôle et relit une lettre. Retourne (lettre, revue, réserves).

    Le contrôle déterministe passe avant la revue LLM : il est gratuit, et une
    lettre bourrée de formules interdites ne mérite pas qu'on dépense un appel
    de relecture pour l'apprendre.

    Les reprises des deux motifs puisent dans le même compteur : c'est le nombre
    d'appels payants supplémentaires que l'on s'autorise par offre, quelle qu'en
    soit la raison.
    """
    job, fit = record.job, record.fit
    letter = record.letter
    if letter is not None and record.review is not None and record.review.approved:
        # Reprise après une interruption entre la revue et l'écriture du
        # dossier : tout est déjà payé et validé, il ne reste qu'à livrer.
        return letter, record.review, []
    if letter is None:
        letter = letter_agent.write(job, fit)
        report.generated += 1

    remaining = max_regenerations
    review: ReviewVerdict | None = None

    while True:
        defects = letter_agent.defects(letter)
        if defects and remaining > 0:
            remaining -= 1
            report.regenerated += 1
            report.generated += 1
            letter = letter_agent.write(
                job,
                fit,
                feedback="Ta version précédente a été rejetée : "
                + " ; ".join(defects)
                + ". Réécris la lettre entière.",
            )
            continue
        if defects:
            return letter, review, defects

        review = review_agent.review(job, letter)
        report.reviewed += 1
        if review.approved:
            return letter, review, []
        if remaining <= 0:
            return letter, review, _review_warnings(review)

        remaining -= 1
        report.regenerated += 1
        report.generated += 1
        letter = letter_agent.write(job, fit, feedback=review_agent.feedback(review))


def _review_warnings(review: ReviewVerdict) -> list[str]:
    """Réserves d'une revue négative, telles qu'affichées à l'utilisateur."""
    warnings = [f"invention probable : {claim}" for claim in review.unsupported_claims]
    warnings += [f"relecture : {issue}" for issue in review.issues]
    return warnings or ["relecture : lettre refusée sans motif détaillé"]


def run_draft(
    config: Config,
    *,
    seen: SeenStore,
    job_store: JobStore,
    profile: Profile,
    letter_agent: LetterAgent,
    review_agent: ReviewAgent,
    limit: int | None = None,
    record: bool = True,
    candidates: list[JobRecord] | None = None,
) -> DraftReport:
    """Prépare les dossiers des offres retenues.

    `record=False` fait une passe à blanc : les appels LLM ont bien lieu (c'est
    le seul moyen de juger une lettre), mais ni la mémoire, ni `jobs.jsonl`, ni
    `outbox/` ne sont touchés.

    `candidates` court-circuite la sélection dans les mémoires. C'est ce dont a
    besoin une passe à blanc enchaînée : le filtrage n'y a rien persisté, donc
    les offres qu'il vient de retenir n'existent nulle part ailleurs que dans
    son rapport.
    """
    report = DraftReport()
    if candidates is None:
        candidates = select_candidates(job_store, seen, limit=limit)
    elif limit is not None:
        candidates = candidates[:limit]
    report.candidates = len(candidates)
    consecutive_failures = 0

    for job_record in candidates:
        job = job_record.job
        if job_record.letter is not None:
            report.resumed += 1

        try:
            letter, review, warnings = compose(
                job_record,
                letter_agent=letter_agent,
                review_agent=review_agent,
                max_regenerations=config.letter.max_regenerations,
                report=report,
            )
        except BudgetExceeded as exc:
            # Plafond atteint : la passe s'arrête proprement. Les offres non
            # traitées restent en `fit_ok` et seront reprises telles quelles.
            report.stopped = str(exc)
            logger.warning("passe interrompue: %s", exc)
            break
        except LlmError as exc:
            report.errors.append(f"rédaction {job.title}: {exc}")
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                cause = (
                    "réseau injoignable"
                    if exc.transient
                    else "route letter ou review inutilisable"
                )
                report.stopped = (
                    f"{consecutive_failures} échecs LLM consécutifs — {cause}"
                )
                logger.warning("passe interrompue: %s", report.stopped)
                break
            continue

        consecutive_failures = 0

        cv_pdf = profile.cv_pdf(letter.language)
        if cv_pdf is None:
            warnings = warnings + [
                f"aucun CV à joindre pour la langue « {letter.language} »"
            ]

        drafted = job_record.model_copy(update={"letter": letter, "review": review})

        if not record:
            report.prepared.append(
                DraftedApplication(
                    record=drafted,
                    directory=bundle_path(config.paths.outbox, drafted),
                    warnings=warnings,
                )
            )
            continue

        # La lettre est persistée avant l'écriture du dossier : une interruption
        # entre les deux ne doit pas faire repayer l'appel.
        saved = job_store.save(job, letter=letter, review=review)
        entry = seen.get(job.id)
        if entry is not None and entry.state is JobState.FIT_OK:
            seen.advance(job.id, JobState.DRAFTED, f"lettre:{letter.word_count} mots")
        seen.advance(
            job.id,
            JobState.REVIEWED,
            "revue:ok" if review is not None and review.approved else "revue:réserves",
        )

        try:
            directory = write_bundle(config.paths.outbox, saved, cv_pdf)
        except OSError as exc:
            # L'offre reste en `reviewed` : la prochaine passe reprendra à
            # l'écriture du dossier, sans réécrire la lettre.
            report.errors.append(f"dossier {job.title}: {exc}")
            logger.warning("écriture du dossier échouée (%s): %s", job.title, exc)
            continue

        saved = job_store.save(job, outbox=str(directory))
        seen.advance(job.id, JobState.AWAITING_USER, f"dossier:{directory.name}")
        report.prepared.append(
            DraftedApplication(record=saved, directory=directory, warnings=warnings)
        )

    return report
