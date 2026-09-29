"""Ce que l'interface peut modifier, et à quelles conditions.

Chaque action appelle la fonction que la CLI appelle déjà : `review_cli.approve`
et `reject`, `mark_submitted`, l'archivage. Ce module n'ajoute que la garde : à
quel état l'action a un sens, et quel message rendre sinon. Une action refusée
lève `ActionRefused`, que la route traduit en 409 : la page affichée date
peut-être d'avant une passe CLI qui a fait avancer l'offre entre-temps.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from agent_emploi import manual
from agent_emploi.agents.fit import FitAgent
from agent_emploi.config import Config
from agent_emploi.draft import deliver_own_letter
from agent_emploi.llm.budget import BudgetExceeded
from agent_emploi.llm.router import LlmError
from agent_emploi.models import JobRecord, JobState, SeenEntry, UserDecision, utcnow
from agent_emploi.outbox import letter_markdown, refresh_preview
from agent_emploi.profile import BannedPhrases
from agent_emploi.review_cli import Pending, approve, concerns, reject
from agent_emploi.sent import SENDABLE, mark_interview, mark_submitted
from agent_emploi.store.archive import ARCHIVABLE, archive_and_save
from agent_emploi.store.jobs import JobStore
from agent_emploi.store.seen import SeenStore
from agent_emploi.web.passes import PassBusy, PassRun, PassRunner

#: États où la lettre peut encore changer. Au-delà, elle est partie : la
#: modifier ici ne changerait pas ce qui a été envoyé.
LETTER_EDITABLE = frozenset(
    {
        JobState.DRAFTED,
        JobState.REVIEWED,
        JobState.AWAITING_USER,
        JobState.APPROVED,
    }
)

#: États d'un dossier qu'on peut encore abandonner. Plus tôt, l'offre n'a pas de
#: dossier et le rejet appartient aux passes automatiques.
REJECTABLE = frozenset(
    {
        JobState.AWAITING_USER,
        JobState.APPROVED,
        JobState.PREFILLED,
        JobState.HANDOFF,
    }
)


class ActionRefused(RuntimeError):
    """L'action n'a pas de sens pour l'offre dans son état actuel."""


@dataclass(frozen=True)
class Allowed:
    """Les actions proposées sur une fiche, selon l'état de l'offre."""

    edit_letter: bool = False
    #: Offre retenue sans lettre : l'utilisateur peut l'écrire lui-même au lieu
    #: d'attendre la passe `draft`.
    write_letter: bool = False
    decide: bool = False
    reject: bool = False
    mark_sent: bool = False
    mark_interview: bool = False
    archive: bool = False
    #: Offre ajoutée à la main, à faire passer au fit-check (ou à y repasser).
    prepare: bool = False


def allowed(entry: SeenEntry | None, record: JobRecord | None) -> Allowed:
    if entry is None or record is None:
        return Allowed()
    state = entry.state
    has_bundle = bool(record.outbox) and Path(record.outbox).exists()
    tracked = state is JobState.TRACKED
    return Allowed(
        edit_letter=record.letter is not None
        and state in LETTER_EDITABLE
        and record.archive is None,
        write_letter=state is JobState.FIT_OK
        and record.fit is not None
        and record.letter is None,
        decide=state is JobState.AWAITING_USER and has_bundle,
        reject=(state in REJECTABLE and has_bundle) or tracked,
        mark_sent=state in SENDABLE,
        mark_interview=state is JobState.SUBMITTED,
        archive=state in ARCHIVABLE and has_bundle and record.archive is None,
        prepare=manual.preparable(entry, record.job, has_fit=record.fit is not None),
    )


def _load(job_id: str, seen: SeenStore, jobs: JobStore) -> tuple[SeenEntry, JobRecord]:
    entry, record = seen.get(job_id), jobs.get(job_id)
    if entry is None or record is None:
        raise ActionRefused("offre inconnue ou jamais retenue")
    return entry, record


def save_letter(
    job_id: str, text: str, *, seen: SeenStore, jobs: JobStore, config: Config
) -> JobRecord:
    """Enregistre une lettre corrigée, là où chaque étape ira la lire.

    `lettre.md` d'abord, parce qu'il fait foi : `review_cli.approve` le relit,
    et c'est lui qui part à l'archivage. `jobs.jsonl` ensuite, avec
    `edited=True`. Les formules interdites sont revérifiées :
    une correction à la main peut en réintroduire une.
    """
    entry, record = _load(job_id, seen, jobs)
    if not allowed(entry, record).edit_letter:
        raise ActionRefused(
            f"lettre non modifiable à l'état « {entry.state.value} »"
        )
    text = text.replace("\r\n", "\n").strip()
    if not text:
        raise ActionRefused("une lettre vide ne remplace pas la précédente")

    banned = BannedPhrases.load(config.profile.banned_phrases)
    letter = record.letter.model_copy(
        update={"text": text, "banned_hits": banned.find(text), "edited": True}
    )
    updated = record.model_copy(update={"letter": letter})
    if record.outbox and Path(record.outbox).exists():
        directory = Path(record.outbox)
        (directory / "lettre.md").write_text(letter_markdown(updated), encoding="utf-8")
        refresh_preview(directory, updated)
    return jobs.save(record.job, letter=letter)


def write_own_letter(
    job_id: str, text: str, *, seen: SeenStore, jobs: JobStore, config: Config
) -> Path:
    """Livre la lettre que l'utilisateur a écrite, dossier compris.

    L'offre rejoint `awaiting_user` comme après `draft` : la décision reste une
    étape à part, avec les réserves (longueur, formules, CV) sous les yeux.
    """
    entry, record = _load(job_id, seen, jobs)
    if not allowed(entry, record).write_letter:
        raise ActionRefused(
            f"seule une offre retenue sans lettre s'écrit à la main "
            f"(état « {entry.state.value} »)"
        )
    text = text.replace("\r\n", "\n").strip()
    if not text:
        raise ActionRefused("la lettre est vide")
    try:
        drafted = deliver_own_letter(config, record, text, seen=seen, job_store=jobs)
    except OSError as exc:
        raise ActionRefused(f"dossier impossible à écrire : {exc}") from exc
    return drafted.directory


def _pending(record: JobRecord, config: Config) -> Pending:
    directory = Path(record.outbox)
    return Pending(
        record=record, directory=directory, concerns=concerns(record, directory, config)
    )


def decide(
    job_id: str,
    *,
    approved: bool,
    note: str | None,
    seen: SeenStore,
    jobs: JobStore,
    config: Config,
) -> None:
    """Approuve ou rejette un dossier, exactement comme `review` le ferait."""
    entry, record = _load(job_id, seen, jobs)
    permitted = allowed(entry, record)
    if approved and not permitted.decide:
        raise ActionRefused(
            f"seul un dossier en attente s'approuve (état « {entry.state.value} »)"
        )
    if not approved and not permitted.reject:
        raise ActionRefused(f"rien à rejeter à l'état « {entry.state.value} »")

    if entry.state is JobState.TRACKED:
        # Pas de dossier : la décision ne vit que dans les journaux.
        jobs.save(
            record.job,
            decision=UserDecision(decision="rejected", at=utcnow(), note=note),
        )
        seen.transition(job_id, JobState.REJECTED, f"utilisateur:{note or 'abandonnée'}"[:200])
        return

    item = _pending(record, config)
    if approved:
        banned = BannedPhrases.load(config.profile.banned_phrases)
        approve(item, seen=seen, job_store=jobs, banned=banned, config=config, note=note)
    else:
        reject(item, seen=seen, job_store=jobs, note=note)


def declare_sent(
    job_id: str, *, note: str | None, seen: SeenStore, jobs: JobStore
) -> None:
    """Prend acte d'un envoi fait à la main — le programme n'envoie jamais rien."""
    entry, record = _load(job_id, seen, jobs)
    if not allowed(entry, record).mark_sent:
        raise ActionRefused(
            f"rien à déclarer envoyé à l'état « {entry.state.value} »"
        )
    mark_submitted(job_id, seen=seen, job_store=jobs, note=note)


def declare_interview(
    job_id: str,
    *,
    day: str | None,
    note: str | None,
    seen: SeenStore,
    jobs: JobStore,
) -> None:
    """Prend acte d'un entretien obtenu après l'envoi.

    `day` est la date saisie (`AAAA-MM-JJ`), facultative.
    """
    entry, record = _load(job_id, seen, jobs)
    if not allowed(entry, record).mark_interview:
        raise ActionRefused(
            f"seule une candidature envoyée débouche sur un entretien "
            f"(état « {entry.state.value} »)"
        )
    at = None
    if day:
        try:
            # Midi local : la date reste la même une fois convertie en UTC.
            at = datetime.fromisoformat(day).replace(hour=12).astimezone()
        except ValueError as exc:
            raise ActionRefused(f"date d'entretien illisible : {day}") from exc
    mark_interview(job_id, seen=seen, job_store=jobs, at=at, note=note)


def start_prepare(
    job_id: str,
    *,
    seen: SeenStore,
    jobs: JobStore,
    config: Config,
    runner: PassRunner,
    fit_agent: Callable[[], FitAgent],
) -> PassRun:
    """Lance le fit-check d'une offre manuelle en arrière-plan.

    L'offre est vérifiée ici, avant de partir : un refus doit s'afficher sur la
    fiche, pas au fond du journal d'une passe. La passe revérifie de toute
    façon, sur les journaux relus.
    """
    entry, record = _load(job_id, seen, jobs)
    if not allowed(entry, record).prepare:
        raise ActionRefused(
            "seule une offre ajoutée à la main, avec sa description, se prépare"
        )
    try:
        return runner.start_task(
            "prepare",
            "Fit-check manuel",
            prepare_task(job_id, config=config, fit_agent=fit_agent),
            link=(f"/offres/{job_id}", "Voir la fiche de l'offre"),
        )
    except PassBusy as exc:
        raise ActionRefused(str(exc)) from exc


def prepare_task(
    job_id: str, *, config: Config, fit_agent: Callable[[], FitAgent]
) -> Callable[[], int]:
    """La passe elle-même : fit-check, puis `FIT_OK` quel que soit le verdict.

    Elle relit les journaux au lieu d'emprunter les stores du serveur, comme
    une passe CLI : le serveur les rechargera en voyant les fichiers changer.
    `fit_agent` est une fabrique, appelée ici : construire le routeur lit le CV
    et les clés, ce qui n'a rien à faire dans la requête.
    """

    def run() -> int:
        seen = SeenStore(
            config.paths.seen_file, dedup_window_days=config.filters.dedup_window_days
        )
        jobs = JobStore(config.paths.jobs_file)
        record = jobs.get(job_id)
        if record is not None:
            print(f"Fit-check : {record.job.company} — {record.job.title}")
        try:
            verdict = manual.prepare(job_id, seen=seen, jobs=jobs, fit_agent=fit_agent())
        except (manual.NotPreparable, FileNotFoundError) as exc:
            print(f"Impossible : {exc}")
            return 1
        except (LlmError, BudgetExceeded) as exc:
            print(f"Fit-check impossible pour l'instant : {exc}")
            print("L'offre reste enregistrée ; relancez l'évaluation depuis sa fiche.")
            return 1

        print(f"\n[{verdict.score:>3}] {verdict.verdict} · langue {verdict.language}")
        print(f"  {verdict.reason}")
        if verdict.matched:
            print(f"  atouts : {', '.join(verdict.matched)}")
        if verdict.gaps:
            print(f"  manques : {', '.join(verdict.gaps)}")
        print("\nOffre retenue : la prochaine passe `draft` rédigera sa lettre.")
        return 0

    return run


def archive(job_id: str, *, seen: SeenStore, jobs: JobStore, config: Config) -> Path:
    """Classe le dossier d'une candidature close dans `applications/`."""
    entry, record = _load(job_id, seen, jobs)
    if not allowed(entry, record).archive:
        raise ActionRefused(
            "seule une candidature envoyée ou rejetée, avec son dossier, s'archive"
        )
    try:
        archived = archive_and_save(config, record, entry, job_store=jobs)
    except OSError as exc:
        raise ActionRefused(f"archivage impossible : {exc}") from exc
    return archived.directory
