"""Ce que l'interface peut modifier, et à quelles conditions.

Chaque action appelle la fonction que la CLI appelle déjà : `review_cli.approve`
et `reject`, `mark_submitted`, l'archivage. Ce module n'ajoute que la garde : à
quel état l'action a un sens, et quel message rendre sinon. Une action refusée
lève `ActionRefused`, que la route traduit en 409 : la page affichée date
peut-être d'avant une passe CLI qui a fait avancer l'offre entre-temps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agent_emploi.apply.runner import mark_submitted
from agent_emploi.config import Config
from agent_emploi.models import JobRecord, JobState, SeenEntry
from agent_emploi.outbox import letter_markdown, refresh_preview
from agent_emploi.profile import BannedPhrases
from agent_emploi.review_cli import Pending, approve, concerns, reject
from agent_emploi.store.archive import ARCHIVABLE, archive_and_save
from agent_emploi.store.jobs import JobStore
from agent_emploi.store.seen import SeenStore

#: États où la lettre peut encore changer. Au-delà, elle a été collée dans un
#: formulaire : la modifier ici ne changerait pas ce qui est parti.
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

#: États d'où l'on peut déclarer un envoi : le formulaire a été rempli, ou la
#: main a été rendue. `mark_submitted` exige un résultat de remplissage.
SENDABLE = frozenset({JobState.PREFILLED, JobState.HANDOFF})


class ActionRefused(RuntimeError):
    """L'action n'a pas de sens pour l'offre dans son état actuel."""


@dataclass(frozen=True)
class Allowed:
    """Les actions proposées sur une fiche, selon l'état de l'offre."""

    edit_letter: bool = False
    decide: bool = False
    reject: bool = False
    mark_sent: bool = False
    archive: bool = False
    #: Commande à copier pour l'étape 6, qui reste en CLI : elle ouvre un
    #: navigateur visible et peut demander un mot de passe à l'invite.
    apply_command: str | None = None


def allowed(entry: SeenEntry | None, record: JobRecord | None) -> Allowed:
    if entry is None or record is None:
        return Allowed()
    state = entry.state
    has_bundle = bool(record.outbox) and Path(record.outbox).exists()
    return Allowed(
        edit_letter=record.letter is not None
        and state in LETTER_EDITABLE
        and record.archive is None,
        decide=state is JobState.AWAITING_USER and has_bundle,
        reject=state in REJECTABLE and has_bundle,
        mark_sent=state in SENDABLE and record.application is not None,
        archive=state in ARCHIVABLE and has_bundle and record.archive is None,
        apply_command=(
            f"python -m agent_emploi apply {record.job.id[:8]}"
            if state is JobState.APPROVED
            else None
        ),
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
    `edited=True`, pour l'étape 6. Les formules interdites sont revérifiées :
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
            f"aucun formulaire préparé à déclarer envoyé (état « {entry.state.value} »)"
        )
    mark_submitted(job_id, seen=seen, job_store=jobs, note=note)


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
