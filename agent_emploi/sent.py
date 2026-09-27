"""Déclaration d'un envoi fait par l'utilisateur.

Le programme ne remplit ni n'envoie aucun formulaire : chaque site a le sien.
Il prépare le dossier, l'utilisateur candidate lui-même sur le site, puis
déclare l'envoi — c'est cette déclaration, et elle seule, qui fait passer une
offre à `submitted`.
"""

from __future__ import annotations

from agent_emploi.models import JobRecord, JobState, utcnow
from agent_emploi.store.jobs import JobStore
from agent_emploi.store.seen import SeenStore

#: États d'où l'on peut déclarer un envoi : un dossier approuvé, une offre
#: suivie à la main, ou un dossier laissé par l'ancien remplissage automatique.
SENDABLE = frozenset(
    {JobState.APPROVED, JobState.TRACKED, JobState.PREFILLED, JobState.HANDOFF}
)


def mark_submitted(
    job_id: str,
    *,
    seen: SeenStore,
    job_store: JobStore,
    note: str | None = None,
) -> JobRecord | None:
    """Enregistre qu'une candidature a été envoyée par l'utilisateur.

    Retourne `None` si l'offre est inconnue ou n'est pas dans un état d'où
    l'on envoie.
    """
    entry, record = seen.get(job_id), job_store.get(job_id)
    if entry is None or record is None or entry.state not in SENDABLE:
        return None

    saved = job_store.save(record.job, submitted_at=utcnow())
    seen.transition(
        job_id,
        JobState.SUBMITTED,
        f"utilisateur:envoyé{f' — {note}' if note else ''}"[:200],
    )
    return saved
