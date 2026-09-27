"""Offres ajoutées à la main : vues ailleurs, choisies par l'utilisateur.

Deux usages, et ils n'ont pas le même parcours :

- **préparer une candidature** : l'offre rejoint le flux au pré-filtrage, sans
  en subir les filtres ni la porte Jev, puis passe le fit-check. Elle est
  retenue **quel que soit le score** : le choix est déjà fait, le verdict ne
  sert qu'à nourrir la lettre (langue, donc CV ; atouts et manques). `draft`
  la reprend ensuite comme les autres.
- **suivre seulement** : l'offre reste en `TRACKED`, visible dans
  l'historique. On la déclare envoyée, on l'abandonne, ou on la bascule plus
  tard en préparation.

Dans les deux cas elle entre dans `seen.jsonl` : le dédoublonnage empêche une
passe `search` de la retrouver et de la traiter une seconde fois.
"""

from __future__ import annotations

from datetime import datetime

from agent_emploi.agents.fit import FitAgent
from agent_emploi.models import FitVerdict, Job, JobState, SeenEntry, job_id
from agent_emploi.store.jobs import JobStore
from agent_emploi.store.seen import SeenStore

SOURCE = "manual"

MODES = ("prepare", "track")


class DuplicateOffer(ValueError):
    """L'offre est déjà connue, sous cette URL ou sous son entreprise et son titre."""

    def __init__(self, existing: SeenEntry) -> None:
        super().__init__(f"offre déjà connue : {existing.company} — {existing.title}")
        self.existing = existing


class NotPreparable(ValueError):
    """L'offre ne peut pas (ou plus) passer en préparation."""


def build_job(
    *,
    url: str,
    title: str,
    company: str,
    description: str = "",
    location: str | None = None,
    contract: str | None = None,
    remote: str | None = None,
    apply_url: str | None = None,
    posted_at: datetime | None = None,
) -> Job:
    """Construit l'offre. Lève `ValueError` (pydantic) sur une URL invalide.

    L'identifiant garde la requête de l'URL : collée à la main, elle vient de
    n'importe quel site, et certains y mettent la référence de l'offre.
    """
    url = url.strip()
    return Job(
        id=job_id(SOURCE, url, keep_query=True),
        source=SOURCE,
        url=url,
        title=title.strip(),
        company=company.strip(),
        description=description.strip(),
        location=location or None,
        contract=contract or None,
        remote=remote or None,
        # Sans lien de candidature distinct, c'est l'annonce qui en tient lieu.
        apply_url=apply_url or url,
        posted_at=posted_at,
    )


def find_existing(job: Job, seen: SeenStore) -> SeenEntry | None:
    """L'entrée qui rend cet ajout redondant, s'il y en a une.

    Une offre écartée par les filtres automatiques ne bloque pas l'ajout :
    c'est précisément le cas où l'utilisateur la veut malgré le verdict. Seule
    une offre encore vivante, ou la même URL, fait doublon.
    """
    same = seen.get(job.id)
    if same is not None:
        return same
    duplicate = seen.find_duplicate(job)
    if duplicate is not None and duplicate.state is not JobState.REJECTED:
        return duplicate
    return None


def add(job: Job, *, mode: str, seen: SeenStore, jobs: JobStore) -> SeenEntry:
    """Enregistre l'offre dans les deux journaux. Lève `DuplicateOffer`."""
    if mode not in MODES:
        raise ValueError(f"mode inconnu : {mode!r}")
    existing = find_existing(job, seen)
    if existing is not None:
        raise DuplicateOffer(existing)

    jobs.save(job)
    if mode == "track":
        return seen.record(job, JobState.TRACKED, reason="manuel:suivi")
    seen.record(job, JobState.DISCOVERED, reason="manuel:ajoutée")
    return seen.transition(job.id, JobState.PRESCREENED, "manuel:préparation")


def preparable(entry: SeenEntry | None, job: Job | None, *, has_fit: bool) -> bool:
    """Vrai si l'offre peut passer (ou repasser) le fit-check de préparation.

    Une offre en pré-filtrage sans verdict est une préparation dont le
    fit-check a échoué (quota, réseau) : elle se relance d'ici.
    """
    if entry is None or job is None or job.source != SOURCE or not job.description:
        return False
    return entry.state is JobState.TRACKED or (
        entry.state is JobState.PRESCREENED and not has_fit
    )


def prepare(
    job_id: str, *, seen: SeenStore, jobs: JobStore, fit_agent: FitAgent
) -> FitVerdict:
    """Fit-check, puis `FIT_OK` quel que soit le verdict.

    Propage `LlmError` et `BudgetExceeded` : l'offre reste alors en
    pré-filtrage, prête à être relancée.
    """
    entry, record = seen.get(job_id), jobs.get(job_id)
    job = record.job if record else None
    if not preparable(entry, job, has_fit=bool(record and record.fit)):
        raise NotPreparable(
            "seule une offre ajoutée à la main, avec sa description, se prépare"
        )
    if entry.state is JobState.TRACKED:
        seen.transition(job_id, JobState.PRESCREENED, "manuel:préparation")

    verdict = fit_agent.evaluate(job)
    jobs.save(job, fit=verdict)
    seen.transition(
        job_id, JobState.FIT_OK, f"manuel:fit:{verdict.verdict}({verdict.score})"
    )
    return verdict
