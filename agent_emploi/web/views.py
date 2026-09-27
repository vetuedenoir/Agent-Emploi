"""Ce que les pages affichent, calculé depuis les deux journaux.

`seen.jsonl` connaît toutes les offres, rejets compris, mais seulement leur
état. `jobs.jsonl` ne connaît que celles qui ont franchi le pré-filtrage, mais
en entier. Une ligne d'historique est la jointure des deux : la mémoire fait foi
sur l'état, `jobs.jsonl` apporte le reste quand il existe.

Rien ici n'écrit : ces fonctions lisent et trient, les routes les affichent.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from agent_emploi.config import Config
from agent_emploi.llm.budget import BudgetTracker
from agent_emploi.models import JobRecord, JobState, SeenEntry, normalize, utcnow
from agent_emploi.outbox import read_letter
from agent_emploi.review_cli import concerns, pending
from agent_emploi.store.jobs import JobStore
from agent_emploi.store.seen import SeenStore

#: Libellés affichés, dans l'ordre du parcours nominal.
STATE_LABELS: dict[JobState, str] = {
    JobState.DISCOVERED: "découverte",
    JobState.PRESCREENED: "pré-filtrée",
    JobState.FIT_OK: "retenue",
    JobState.DRAFTED: "rédigée",
    JobState.REVIEWED: "relue",
    JobState.AWAITING_USER: "en attente",
    JobState.APPROVED: "approuvée",
    JobState.PREFILLED: "pré-remplie",
    JobState.SUBMITTED: "envoyée",
    JobState.HANDOFF: "à finir à la main",
    JobState.REJECTED: "rejetée",
}

#: Périodes proposées au filtre, en jours.
PERIODS = (7, 30, 90)

SORTS = ("date", "score")


def fold(text: str) -> str:
    """`normalize`, accents retirés : « societe » doit trouver « Société »."""
    decomposed = unicodedata.normalize("NFKD", normalize(text))
    return "".join(char for char in decomposed if not unicodedata.combining(char))


@dataclass
class OfferRow:
    """Une offre de l'historique : son état, et ce qu'on sait d'elle."""

    entry: SeenEntry
    record: JobRecord | None

    @property
    def id(self) -> str:
        return self.entry.id

    @property
    def score(self) -> int | None:
        if self.record is None or self.record.fit is None:
            return None
        return self.record.fit.score

    @property
    def location(self) -> str | None:
        return self.record.job.location if self.record else None

    @property
    def contract(self) -> str | None:
        return self.record.job.contract if self.record else None


@dataclass
class OfferFilters:
    """Critères de la page d'historique, tels que lus dans l'URL."""

    q: str = ""
    state: str = ""
    source: str = ""
    days: int | None = None
    sort: str = "date"

    def matches(self, row: OfferRow, now: datetime) -> bool:
        entry = row.entry
        if self.state and entry.state.value != self.state:
            return False
        if self.source and entry.source != self.source:
            return False
        if self.days is not None and entry.last_state_change < now - timedelta(
            days=self.days
        ):
            return False
        if self.q:
            needle = fold(self.q)
            if needle not in fold(entry.company) and needle not in fold(entry.title):
                return False
        return True


def offer_rows(
    seen: SeenStore, jobs: JobStore, filters: OfferFilters | None = None
) -> list[OfferRow]:
    """L'historique filtré et trié : les plus récentes, ou les mieux notées, d'abord."""
    filters = filters or OfferFilters()
    now = utcnow()
    rows = [
        row
        for entry in seen.entries()
        if filters.matches(row := OfferRow(entry, jobs.get(entry.id)), now)
    ]
    rows.sort(key=lambda row: row.entry.last_state_change, reverse=True)
    if filters.sort == "score":
        # Tri stable : à score égal, la plus récente reste devant ; les offres
        # jamais notées ferment la marche.
        rows.sort(key=lambda row: (row.score is not None, row.score or 0), reverse=True)
    return rows


@dataclass
class Dashboard:
    total: int
    counts: list[tuple[JobState, int]]
    pending: int
    budget: BudgetTracker
    recent: list[OfferRow] = field(default_factory=list)


def dashboard(
    seen: SeenStore, jobs: JobStore, config: Config, *, recent: int = 10
) -> Dashboard:
    """Vue d'ensemble : où en sont les offres, ce qui attend, ce qu'on a dépensé."""
    by_state = seen.count_by_state()
    counts = [(state, by_state[state]) for state in JobState if by_state.get(state)]
    return Dashboard(
        total=len(seen),
        counts=counts,
        pending=len(pending(jobs, seen, config)),
        budget=BudgetTracker(config.paths.usage_file, config.budget),
        recent=offer_rows(seen, jobs)[:recent],
    )


@dataclass
class OfferDetail:
    """Tout ce qu'on sait d'une offre, pour sa page."""

    id: str
    entry: SeenEntry | None
    record: JobRecord | None
    history: list[SeenEntry]
    letter: str | None = None
    #: D'où vient la lettre affichée : `lettre.md` du dossier, qui fait foi, ou
    #: `jobs.jsonl` quand le dossier n'existe pas (ou plus).
    letter_origin: str | None = None
    concerns: list[str] = field(default_factory=list)

    @property
    def title(self) -> str:
        if self.record:
            return self.record.job.title
        return self.entry.title if self.entry else self.id

    @property
    def company(self) -> str:
        if self.record:
            return self.record.job.company
        return self.entry.company if self.entry else ""

    @property
    def url(self) -> str | None:
        if self.record:
            return str(self.record.job.url)
        return self.entry.url if self.entry else None


def offer_detail(
    job_id: str, seen: SeenStore, jobs: JobStore, config: Config
) -> OfferDetail | None:
    """La fiche d'une offre, ou `None` si aucun journal ne la connaît."""
    entry = seen.get(job_id)
    record = jobs.get(job_id)
    if entry is None and record is None:
        return None

    detail = OfferDetail(
        id=job_id, entry=entry, record=record, history=seen.history(job_id)
    )
    if record is None:
        return detail

    if record.outbox:
        text = read_letter(Path(record.outbox))
        if text:
            detail.letter, detail.letter_origin = text, "outbox"
    if detail.letter is None and record.letter is not None:
        detail.letter, detail.letter_origin = record.letter.text, "jobs"

    if entry is not None and entry.state is JobState.AWAITING_USER and record.outbox:
        detail.concerns = concerns(record, Path(record.outbox), config)
    return detail
