"""Mémoire des offres déjà rencontrées.

Le fichier `seen.jsonl` est un journal en ajout seul : chaque changement d'état
écrit une nouvelle ligne, et la dernière ligne d'un identifiant fait foi. Cette
forme survit à une interruption en cours d'écriture (au pire, une ligne
tronquée en fin de fichier, ignorée au chargement) et garde l'historique des
transitions pour analyse.
"""

from __future__ import annotations

import json
import logging
from datetime import timedelta
from pathlib import Path

from agent_emploi.models import (
    InvalidTransition,
    Job,
    JobState,
    SeenEntry,
    check_transition,
    normalize,
    utcnow,
)

logger = logging.getLogger(__name__)


class SeenStore:
    """Index en mémoire des offres vues, adossé à un journal JSONL."""

    def __init__(self, path: Path, *, dedup_window_days: int = 60) -> None:
        self.path = Path(path)
        self.dedup_window = timedelta(days=dedup_window_days)
        self._entries: dict[str, SeenEntry] = {}
        self._load()

    # ------------------------------------------------------------------ lecture

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = SeenEntry.model_validate_json(line)
                except (ValueError, json.JSONDecodeError):
                    # Ligne tronquée par une interruption, ou format obsolète :
                    # on la signale et on continue plutôt que de tout perdre.
                    logger.warning("%s:%d ligne illisible, ignorée", self.path, line_no)
                    continue
                self._entries[entry.id] = entry

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, job_id: str) -> bool:
        return job_id in self._entries

    def get(self, job_id: str) -> SeenEntry | None:
        return self._entries.get(job_id)

    def entries(self) -> list[SeenEntry]:
        return list(self._entries.values())

    # ------------------------------------------------------------ dédoublonnage

    def is_known(self, job: Job) -> bool:
        """Vrai si l'offre a déjà été traitée, par identifiant ou par similarité.

        Le second test rattrape les offres republiées : même entreprise et même
        intitulé sous la fenêtre configurée, mais URL — donc identifiant —
        différente.
        """
        if job.id in self._entries:
            return True
        return self.find_duplicate(job) is not None

    def find_duplicate(self, job: Job) -> SeenEntry | None:
        """Cherche une offre équivalente (entreprise + titre) encore dans la fenêtre."""
        key = (normalize(job.company), normalize(job.title))
        cutoff = utcnow() - self.dedup_window
        for entry in self._entries.values():
            if entry.id == job.id:
                continue
            if entry.dedup_key() == key and entry.first_seen >= cutoff:
                return entry
        return None

    def filter_new(self, jobs: list[Job]) -> list[Job]:
        """Retire les offres déjà connues, en dédoublonnant aussi le lot courant.

        Deux sources peuvent renvoyer la même offre dans la même passe ; le
        dédoublonnage interne évite de la traiter deux fois avant qu'elle ne soit
        enregistrée.
        """
        fresh: list[Job] = []
        batch_keys: set[tuple[str, str]] = set()
        for job in jobs:
            if self.is_known(job):
                continue
            key = (normalize(job.company), normalize(job.title))
            if key in batch_keys:
                continue
            batch_keys.add(key)
            fresh.append(job)
        return fresh

    # ------------------------------------------------------------------ écriture

    def record(
        self,
        job: Job,
        state: JobState = JobState.DISCOVERED,
        reason: str | None = None,
    ) -> SeenEntry:
        """Enregistre une offre nouvellement découverte.

        Si elle est déjà connue, son entrée existante est retournée inchangée —
        `record` est donc sûr à rappeler.
        """
        existing = self._entries.get(job.id)
        if existing is not None:
            return existing

        now = utcnow()
        entry = SeenEntry(
            id=job.id,
            source=job.source,
            url=str(job.url),
            company=job.company,
            title=job.title,
            state=state,
            first_seen=now,
            last_state_change=now,
            reason=reason,
        )
        self._append(entry)
        return entry

    def transition(
        self, job_id: str, target: JobState, reason: str | None = None
    ) -> SeenEntry:
        """Fait passer une offre à un nouvel état, en validant la transition."""
        entry = self._entries.get(job_id)
        if entry is None:
            raise KeyError(f"offre inconnue: {job_id}")

        check_transition(entry.state, target)

        updated = entry.model_copy(
            update={
                "state": target,
                "last_state_change": utcnow(),
                "reason": reason,
            }
        )
        self._append(updated)
        return updated

    def advance(
        self, job_id: str, target: JobState, reason: str | None = None
    ) -> SeenEntry | None:
        """Transition tolérante, pour les passes rejouables.

        Une offre absente de la mémoire (passe à blanc) ou déjà plus avancée que
        l'état visé n'est pas une erreur : les passes sont rejouables, et la
        mémoire fait foi. Retourne `None` quand rien n'a été écrit.
        """
        try:
            return self.transition(job_id, target, reason)
        except (InvalidTransition, KeyError) as exc:
            logger.debug("transition ignorée pour %s: %s", job_id, exc)
            return None

    def _append(self, entry: SeenEntry) -> None:
        """Écrit une ligne dans le journal et met l'index à jour."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(entry.model_dump_json() + "\n")
        self._entries[entry.id] = entry

    # ------------------------------------------------------------------ analyse

    def history(self, job_id: str) -> list[SeenEntry]:
        """Toutes les transitions d'une offre, de la découverte à l'état actuel.

        L'index ne garde que la dernière ligne de chaque offre : le journal est
        donc relu en entier. C'est le prix d'une frise, pas celui d'une passe,
        qui n'en a jamais besoin.
        """
        if not self.path.exists():
            return []
        steps: list[SeenEntry] = []
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                # Filtre textuel avant l'analyse : quelques centaines de lignes
                # sur des milliers concernent l'offre, les autres sont sautées.
                if job_id not in line:
                    continue
                try:
                    entry = SeenEntry.model_validate_json(line)
                except (ValueError, json.JSONDecodeError):
                    continue
                if entry.id == job_id:
                    steps.append(entry)
        return steps

    def count_by_state(self) -> dict[JobState, int]:
        """Répartition des offres par état — utile pour le suivi et les rapports."""
        counts: dict[JobState, int] = {}
        for entry in self._entries.values():
            counts[entry.state] = counts.get(entry.state, 0) + 1
        return counts
