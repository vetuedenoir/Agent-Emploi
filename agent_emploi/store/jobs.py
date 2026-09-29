"""Offres retenues et leur verdict — `data/jobs.jsonl`.

Complément de `seen.jsonl`, qui ne mémorise que des états. Ici on garde l'offre
entière (description comprise) et son verdict d'adéquation, pour que l'étape de
rédaction puisse travailler sans réinterroger la source — et pour qu'une passe
interrompue reprenne là où elle s'est arrêtée.

Même forme que `seen.jsonl` : journal en ajout seul, dernière ligne d'un
identifiant faisant foi.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from agent_emploi.models import (
    FitVerdict,
    GateVerdict,
    Job,
    JobRecord,
    Letter,
    ReviewVerdict,
    UserDecision,
)

logger = logging.getLogger(__name__)


class JobStore:
    """Index en mémoire des offres retenues, adossé à un journal JSONL."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._records: dict[str, JobRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = JobRecord.model_validate_json(line)
                except (ValueError, json.JSONDecodeError):
                    logger.warning("%s:%d ligne illisible, ignorée", self.path, line_no)
                    continue
                self._records[record.job.id] = record

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, job_id: str) -> bool:
        return job_id in self._records

    def get(self, job_id: str) -> JobRecord | None:
        return self._records.get(job_id)

    def records(self) -> list[JobRecord]:
        return list(self._records.values())

    def accepted(self) -> list[JobRecord]:
        """Offres dont le verdict autorise la rédaction, les plus fortes d'abord."""
        scored = [record for record in self._records.values() if record.fit is not None]
        return sorted(scored, key=lambda record: record.fit.score, reverse=True)

    def save(
        self,
        job: Job,
        *,
        gate: GateVerdict | None = None,
        fit: FitVerdict | None = None,
        lexical_score: float | None = None,
        letter: Letter | None = None,
        review: ReviewVerdict | None = None,
        outbox: str | None = None,
        decision: UserDecision | None = None,
        submitted_at: datetime | None = None,
        interview_at: datetime | None = None,
        archive: str | None = None,
    ) -> JobRecord:
        """Écrit ou met à jour une offre. Les champs non fournis sont conservés.

        Le pré-filtrage enregistre l'offre et son score lexical, le fit-check
        repasse pour y ajouter le verdict, puis la rédaction y dépose la lettre
        et sa revue : ces passages successifs sur la même offre ne doivent pas
        s'effacer les uns les autres.
        """
        existing = self._records.get(job.id)

        def keep(value, field: str):
            """Valeur fournie, sinon celle déjà enregistrée."""
            if value is not None:
                return value
            return getattr(existing, field) if existing else None

        record = JobRecord(
            job=job,
            gate=keep(gate, "gate"),
            fit=keep(fit, "fit"),
            lexical_score=keep(lexical_score, "lexical_score"),
            letter=keep(letter, "letter"),
            review=keep(review, "review"),
            outbox=keep(outbox, "outbox"),
            decision=keep(decision, "decision"),
            submitted_at=keep(submitted_at, "submitted_at"),
            interview_at=keep(interview_at, "interview_at"),
            archive=keep(archive, "archive"),
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json() + "\n")
        self._records[job.id] = record
        return record
