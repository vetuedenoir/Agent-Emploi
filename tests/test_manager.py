"""La boucle bout en bout, déroulée sans réseau ni LLM.

Ce qu'on vérifie ici n'est pas le détail de chaque passe — chacune a ses tests —
mais leur enchaînement : une offre découverte doit ressortir en dossier prêt à
relire, et **s'arrêter là**. La boucle ne valide rien à votre place et n'ouvre
aucun navigateur.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from agent_emploi.llm.budget import BudgetExceeded
from agent_emploi.manager import run_pipeline
from agent_emploi.models import (
    FitVerdict,
    Job,
    JobState,
    Letter,
    ReviewVerdict,
    UserDecision,
    utcnow,
)
from agent_emploi.store.jobs import JobStore
from agent_emploi.store.seen import SeenStore

CV = """
Stage machine learning. Python, Tensorflow, Pandas, numpy, Docker.
LLM, RAG, agents, deep learning.
"""

DESCRIPTION = (
    "Stage machine learning : entraînement de modèles deep learning en Python "
    "avec Tensorflow, industrialisation Docker, agents LLM et RAG."
)

LETTRE = " ".join(["mot"] * 170)


def make_job(n: int = 1, **fields) -> Job:
    base = {
        "title": "Ingénieur machine learning",
        "company": f"Acme {n}",
        "contract": "stage",
        "description": "",
    }
    return Job.build(
        source="fake", url=f"https://example.com/jobs/{n}", **{**base, **fields}
    )


def verdict(**fields) -> FitVerdict:
    base = {
        "score": 80,
        "verdict": "apply",
        "matched": ["Python"],
        "gaps": [],
        "reason": "ok",
        "language": "fr",
    }
    return FitVerdict.model_validate({**base, **fields})


class FakeSource:
    """Source scriptée : elle rend des offres, puis leur description."""

    name = "fake"
    jobs: list[Job] = []

    def __init__(self) -> None:
        self.queries: list[str] = []
        self.closed = False

    def search(self, query):
        self.queries.append(query.text)
        return list(FakeSource.jobs)

    def enrich(self, job: Job) -> Job:
        return job.model_copy(update={"description": DESCRIPTION})

    def close(self) -> None:
        self.closed = True


class FakeFitAgent:
    def __init__(self, results: list | None = None) -> None:
        self.results = list(results or [])
        self.seen: list[str] = []

    def evaluate(self, job: Job) -> FitVerdict:
        self.seen.append(job.id)
        result = self.results.pop(0) if self.results else verdict()
        if isinstance(result, Exception):
            raise result
        return result

    def accepts(self, v: FitVerdict) -> bool:
        return v.verdict == "apply" and v.score >= 65

    def rejection_reason(self, v: FitVerdict) -> str:
        return f"fit:{v.verdict}({v.score})"


class FakeLetterAgent:
    def __init__(self, texts: list | None = None) -> None:
        self.texts = list(texts or [])
        self.calls = 0

    def write(self, job: Job, fit: FitVerdict, *, feedback: str | None = None) -> Letter:
        self.calls += 1
        result = self.texts.pop(0) if self.texts else LETTRE
        if isinstance(result, Exception):
            raise result
        return Letter(text=result, language=fit.language)

    def defects(self, letter: Letter) -> list[str]:
        return []


class FakeReviewAgent:
    def __init__(self) -> None:
        self.calls = 0

    def review(self, job: Job, letter: Letter) -> ReviewVerdict:
        self.calls += 1
        return ReviewVerdict(approved=True)

    @staticmethod
    def feedback(v: ReviewVerdict) -> str:
        return "reprends"


@pytest.fixture
def wired(config, tmp_path, profile, monkeypatch):
    """Une chaîne complète branchée sur des doublures, sans réseau ni LLM."""
    monkeypatch.setattr("agent_emploi.search.REGISTRY", {"fake": FakeSource})
    config.search.sources = ["fake"]
    config.search.queries = ["machine learning"]
    config.search.contracts = ["stage", "CDI"]
    config.filters.required_any = ["machine learning", "ia"]
    config.filters.excluded_any = ["commercial"]
    config.filters.min_lexical_score = 0.05
    config.profile.cv_fr = profile.cv_fr
    FakeSource.jobs = []
    return config, SeenStore(config.paths.seen_file), JobStore(config.paths.jobs_file)


def pipeline(wired, profile, **kwargs):
    config, seen, job_store = wired
    kwargs.setdefault("fit_agent", FakeFitAgent())
    kwargs.setdefault("letter_agent", FakeLetterAgent())
    kwargs.setdefault("review_agent", FakeReviewAgent())
    return run_pipeline(
        config,
        seen=seen,
        job_store=job_store,
        profile=profile,
        cv_text=CV,
        sources={"fake": FakeSource()},
        **kwargs,
    )


def test_une_offre_traverse_toute_la_chaine(wired, profile):
    """Découverte, filtrée, notée, rédigée, livrée — et pas un pas de plus."""
    config, seen, job_store = wired
    FakeSource.jobs = [make_job(1)]

    report = pipeline(wired, profile)

    assert len(report.search.new) == 1
    assert report.screen.prescreened == 1
    assert len(report.screen.accepted) == 1
    assert len(report.draft.prepared) == 1
    assert report.errors == []

    job_id = report.search.new[0].id
    assert seen.get(job_id).state is JobState.AWAITING_USER

    directory = Path(job_store.get(job_id).outbox)
    assert (directory / "lettre.md").exists()
    assert (directory / "preview.html").exists()
    # Le dossier attend une décision : rien ne l'a approuvé au passage.
    assert job_store.get(job_id).decision is None
    assert len(report.awaiting) == 1


def test_la_boucle_s_arrete_avant_toute_candidature(wired, profile):
    """`approved` ne s'obtient que par une décision humaine, jamais par la boucle."""
    config, seen, job_store = wired
    FakeSource.jobs = [make_job(1)]

    report = pipeline(wired, profile)

    states = {entry.state for entry in seen.entries()}
    assert JobState.APPROVED not in states
    assert JobState.SUBMITTED not in states
    assert report.archive.archived == []


def test_une_offre_hors_sujet_s_arrete_au_filtrage(wired, profile):
    config, seen, job_store = wired
    FakeSource.jobs = [make_job(1, title="Commercial grands comptes")]

    report = pipeline(wired, profile)

    assert report.draft.prepared == []
    assert report.awaiting == []
    assert seen.get(FakeSource.jobs[0].id).state is JobState.REJECTED


def test_le_plafond_de_budget_arrete_la_boucle(wired, profile):
    """Interruption propre : l'offre reste où elle est, la relance la reprend."""
    config, seen, job_store = wired
    FakeSource.jobs = [make_job(1)]

    report = pipeline(
        wired,
        profile,
        fit_agent=FakeFitAgent([BudgetExceeded("plafond de dépense atteint")]),
    )

    assert "plafond" in report.stopped
    assert report.draft.prepared == []
    assert seen.get(FakeSource.jobs[0].id).state is JobState.PRESCREENED


def test_la_redaction_reprend_une_offre_deja_notee(wired, profile):
    """Une passe précédente interrompue avant la lettre n'est pas perdue."""
    config, seen, job_store = wired
    old = make_job(9, description=DESCRIPTION)
    seen.record(old)
    seen.transition(old.id, JobState.PRESCREENED)
    seen.transition(old.id, JobState.FIT_OK)
    job_store.save(old, fit=verdict())
    FakeSource.jobs = []

    report = pipeline(wired, profile)

    assert [item.record.job.id for item in report.draft.prepared] == [old.id]
    assert seen.get(old.id).state is JobState.AWAITING_USER


def test_la_boucle_archive_les_candidatures_envoyees(wired, profile, tmp_path):
    """Le rangement de fin de passe : `outbox/` ne garde que ce qui est en cours."""
    config, seen, job_store = wired
    sent = make_job(7, description=DESCRIPTION)
    directory = config.paths.outbox / "2026-08-09_acme-7_ingenieur"
    directory.mkdir(parents=True)
    (directory / "lettre.md").write_text("Ma lettre.\n", encoding="utf-8")

    seen.record(sent)
    for state in (
        JobState.PRESCREENED,
        JobState.FIT_OK,
        JobState.DRAFTED,
        JobState.REVIEWED,
        JobState.AWAITING_USER,
        JobState.APPROVED,
        JobState.SUBMITTED,
    ):
        seen.transition(sent.id, state)
    job_store.save(
        sent,
        fit=verdict(),
        letter=Letter(text=LETTRE, language="fr"),
        outbox=str(directory),
        decision=UserDecision(decision="approved"),
        submitted_at=utcnow(),
    )
    FakeSource.jobs = []

    report = pipeline(wired, profile)

    assert len(report.archive.archived) == 1
    assert not directory.exists()
    assert (report.archive.archived[0].directory / "README.md").exists()


def test_passe_a_blanc_traverse_la_chaine_sans_rien_ecrire(wired, profile):
    """La chaîne se déroule jusqu'au bout, mais les mémoires restent vierges.

    Le filtrage n'ayant rien persisté, la rédaction reprend ses offres depuis le
    rapport — sans quoi une passe à blanc s'arrêterait au filtrage et ne
    montrerait jamais ce que la chaîne produit.
    """
    config, seen, job_store = wired
    FakeSource.jobs = [make_job(1)]

    report = pipeline(wired, profile, record=False)

    assert len(report.draft.prepared) == 1
    assert len(seen) == 0
    assert len(job_store) == 0
    assert report.awaiting == []
    # Le dossier n'existe que sur le papier : la passe est rejouable telle quelle.
    assert not report.draft.prepared[0].directory.exists()


def test_les_lettres_sont_bornees_par_leur_propre_limite(wired, profile):
    """La recherche peut être large, la rédaction reste au rythme configuré."""
    config, seen, job_store = wired
    config.letter.max_per_day = 5
    FakeSource.jobs = [make_job(n) for n in range(1, 4)]
    letter_agent = FakeLetterAgent()

    report = pipeline(wired, profile, draft_limit=1, letter_agent=letter_agent)

    assert len(report.screen.accepted) == 3
    assert letter_agent.calls == 1
    assert len(report.draft.prepared) == 1
    assert len(report.awaiting) == 1
