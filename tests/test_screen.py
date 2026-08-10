import pytest

from agent_emploi.llm.budget import BudgetExceeded
from agent_emploi.llm.router import LlmError
from agent_emploi.models import FitVerdict, Job, JobState
from agent_emploi.screen import load_cv_text, run_screen
from agent_emploi.store.jobs import JobStore
from agent_emploi.store.seen import SeenStore

CV = """
Stage machine learning. Python, Tensorflow, Pandas, numpy, Docker.
LLM, RAG, agents, deep learning, computer vision.
"""

DESCRIPTION = (
    "Stage machine learning : entraînement de modèles deep learning en Python "
    "avec Tensorflow, industrialisation Docker, agents LLM et RAG."
)


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


class FakeSource:
    """Source qui remplit la description à l'enrichissement, ou échoue."""

    name = "fake"

    def __init__(self, description: str = DESCRIPTION, error: Exception | None = None):
        self.error = error
        self.description = description
        self.enriched: list[str] = []

    def enrich(self, job: Job) -> Job:
        self.enriched.append(job.id)
        if self.error is not None:
            raise self.error
        return job.model_copy(update={"description": self.description})


class FakeAgent:
    """Agent de fit scripté : un verdict (ou une exception) par appel."""

    def __init__(self, results: list) -> None:
        self.results = list(results)
        self.seen: list[str] = []

    def evaluate(self, job: Job) -> FitVerdict:
        self.seen.append(job.id)
        result = self.results.pop(0) if self.results else verdict(score=90)
        if isinstance(result, Exception):
            raise result
        return result

    def accepts(self, v: FitVerdict) -> bool:
        return v.verdict == "apply" and v.score >= 65

    def rejection_reason(self, v: FitVerdict) -> str:
        return f"fit:{v.verdict}({v.score})"


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


@pytest.fixture
def wired(config, tmp_path):
    """Config filtrante, mémoires vierges, offres déjà découvertes."""
    config.filters.required_any = ["machine learning", "ia"]
    config.filters.excluded_any = ["commercial"]
    config.filters.min_lexical_score = 0.05
    config.search.contracts = ["stage", "CDI"]
    seen = SeenStore(tmp_path / "seen.jsonl")
    job_store = JobStore(tmp_path / "jobs.jsonl")
    return config, seen, job_store


def screen(wired, jobs, agent=None, source=None, **kwargs):
    config, seen, job_store = wired
    for job in jobs:
        seen.record(job)
    return run_screen(
        config,
        jobs,
        seen=seen,
        job_store=job_store,
        fit_agent=agent or FakeAgent([]),
        cv_text=CV,
        sources={"fake": source or FakeSource()},
        **kwargs,
    )


class TestPipeline:
    def test_accepts_a_matching_offer(self, wired):
        report = screen(wired, [make_job()], FakeAgent([verdict(score=88)]))

        assert report.prescreened == 1
        assert report.evaluated == 1
        assert [v.score for _, v in report.accepted] == [88]
        _, seen, _ = wired
        assert seen.entries()[0].state is JobState.FIT_OK

    def test_metadata_rejection_costs_no_request(self, wired):
        source = FakeSource()
        agent = FakeAgent([])
        report = screen(wired, [make_job(title="Commercial IA")], agent, source)

        assert source.enriched == []
        assert agent.seen == []
        assert report.reasons["exclu"] == 1
        _, seen, _ = wired
        assert seen.entries()[0].state is JobState.REJECTED

    def test_content_rejection_costs_no_llm_call(self, wired):
        # Titre dans le sujet, contenu réel hors sujet : c'est le score lexical,
        # calculé sur la description complète, qui doit trancher.
        wired[0].filters.min_lexical_score = 0.2
        source = FakeSource(
            description=(
                "Vente de mobilier de bureau aux collectivités territoriales. "
                "Prospection téléphonique, rendez-vous clients, négociation "
                "tarifaire, suivi des commandes, reporting hebdomadaire auprès "
                "de la direction régionale des ventes. Permis B indispensable, "
                "véhicule de fonction, primes sur objectifs commerciaux."
            )
        )
        agent = FakeAgent([])
        report = screen(wired, [make_job()], agent, source)

        assert source.enriched  # le détail a bien été téléchargé
        assert agent.seen == []
        assert report.prescreened == 0
        assert sum(report.reasons.values()) == 1

    def test_low_fit_is_rejected_with_its_reason(self, wired):
        report = screen(wired, [make_job()], FakeAgent([verdict(verdict="skip", score=20)]))

        assert report.accepted == []
        assert len(report.refused) == 1
        _, seen, _ = wired
        entry = seen.entries()[0]
        assert entry.state is JobState.REJECTED
        assert entry.reason == "fit:skip(20)"

    def test_accepted_offers_are_sorted_by_score(self, wired):
        jobs = [make_job(1), make_job(2), make_job(3)]
        agent = FakeAgent([verdict(score=70), verdict(score=95), verdict(score=80)])
        report = screen(wired, jobs, agent)

        assert [v.score for _, v in report.accepted] == [95, 80, 70]


class TestPersistence:
    def test_stores_enriched_job_and_verdict(self, wired):
        _, _, job_store = wired
        job = make_job()
        screen(wired, [job], FakeAgent([verdict(score=88)]))

        record = job_store.get(job.id)
        assert record is not None
        assert record.job.is_enriched
        assert record.fit.score == 88
        assert record.lexical_score and record.lexical_score > 0.05

    def test_dry_run_persists_nothing(self, wired):
        _, seen, job_store = wired
        job = make_job()
        report = screen(wired, [job], FakeAgent([verdict()]), record=False)

        assert report.accepted
        assert len(job_store) == 0
        assert seen.get(job.id).state is JobState.DISCOVERED

    def test_second_pass_keeps_the_last_verdict(self, wired):
        _, _, job_store = wired
        job = make_job()
        screen(wired, [job], FakeAgent([verdict(score=70)]))
        record = job_store.save(job, fit=verdict(score=95))

        assert record.fit.score == 95
        assert record.lexical_score is not None  # conservé du premier passage


class TestFailures:
    def test_enrichment_failure_skips_the_offer_only(self, wired):
        jobs = [make_job(1), make_job(2)]
        report = run_screen(
            wired[0],
            jobs,
            seen=wired[1],
            job_store=wired[2],
            fit_agent=FakeAgent([]),
            cv_text=CV,
            sources={"fake": FakeSource(error=RuntimeError("502"))},
        )
        assert report.errors and report.accepted == []

    def test_llm_failure_leaves_offer_prescreened(self, wired):
        job = make_job()
        report = screen(wired, [job], FakeAgent([LlmError("tous les modèles ont échoué")]))

        assert report.errors
        assert report.evaluated == 0
        _, seen, _ = wired
        assert seen.get(job.id).state is JobState.PRESCREENED

    def test_repeated_llm_failures_stop_the_pass(self, wired):
        """Une route cassée ne doit pas coûter un appel par offre."""
        jobs = [make_job(n) for n in range(1, 6)]
        agent = FakeAgent([LlmError("clé absente")] * 5)
        report = screen(wired, jobs, agent)

        assert len(agent.seen) == 3
        assert "route fit_check inutilisable" in report.stopped

    def test_a_success_resets_the_failure_counter(self, wired):
        jobs = [make_job(n) for n in range(1, 6)]
        agent = FakeAgent(
            [LlmError("x"), LlmError("x"), verdict(), LlmError("x"), verdict(score=70)]
        )
        report = screen(wired, jobs, agent)

        assert report.stopped is None
        assert len(report.accepted) == 2

    def test_budget_stops_the_pass(self, wired):
        jobs = [make_job(1), make_job(2), make_job(3)]
        agent = FakeAgent([verdict(), BudgetExceeded("plafond atteint")])
        report = screen(wired, jobs, agent)

        assert report.stopped == "plafond atteint"
        assert len(agent.seen) == 2  # la troisième offre n'est pas évaluée
        assert len(report.accepted) == 1

    def test_unknown_source_is_reported(self, wired):
        config, seen, job_store = wired
        job = make_job()
        seen.record(job)
        report = run_screen(
            config,
            [job],
            seen=seen,
            job_store=job_store,
            fit_agent=FakeAgent([]),
            cv_text=CV,
            sources={},
        )
        assert report.errors == ["source inconnue: fake"]


class TestLoadCvText:
    def test_reads_the_cv(self, config, tmp_path):
        path = tmp_path / "cv.md"
        path.write_text("Mon CV", encoding="utf-8")
        config.profile.cv_markdown = path
        assert load_cv_text(config) == "Mon CV"

    def test_missing_cv_fails_loudly(self, config, tmp_path):
        config.profile.cv_markdown = tmp_path / "absent.md"
        with pytest.raises(FileNotFoundError, match="filtrage lexical"):
            load_cv_text(config)
