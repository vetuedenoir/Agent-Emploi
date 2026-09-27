import pytest

from agent_emploi import manual
from agent_emploi.llm.router import LlmError
from agent_emploi.models import FitVerdict, Job, JobState
from agent_emploi.store.jobs import JobStore
from agent_emploi.store.seen import SeenStore


class FakeFit:
    def __init__(self, score: int = 20, verdict: str = "skip", error: Exception | None = None):
        self.score, self.verdict, self.error = score, verdict, error
        self.calls = 0

    def evaluate(self, job):
        self.calls += 1
        if self.error:
            raise self.error
        return FitVerdict(score=self.score, verdict=self.verdict, reason="r", language="en")


@pytest.fixture
def seen(tmp_path):
    return SeenStore(tmp_path / "seen.jsonl")


@pytest.fixture
def jobs(tmp_path):
    return JobStore(tmp_path / "jobs.jsonl")


def job(url="https://fr.indeed.com/viewjob?jk=abc&utm_source=x", **fields) -> Job:
    values = {"title": "ML Engineer", "company": "Acme", "description": "Python, RAG."}
    values.update(fields)
    return manual.build_job(url=url, **values)


class TestBuild:
    def test_query_identifies_the_offer(self):
        a = job("https://fr.indeed.com/viewjob?jk=abc")
        b = job("https://fr.indeed.com/viewjob?jk=def")
        tracked = job("https://fr.indeed.com/viewjob?utm_source=mail&jk=abc")
        assert a.id != b.id
        assert a.id == tracked.id

    def test_apply_url_defaults_to_offer(self):
        built = job()
        assert built.source == "manual"
        assert built.apply_url == "https://fr.indeed.com/viewjob?jk=abc&utm_source=x"


class TestAdd:
    def test_track_mode(self, seen, jobs):
        built = job()
        manual.add(built, mode="track", seen=seen, jobs=jobs)
        assert seen.get(built.id).state is JobState.TRACKED
        assert jobs.get(built.id).job.description == "Python, RAG."

    def test_prepare_mode_skips_filters(self, seen, jobs):
        built = job()
        manual.add(built, mode="prepare", seen=seen, jobs=jobs)
        assert seen.get(built.id).state is JobState.PRESCREENED

    def test_same_url_is_a_duplicate(self, seen, jobs):
        manual.add(job(), mode="track", seen=seen, jobs=jobs)
        with pytest.raises(manual.DuplicateOffer):
            manual.add(job(), mode="prepare", seen=seen, jobs=jobs)

    def test_live_offer_from_a_source_is_a_duplicate(self, seen, jobs):
        seen.record(Job.build(source="wttj", url="https://wttj.example/1", title="ML Engineer", company="ACME"))
        with pytest.raises(manual.DuplicateOffer):
            manual.add(job(), mode="track", seen=seen, jobs=jobs)

    def test_offer_rejected_by_filters_can_be_added(self, seen, jobs):
        other = Job.build(source="wttj", url="https://wttj.example/1", title="ML Engineer", company="Acme")
        seen.record(other)
        seen.transition(other.id, JobState.REJECTED, "lexical")
        manual.add(job(), mode="track", seen=seen, jobs=jobs)


class TestPrepare:
    def test_low_score_is_still_retained(self, seen, jobs):
        built = job()
        manual.add(built, mode="prepare", seen=seen, jobs=jobs)
        manual.prepare(built.id, seen=seen, jobs=jobs, fit_agent=FakeFit(score=20))
        entry = seen.get(built.id)
        assert entry.state is JobState.FIT_OK
        assert entry.reason == "manuel:fit:skip(20)"
        assert jobs.get(built.id).fit.language == "en"

    def test_tracked_offer_can_be_prepared_later(self, seen, jobs):
        built = job()
        manual.add(built, mode="track", seen=seen, jobs=jobs)
        manual.prepare(built.id, seen=seen, jobs=jobs, fit_agent=FakeFit())
        states = [step.state for step in seen.history(built.id)]
        assert states == [JobState.TRACKED, JobState.PRESCREENED, JobState.FIT_OK]

    def test_failure_leaves_offer_ready_to_retry(self, seen, jobs):
        built = job()
        manual.add(built, mode="track", seen=seen, jobs=jobs)
        with pytest.raises(LlmError):
            manual.prepare(
                built.id, seen=seen, jobs=jobs, fit_agent=FakeFit(error=LlmError("quota"))
            )
        assert seen.get(built.id).state is JobState.PRESCREENED
        manual.prepare(built.id, seen=seen, jobs=jobs, fit_agent=FakeFit())
        assert seen.get(built.id).state is JobState.FIT_OK

    def test_needs_description(self, seen, jobs):
        built = job(description="")
        manual.add(built, mode="track", seen=seen, jobs=jobs)
        with pytest.raises(manual.NotPreparable):
            manual.prepare(built.id, seen=seen, jobs=jobs, fit_agent=FakeFit())

    def test_source_offers_are_not_preparable(self, seen, jobs):
        other = Job.build(source="wttj", url="https://wttj.example/1", title="T", company="C", description="d")
        seen.record(other)
        seen.transition(other.id, JobState.PRESCREENED)
        jobs.save(other)
        with pytest.raises(manual.NotPreparable):
            manual.prepare(other.id, seen=seen, jobs=jobs, fit_agent=FakeFit())
