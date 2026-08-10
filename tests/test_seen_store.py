from datetime import timedelta

import pytest

from agent_emploi.models import InvalidTransition, Job, JobState, utcnow
from agent_emploi.store.seen import SeenStore


def make_job(url: str = "https://example.com/jobs/1", **overrides) -> Job:
    fields = {"title": "ML Engineer", "company": "Acme"}
    fields.update(overrides)
    return Job.build(source="wttj", url=url, **fields)


@pytest.fixture
def store(tmp_path):
    return SeenStore(tmp_path / "seen.jsonl", dedup_window_days=60)


class TestRecording:
    def test_records_and_finds_job(self, store):
        job = make_job()
        store.record(job)
        assert job.id in store
        assert store.get(job.id).state is JobState.DISCOVERED

    def test_record_is_idempotent(self, store):
        job = make_job()
        first = store.record(job)
        second = store.record(job)
        assert first.first_seen == second.first_seen
        assert len(store) == 1

    def test_survives_reload(self, tmp_path):
        path = tmp_path / "seen.jsonl"
        job = make_job()
        SeenStore(path).record(job)

        reloaded = SeenStore(path)
        assert job.id in reloaded

    def test_truncated_line_is_skipped_not_fatal(self, tmp_path):
        path = tmp_path / "seen.jsonl"
        job = make_job()
        SeenStore(path).record(job)
        # Simule une écriture interrompue en fin de fichier.
        with path.open("a", encoding="utf-8") as handle:
            handle.write('{"id": "tronq')

        reloaded = SeenStore(path)
        assert len(reloaded) == 1


class TestTransitions:
    def test_last_state_wins_after_reload(self, tmp_path):
        path = tmp_path / "seen.jsonl"
        job = make_job()
        store = SeenStore(path)
        store.record(job)
        store.transition(job.id, JobState.PRESCREENED)
        store.transition(job.id, JobState.REJECTED, reason="hors périmètre")

        reloaded = SeenStore(path)
        entry = reloaded.get(job.id)
        assert entry.state is JobState.REJECTED
        assert entry.reason == "hors périmètre"

    def test_invalid_transition_is_refused(self, store):
        job = make_job()
        store.record(job)
        with pytest.raises(InvalidTransition):
            store.transition(job.id, JobState.SUBMITTED)

    def test_unknown_job_raises(self, store):
        with pytest.raises(KeyError):
            store.transition("inconnu", JobState.PRESCREENED)


class TestDeduplication:
    def test_known_by_id(self, store):
        job = make_job()
        store.record(job)
        assert store.is_known(job)

    def test_republished_offer_detected_by_company_and_title(self, store):
        store.record(make_job(url="https://example.com/jobs/1"))
        republished = make_job(url="https://example.com/jobs/2")

        assert republished.id != make_job().id
        assert store.is_known(republished)

    def test_same_title_at_other_company_is_not_duplicate(self, store):
        store.record(make_job())
        other = make_job(url="https://example.com/jobs/9", company="Globex")
        assert not store.is_known(other)

    def test_duplicate_outside_window_is_allowed_again(self, tmp_path):
        path = tmp_path / "seen.jsonl"
        store = SeenStore(path, dedup_window_days=60)
        entry = store.record(make_job(url="https://example.com/jobs/1"))
        # Rembobine la date de première vue au-delà de la fenêtre.
        old = entry.model_copy(update={"first_seen": utcnow() - timedelta(days=90)})
        store._entries[entry.id] = old

        republished = make_job(url="https://example.com/jobs/2")
        assert not store.is_known(republished)

    def test_filter_new_removes_known_and_intra_batch_duplicates(self, store):
        store.record(make_job(url="https://example.com/jobs/1"))
        batch = [
            make_job(url="https://example.com/jobs/1"),  # déjà connue
            make_job(url="https://example.com/jobs/2"),  # republiée
            make_job(url="https://example.com/jobs/3", company="Globex"),
            make_job(url="https://example.com/jobs/4", company="Globex"),  # doublon du lot
        ]
        fresh = store.filter_new(batch)
        assert [job.company for job in fresh] == ["Globex"]


def test_count_by_state(store):
    store.record(make_job(url="https://example.com/jobs/1"))
    second = store.record(make_job(url="https://example.com/jobs/2", company="Globex"))
    store.transition(second.id, JobState.PRESCREENED)

    counts = store.count_by_state()
    assert counts == {JobState.DISCOVERED: 1, JobState.PRESCREENED: 1}
