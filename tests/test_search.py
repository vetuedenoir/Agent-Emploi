import pytest

from agent_emploi.models import Job, JobState
from agent_emploi.search import run_search
from agent_emploi.sources.base import SourceError
from agent_emploi.store.seen import SeenStore


class FakeSource:
    """Source scriptée : une liste d'offres par requête, ou une exception."""

    name = "fake"
    instances: list["FakeSource"] = []
    script: dict[str, object] = {}

    def __init__(self) -> None:
        self.closed = False
        self.queries: list[str] = []
        FakeSource.instances.append(self)

    def search(self, query):
        self.queries.append(query.text)
        result = FakeSource.script.get(query.text, [])
        if isinstance(result, Exception):
            raise result
        return list(result)

    def enrich(self, job):
        return job

    def close(self) -> None:
        self.closed = True


def make_job(n: int, company: str = "Acme") -> Job:
    return Job.build(
        source="fake",
        url=f"https://example.com/jobs/{n}",
        title=f"Poste {n}",
        company=company,
    )


@pytest.fixture
def wired(config, monkeypatch, tmp_path):
    """Configuration à deux requêtes, branchée sur FakeSource."""
    FakeSource.instances = []
    FakeSource.script = {}
    monkeypatch.setattr("agent_emploi.search.REGISTRY", {"fake": FakeSource})
    config.search.sources = ["fake"]
    config.search.queries = ["ia", "ml"]
    store = SeenStore(tmp_path / "seen.jsonl")
    return config, store


class TestRunSearch:
    def test_collects_from_every_query(self, wired):
        config, store = wired
        FakeSource.script = {"ia": [make_job(1)], "ml": [make_job(2)]}

        report = run_search(config, store)
        assert report.found == 2
        assert len(report.new) == 2
        assert FakeSource.instances[0].queries == ["ia", "ml"]

    def test_deduplicates_across_queries(self, wired):
        """Deux requêtes ramènent forcément des offres communes."""
        config, store = wired
        FakeSource.script = {"ia": [make_job(1)], "ml": [make_job(1)]}

        report = run_search(config, store)
        assert report.found == 2
        assert len(report.new) == 1
        assert report.duplicates == 1

    def test_already_known_offers_are_skipped(self, wired):
        config, store = wired
        store.record(make_job(1))
        FakeSource.script = {"ia": [make_job(1), make_job(2, "Globex")], "ml": []}

        report = run_search(config, store)
        assert [job.title for job in report.new] == ["Poste 2"]

    def test_records_new_offers(self, wired):
        config, store = wired
        FakeSource.script = {"ia": [make_job(1)], "ml": []}

        run_search(config, store)
        assert len(store) == 1

    def test_dry_run_records_nothing(self, wired):
        config, store = wired
        FakeSource.script = {"ia": [make_job(1)], "ml": []}

        report = run_search(config, store, record=False)
        assert len(report.new) == 1
        assert len(store) == 0

    def test_dry_run_is_repeatable(self, wired):
        config, store = wired
        FakeSource.script = {"ia": [make_job(1)], "ml": []}

        first = run_search(config, store, record=False)
        second = run_search(config, store, record=False)
        assert len(first.new) == len(second.new) == 1


class TestFailures:
    def test_failed_query_does_not_lose_the_others(self, wired):
        config, store = wired
        FakeSource.script = {"ia": SourceError("403 refusé"), "ml": [make_job(2)]}

        report = run_search(config, store)
        assert len(report.new) == 1
        assert any("403" in error for error in report.errors)

    def test_unknown_source_is_reported(self, wired):
        config, store = wired
        config.search.sources = ["inexistante"]

        report = run_search(config, store)
        assert report.new == []
        assert any("source inconnue" in error for error in report.errors)

    def test_source_without_credentials_is_skipped_once(self, wired, monkeypatch):
        """Une source non configurée est annoncée une fois, pas à chaque requête.

        Sans ce raccourci, chacune des requêtes de chaque passe produirait le
        même message d'identifiants absents, noyant les vraies erreurs.
        """
        config, store = wired
        monkeypatch.delenv("CLE_ABSENTE", raising=False)
        monkeypatch.setattr(FakeSource, "required_env", ("CLE_ABSENTE",), raising=False)
        FakeSource.script = {"ia": [make_job(1)], "ml": [make_job(2)]}

        report = run_search(config, store)
        assert report.new == []
        assert FakeSource.instances == []
        assert report.errors == ["fake: ignorée, variables absentes: CLE_ABSENTE"]

    def test_configured_source_runs_normally(self, wired, monkeypatch):
        config, store = wired
        monkeypatch.setenv("CLE_PRESENTE", "valeur")
        monkeypatch.setattr(FakeSource, "required_env", ("CLE_PRESENTE",), raising=False)
        FakeSource.script = {"ia": [make_job(1)], "ml": []}

        report = run_search(config, store)
        assert len(report.new) == 1
        assert report.errors == []

    def test_source_is_closed_even_on_failure(self, wired):
        config, store = wired
        FakeSource.script = {"ia": SourceError("boom"), "ml": SourceError("boom")}

        run_search(config, store)
        assert FakeSource.instances[0].closed


def test_limit_overrides_config(wired):
    config, store = wired
    config.search.per_query_limit = 40
    FakeSource.script = {"ia": [], "ml": []}

    captured: list[int] = []
    original = FakeSource.search

    def spy(self, query):
        captured.append(query.limit)
        return original(self, query)

    FakeSource.search = spy
    try:
        run_search(config, store, limit=7)
    finally:
        FakeSource.search = original

    assert captured == [7, 7]


class TestPending:
    """Une offre mémorisée mais jamais filtrée doit revenir dans la passe suivante."""

    def test_known_but_unscreened_offer_is_reported(self, wired):
        config, store = wired
        job = make_job(1)
        store.record(job)
        FakeSource.script = {"ia": [job], "ml": []}

        report = run_search(config, store)

        assert report.new == []
        assert [pending.id for pending in report.pending] == [job.id]
        assert report.to_screen == report.pending

    def test_already_screened_offer_is_not_reported(self, wired):
        config, store = wired
        job = make_job(1)
        store.record(job)
        store.transition(job.id, JobState.REJECTED, "exclu:commercial")
        FakeSource.script = {"ia": [job], "ml": []}

        report = run_search(config, store)

        assert report.pending == []

    def test_new_offers_are_not_counted_twice(self, wired):
        config, store = wired
        FakeSource.script = {"ia": [make_job(1)], "ml": []}

        report = run_search(config, store)

        assert len(report.new) == 1
        assert report.pending == []
        assert len(report.to_screen) == 1
