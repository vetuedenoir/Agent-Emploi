"""Calibration de la porte : échantillon équilibré, cache, tableau des seuils."""

from __future__ import annotations

from agent_emploi.calibrate import collect, fit_accepted, grid, load_cache, sample
from agent_emploi.config import FitConfig
from agent_emploi.llm.jev import JevError, JevUnavailable
from agent_emploi.models import FitVerdict, GateVerdict, Job, JobRecord

FIT = FitConfig(min_score=65, accept_verdicts=["apply", "maybe"])


def record(n: int, score: int, verdict: str = "maybe") -> JobRecord:
    job = Job.build(
        source="fake",
        url=f"https://example.com/{n}",
        title=f"Poste {n}",
        company="Acme",
        description="Stage ML.",
    )
    fit = FitVerdict(score=score, verdict=verdict, reason="r", language="fr")
    return JobRecord(job=job, fit=fit)


def raw(adequation: float, experience: float = 0.1) -> GateVerdict:
    return GateVerdict(
        adequation=adequation, experience_blocking=experience, contract_ok=0.9, passed=True
    )


class FakeGate:
    def __init__(self, results) -> None:
        self.results = list(results)
        self.calls = 0

    def evaluate(self, job):
        self.calls += 1
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def test_label_follows_the_fit_config():
    assert fit_accepted(record(1, 70), FIT)
    assert not fit_accepted(record(2, 60), FIT)
    assert not fit_accepted(record(3, 90, "skip"), FIT)


def test_sample_favours_the_rare_accepted_offers():
    records = [record(n, 80) for n in range(3)] + [record(n, 20) for n in range(3, 40)]
    picked = sample(records, FIT, 10)
    assert len(picked) == 10
    assert sum(fit_accepted(r, FIT) for r in picked) == 3


def test_collect_uses_the_cache_and_stops_on_unavailable(tmp_path):
    cache = tmp_path / "cal.jsonl"
    records = [record(n, 80) for n in range(3)]
    first = collect(FakeGate([raw(2.0), JevError("422"), JevUnavailable("402")]), records, cache)
    assert first.queried == 1
    assert len(first.errors) == 1
    assert first.stopped
    assert set(load_cache(cache)) == {records[0].job.id}

    gate = FakeGate([raw(1.0), raw(0.5)])
    second = collect(gate, records, cache)
    assert gate.calls == 2  # la première offre vient du cache
    assert len(second.verdicts) == 3


def test_grid_counts_kept_and_cut():
    verdicts = {"a": raw(2.5), "b": raw(1.2), "c": raw(2.5, experience=0.8)}
    labels = {"a": True, "b": False, "c": False}
    rows = {(r.min_score, r.max_blocking): r for r in grid(verdicts, labels, 0.3)}

    strict = rows[(1.5, 0.7)]
    assert (strict.kept, strict.cut) == (1, 2)
    loose = rows[(1.0, 0.9)]
    assert (loose.kept, loose.cut) == (1, 0)
