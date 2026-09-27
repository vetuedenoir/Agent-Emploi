"""Passes en arrière-plan et page de consommation."""

import logging
import sys
import threading

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from agent_emploi.models import Job, JobState, LlmUsage, utcnow
from agent_emploi.store.jobs import JobStore
from agent_emploi.store.seen import SeenStore
from agent_emploi.web.app import create_app
from agent_emploi.web.passes import PassBusy, PassRunner, PassSpec

logger = logging.getLogger("agent_emploi.test_pass")


def printing(config, dry_run, limit, letters):
    print(f"passe dry_run={dry_run} limit={limit} letters={letters}")
    print("ko", file=sys.stderr)
    logger.info("une offre traitée")
    logging.getLogger("httpx").info("HTTP Request: GET …")
    return 0


def crashing(config, dry_run, limit, letters):
    raise RuntimeError("boum")


class Gate:
    """Une passe qui attend qu'on la libère : de quoi tester « en cours »."""

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def __call__(self, config, dry_run, limit, letters):
        self.started.set()
        self.release.wait(5)
        return 1


@pytest.fixture
def gate():
    return Gate()


@pytest.fixture
def runner(config, gate):
    return PassRunner(
        config,
        commands={
            "print": PassSpec("Impression", "", ("limit", "letters"), printing),
            "crash": PassSpec("Plantage", "", (), crashing),
            "wait": PassSpec("Attente", "", (), gate),
        },
    )


@pytest.fixture
def client(config, runner):
    app = create_app(config, runner=runner)
    runner.write_lock = app.state.stores.write_lock
    return TestClient(app, base_url="http://127.0.0.1")


class TestRunner:
    def test_captures_output_and_logs_of_the_pass_only(self, runner, capsys):
        run = runner.start("print", dry_run=True, limit=5, letters=2)
        runner.join(5)
        assert run.status == "ok"
        assert run.exit_code == 0
        assert "passe dry_run=True limit=5 letters=2" in run.lines
        assert "ko" in run.lines
        assert any(line.endswith("INFO une offre traitée") for line in run.lines)
        assert not any("HTTP Request" in line for line in run.lines)
        # Rien n'a fui vers la sortie du processus.
        assert "passe dry_run" not in capsys.readouterr().out

    def test_options_the_pass_does_not_take_are_dropped(self, runner):
        run = runner.start("crash", limit=5)
        runner.join(5)
        assert run.limit is None

    def test_crash_is_reported_not_raised(self, runner):
        run = runner.start("crash")
        runner.join(5)
        assert run.status == "crashed"
        assert any("RuntimeError: boum" in line for line in run.lines)
        assert not runner.busy

    def test_one_pass_at_a_time(self, runner, gate):
        runner.start("wait")
        gate.started.wait(5)
        with pytest.raises(PassBusy):
            runner.start("print")
        gate.release.set()
        runner.join(5)
        assert runner.history()[0].status == "failed"

    def test_history_survives_restart(self, runner, config):
        runner.start("print")
        runner.join(5)
        again = PassRunner(config, commands=runner.commands)
        (run,) = again.history()
        assert run.label == "Impression"
        assert run.status == "ok"


    def test_task_keeps_its_link_across_restarts(self, runner, config):
        runner.start_task("prepare", "Fit-check manuel", lambda: 0, link=("/offres/x", "Fiche"))
        runner.join(5)
        (run,) = PassRunner(config, commands={}).history()
        assert run.link == ("/offres/x", "Fiche")
        assert run.status == "ok"


class TestPages:
    def test_start_redirects_to_log(self, client, runner):
        response = client.post("/passes/print", data={"limit": "3"}, follow_redirects=False)
        assert response.status_code == 303
        runner.join(5)
        page = client.get(response.headers["location"]).text
        assert "limit=3" in page
        assert "terminée" in page
        assert 'hx-trigger="every 2s"' not in page

    def test_running_pass_is_polled_and_blocks_writes(self, client, runner, gate, config):
        job = Job.build(source="wttj", url="https://example.com/1", title="T", company="C")
        seen = SeenStore(config.paths.seen_file)
        seen.record(job, JobState.TRACKED)
        JobStore(config.paths.jobs_file).save(job)

        response = client.post("/passes/wait", follow_redirects=False)
        gate.started.wait(5)
        page = client.get(response.headers["location"]).text
        assert 'hx-trigger="every 2s"' in page

        assert client.post("/passes/print").status_code == 409
        refused = client.post(f"/offres/{job.id}/envoyee", data={})
        assert refused.status_code == 409
        assert "une passe est en cours" in refused.text
        assert "Passe en cours : Attente" in client.get("/offres").text

        gate.release.set()
        runner.join(5)
        done = client.post(f"/offres/{job.id}/envoyee", data={}, follow_redirects=False)
        assert done.status_code == 303

    def test_unknown_pass_is_404(self, client):
        assert client.post("/passes/inconnue").status_code == 404
        assert client.get("/passes/inconnue").status_code == 404

    def test_dashboard_lists_recent_passes(self, client, runner):
        client.post("/passes/print")
        runner.join(5)
        assert "Impression" in client.get("/").text


class TestUsage:
    def test_aggregates_by_task_and_counts_failures(self, client, config):
        path = config.paths.usage_file
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            LlmUsage(task="fit_check", provider="groq", model="gpt-oss", tokens_in=1000),
            LlmUsage(task="fit_check", provider="groq", model="gpt-oss", ok=False, error="429 rate limit"),
            LlmUsage(task="letter", provider="anthropic", model="opus", cost_est=0.05),
            LlmUsage(task="gate", provider="jev", model="jev-latest", at=utcnow().replace(year=2020)),
        ]
        path.write_text("".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8")

        text = client.get("/conso").text
        assert "fit_check" in text and "jev · jev-latest" in text
        assert "0.0500 $" in text
        assert "429 rate limit" in text

        recent = client.get("/conso", params={"jours": 7}).text
        assert "jev-latest" not in recent

    def test_empty_log(self, client):
        assert "Aucun appel" in client.get("/conso").text
