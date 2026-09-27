"""Ajout manuel depuis l'interface : pré-remplissage, préparation, suivi."""

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from agent_emploi.llm.router import LlmError
from agent_emploi.models import FitVerdict, JobState
from agent_emploi.store.jobs import JobStore
from agent_emploi.store.seen import SeenStore
from agent_emploi.web import jsonld
from agent_emploi.web.app import create_app

URL = "https://boards.example.com/acme/42"


class FakeFit:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.calls = 0

    def evaluate(self, job):
        self.calls += 1
        if self.error:
            raise self.error
        return FitVerdict(score=30, verdict="skip", reason="écart", language="fr")


@pytest.fixture
def fit():
    return FakeFit()


@pytest.fixture
def client(config, fit):
    return TestClient(
        create_app(config, fit_agent=lambda: fit), base_url="http://127.0.0.1"
    )


def form(action: str, **fields) -> dict:
    data = {
        "action": action,
        "url": URL,
        "title": "Stage IA",
        "company": "Acme",
        "description": "Agents, RAG, Python.",
    }
    data.update(fields)
    return data


def only_id(config) -> str:
    (entry,) = SeenStore(config.paths.seen_file).entries()
    return entry.id


def test_form_renders(client):
    response = client.get("/offres/nouvelle", params={"url": URL})
    assert response.status_code == 200
    assert f'value="{URL}"' in response.text


def test_prefill_keeps_what_the_user_typed(client, monkeypatch):
    monkeypatch.setattr(
        jsonld,
        "prefill",
        lambda url: jsonld.Prefill(
            title="Titre de la page", company="Acme", description="Texte", structured=True
        ),
    )
    response = client.post(
        "/offres/nouvelle",
        data={"action": "prefill", "url": URL, "title": "Mon titre", "company": "", "description": ""},
    )
    assert 'value="Mon titre"' in response.text
    assert 'value="Acme"' in response.text
    assert "Champs pré-remplis" in response.text


def test_prefill_error_is_shown(client, monkeypatch):
    def fail(url):
        raise jsonld.PrefillError("la page a répondu 403")

    monkeypatch.setattr(jsonld, "prefill", fail)
    response = client.post("/offres/nouvelle", data={"action": "prefill", "url": URL})
    assert "403" in response.text


def test_track_creates_offer_without_llm(client, config, fit):
    response = client.post("/offres/nouvelle", data=form("track"), follow_redirects=False)
    assert response.status_code == 303
    job_id = only_id(config)
    assert SeenStore(config.paths.seen_file).get(job_id).state is JobState.TRACKED
    assert fit.calls == 0
    page = client.get(response.headers["location"]).text
    assert "Offre ajoutée au suivi." in page
    assert "Préparer une candidature" in page


def test_prepare_retains_whatever_the_score(client, config, fit):
    response = client.post("/offres/nouvelle", data=form("prepare"), follow_redirects=False)
    assert response.status_code == 303
    job_id = only_id(config)
    assert SeenStore(config.paths.seen_file).get(job_id).state is JobState.FIT_OK
    assert JobStore(config.paths.jobs_file).get(job_id).fit.score == 30


def test_prepare_requires_description(client, config):
    response = client.post("/offres/nouvelle", data=form("prepare", description=""))
    assert response.status_code == 400
    assert "la description" in response.text
    assert not SeenStore(config.paths.seen_file).entries()


def test_duplicate_links_to_existing(client, config):
    client.post("/offres/nouvelle", data=form("track"))
    response = client.post("/offres/nouvelle", data=form("track"))
    assert response.status_code == 409
    assert f"/offres/{only_id(config)}" in response.text


def test_fit_failure_keeps_offer_and_offers_retry(config):
    failing = FakeFit(error=LlmError("quota épuisé"))
    client = TestClient(
        create_app(config, fit_agent=lambda: failing), base_url="http://127.0.0.1"
    )
    response = client.post("/offres/nouvelle", data=form("prepare"))
    assert "quota épuisé" in response.text
    assert "Relancer l’évaluation" in response.text
    assert SeenStore(config.paths.seen_file).get(only_id(config)).state is JobState.PRESCREENED


class TestTrackedActions:
    @pytest.fixture
    def tracked(self, client, config):
        client.post("/offres/nouvelle", data=form("track"))
        return only_id(config)

    def test_prepare_later(self, client, config, tracked):
        client.post(f"/offres/{tracked}/preparer")
        assert SeenStore(config.paths.seen_file).get(tracked).state is JobState.FIT_OK

    def test_mark_sent(self, client, config, tracked):
        client.post(f"/offres/{tracked}/envoyee", data={"note": "par mail"})
        entry = SeenStore(config.paths.seen_file).get(tracked)
        assert entry.state is JobState.SUBMITTED
        assert entry.reason == "utilisateur:envoyé — par mail"

    def test_abandon(self, client, config, tracked):
        client.post(f"/offres/{tracked}/rejeter", data={"note": ""})
        entry = SeenStore(config.paths.seen_file).get(tracked)
        assert entry.state is JobState.REJECTED
        assert JobStore(config.paths.jobs_file).get(tracked).decision.decision == "rejected"
