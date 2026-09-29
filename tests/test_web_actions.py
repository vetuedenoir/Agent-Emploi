"""Interface web en écriture : lettre, décision, envoi, archivage."""

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from agent_emploi.models import FitVerdict, Job, JobState, Letter
from agent_emploi.outbox import read_letter
from agent_emploi.store.jobs import JobStore
from agent_emploi.store.seen import SeenStore
from agent_emploi.web.app import create_app

TO_AWAITING = (
    JobState.PRESCREENED,
    JobState.FIT_OK,
    JobState.DRAFTED,
    JobState.REVIEWED,
    JobState.AWAITING_USER,
)


@pytest.fixture
def web_config(config, tmp_path):
    banned = tmp_path / "banned.txt"
    banned.write_text("passionné par\n", encoding="utf-8")
    config.profile.banned_phrases = banned
    return config


@pytest.fixture
def client(web_config):
    return TestClient(create_app(web_config), base_url="http://127.0.0.1")


def seen_store(config) -> SeenStore:
    return SeenStore(config.paths.seen_file)


def job_store(config) -> JobStore:
    return JobStore(config.paths.jobs_file)


@pytest.fixture
def bundle(web_config, tmp_path):
    """Une offre en attente de décision, dossier complet dans outbox/."""
    job = Job.build(
        source="wttj", url="https://example.com/jobs/acme", company="Acme", title="Ingénieur IA"
    )
    directory = tmp_path / "outbox" / "acme"
    directory.mkdir(parents=True)
    (directory / "lettre.md").write_text("<!-- en-tête -->\nLettre du modèle.", encoding="utf-8")
    (directory / "cv.pdf").write_bytes(b"%PDF")

    seen = seen_store(web_config)
    seen.record(job)
    for state in TO_AWAITING:
        seen.transition(job.id, state)
    job_store(web_config).save(
        job,
        fit=FitVerdict(
            score=80, verdict="apply", reason="ok", language="fr"
        ),
        letter=Letter(text="Lettre du modèle.", language="fr"),
        outbox=str(directory),
    )
    return job, directory


def state(config, job) -> JobState:
    return seen_store(config).get(job.id).state


class TestLetter:
    def test_edit_writes_outbox_and_jobs(self, client, bundle, web_config):
        job, directory = bundle
        response = client.post(
            f"/offres/{job.id}/lettre",
            data={"text": "Madame, je suis passionné par vos agents.\r\n"},
            follow_redirects=False,
        )
        assert response.status_code == 303

        assert read_letter(directory) == "Madame, je suis passionné par vos agents."
        assert (directory / "preview.html").exists()
        letter = job_store(web_config).get(job.id).letter
        assert letter.edited
        assert letter.banned_hits == ["passionné par"]

        page = client.get(response.headers["location"]).text
        assert "Lettre enregistrée." in page
        assert "Formules interdites : passionné par" in page

    def test_empty_letter_is_refused_and_draft_kept(self, client, bundle):
        job, directory = bundle
        response = client.post(f"/offres/{job.id}/lettre", data={"text": "   "})
        assert response.status_code == 409
        assert read_letter(directory) == "Lettre du modèle."

    def test_read_only_once_sent(self, client, bundle, web_config):
        job, _ = bundle
        seen = seen_store(web_config)
        seen.transition(job.id, JobState.APPROVED)
        seen.transition(job.id, JobState.SUBMITTED)

        assert "<textarea" not in client.get(f"/offres/{job.id}").text
        response = client.post(f"/offres/{job.id}/lettre", data={"text": "Autre."})
        assert response.status_code == 409

    def test_edited_letter_is_the_one_approved(self, client, bundle, web_config):
        job, _ = bundle
        client.post(f"/offres/{job.id}/lettre", data={"text": "Version corrigée."})
        client.post(f"/offres/{job.id}/approuver", data={"note": ""})

        record = job_store(web_config).get(job.id)
        assert record.letter.text == "Version corrigée."
        assert record.decision.letter_edited


class TestDecision:
    def test_approve(self, client, bundle, web_config):
        job, directory = bundle
        response = client.post(
            f"/offres/{job.id}/approuver", data={"note": "go"}, follow_redirects=False
        )
        assert response.status_code == 303
        assert state(web_config, job) is JobState.APPROVED
        assert json.loads((directory / "decision.json").read_text())["note"] == "go"

        page = client.get(response.headers["location"]).text
        assert "J'ai envoyé la candidature" in page

    def test_reject_keeps_reason(self, client, bundle, web_config):
        job, _ = bundle
        client.post(f"/offres/{job.id}/rejeter", data={"note": "trop loin"})
        entry = seen_store(web_config).get(job.id)
        assert entry.state is JobState.REJECTED
        assert entry.reason == "utilisateur:trop loin"

    def test_approving_twice_is_a_conflict(self, client, bundle):
        job, _ = bundle
        client.post(f"/offres/{job.id}/approuver", data={})
        response = client.post(f"/offres/{job.id}/approuver", data={})
        assert response.status_code == 409
        assert "seul un dossier en attente" in response.text

    def test_unknown_offer_is_404(self, client, bundle):
        assert client.post("/offres/inconnue/approuver", data={}).status_code == 404


class TestSentAndArchive:
    @pytest.fixture
    def approved(self, bundle, web_config):
        job, directory = bundle
        seen_store(web_config).transition(job.id, JobState.APPROVED)
        return job, directory

    def test_sent_is_refused_before_approval(self, client, bundle):
        job, _ = bundle
        assert client.post(f"/offres/{job.id}/envoyee", data={}).status_code == 409

    def test_mark_sent_then_archive(self, client, approved, web_config):
        job, directory = approved
        client.post(f"/offres/{job.id}/envoyee", data={"note": "par mail"})
        assert state(web_config, job) is JobState.SUBMITTED
        assert job_store(web_config).get(job.id).submitted_at is not None

        response = client.post(f"/offres/{job.id}/archiver", follow_redirects=False)
        assert response.status_code == 303
        record = job_store(web_config).get(job.id)
        assert record.archive is not None
        assert not directory.exists()
        assert (web_config.paths.applications / record.archive.split("/")[-1]).exists()

        # Plus rien à archiver une seconde fois.
        assert client.post(f"/offres/{job.id}/archiver").status_code == 409


class TestOwnLetter:
    @pytest.fixture
    def retained(self, web_config):
        """Une offre retenue par le fit-check, sans lettre ni dossier."""
        job = Job.build(
            source="wttj", url="https://example.com/jobs/beta", company="Beta", title="Dev IA"
        )
        seen = seen_store(web_config)
        seen.record(job)
        seen.transition(job.id, JobState.PRESCREENED)
        seen.transition(job.id, JobState.FIT_OK)
        job_store(web_config).save(
            job,
            fit=FitVerdict(
                score=70, verdict="apply", reason="ok", language="en", cv="data"
            ),
        )
        return job

    def test_form_shown_only_while_retained(self, client, retained, bundle):
        assert "/ma-lettre" in client.get(f"/offres/{retained.id}").text
        job, _ = bundle
        assert "/ma-lettre" not in client.get(f"/offres/{job.id}").text

    def test_own_letter_builds_the_bundle(self, client, retained, web_config):
        response = client.post(
            f"/offres/{retained.id}/ma-lettre",
            data={"text": "Dear team, I am passionné par agents.\r\n"},
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert state(web_config, retained) is JobState.AWAITING_USER

        record = job_store(web_config).get(retained.id)
        assert record.letter.authored
        assert record.letter.language == "en"
        assert record.letter.cv == "data"
        assert record.letter.banned_hits == ["passionné par"]
        assert record.review is None
        directory = Path(record.outbox)
        assert read_letter(directory) == "Dear team, I am passionné par agents."
        assert (directory / "preview.html").exists()

        page = client.get(response.headers["location"]).text
        assert "écrite à la main" in page
        # La décision reste une étape à part.
        client.post(f"/offres/{retained.id}/approuver", data={})
        assert state(web_config, retained) is JobState.APPROVED

    def test_empty_letter_is_refused(self, client, retained, web_config):
        response = client.post(f"/offres/{retained.id}/ma-lettre", data={"text": "  "})
        assert response.status_code == 409
        assert state(web_config, retained) is JobState.FIT_OK

    def test_refused_once_drafted(self, client, bundle):
        job, _ = bundle
        response = client.post(f"/offres/{job.id}/ma-lettre", data={"text": "Autre."})
        assert response.status_code == 409


class TestInterview:
    @pytest.fixture
    def sent(self, client, bundle, web_config):
        job, directory = bundle
        seen_store(web_config).transition(job.id, JobState.APPROVED)
        client.post(f"/offres/{job.id}/envoyee", data={})
        return job

    def test_refused_before_sending(self, client, bundle, web_config):
        job, _ = bundle
        assert client.post(f"/offres/{job.id}/entretien", data={}).status_code == 409
        assert state(web_config, job) is JobState.AWAITING_USER

    def test_mark_interview_with_date(self, client, sent, web_config):
        response = client.post(
            f"/offres/{sent.id}/entretien",
            data={"day": "2026-10-12", "note": "visio"},
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert state(web_config, sent) is JobState.INTERVIEW
        assert seen_store(web_config).get(sent.id).reason == "utilisateur:entretien — visio"
        record = job_store(web_config).get(sent.id)
        assert record.interview_at.astimezone().date().isoformat() == "2026-10-12"

        page = client.get(response.headers["location"]).text
        assert "Entretien enregistré." in page
        assert "12/10/2026" in page
        # Déclaré une fois : plus de bouton, et un second envoi est refusé.
        assert "/entretien" not in page
        assert client.post(f"/offres/{sent.id}/entretien", data={}).status_code == 409

    def test_bad_date_is_refused(self, client, sent, web_config):
        response = client.post(f"/offres/{sent.id}/entretien", data={"day": "demain"})
        assert response.status_code == 409
        assert state(web_config, sent) is JobState.SUBMITTED

    def test_interview_can_be_archived(self, client, sent, web_config):
        client.post(f"/offres/{sent.id}/entretien", data={})
        client.post(f"/offres/{sent.id}/archiver")
        archive = Path(job_store(web_config).get(sent.id).archive)
        followup = (archive / "README.md").read_text(encoding="utf-8")
        assert "entretien obtenu" in followup


class TestGuards:
    def test_cross_site_post_is_refused(self, client, bundle, web_config):
        job, _ = bundle
        response = client.post(
            f"/offres/{job.id}/approuver",
            data={},
            headers={"Origin": "https://evil.example", "Sec-Fetch-Site": "cross-site"},
        )
        assert response.status_code == 403
        assert state(web_config, job) is JobState.AWAITING_USER

    def test_foreign_origin_without_fetch_metadata_is_refused(self, client, bundle):
        job, _ = bundle
        response = client.post(
            f"/offres/{job.id}/approuver", data={}, headers={"Origin": "http://evil.example"}
        )
        assert response.status_code == 403

    def test_same_origin_post_is_accepted(self, client, bundle, web_config):
        job, _ = bundle
        response = client.post(
            f"/offres/{job.id}/approuver",
            data={},
            headers={"Origin": "http://127.0.0.1", "Sec-Fetch-Site": "same-origin"},
            follow_redirects=False,
        )
        assert response.status_code == 303

    def test_foreign_host_is_refused(self, web_config):
        # DNS rebinding : un domaine étranger qui résout vers 127.0.0.1.
        client = TestClient(create_app(web_config), base_url="http://evil.example")
        assert client.get("/").status_code == 400
