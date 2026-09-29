"""Interface web en lecture : tableau de bord, historique, recherche, fiche."""

from datetime import timedelta

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from agent_emploi.models import FitVerdict, Job, JobState, Letter
from agent_emploi.store.jobs import JobStore
from agent_emploi.store.seen import SeenStore
from agent_emploi.web.app import create_app


def make_job(slug: str, *, company: str, title: str = "Ingénieur IA", **fields) -> Job:
    return Job.build(
        source="wttj",
        url=f"https://example.com/jobs/{slug}",
        company=company,
        title=title,
        **fields,
    )


@pytest.fixture
def stores(config):
    seen = SeenStore(config.paths.seen_file)
    jobs = JobStore(config.paths.jobs_file)
    return seen, jobs


@pytest.fixture
def client(config):
    return TestClient(create_app(config), base_url="http://127.0.0.1")


def fit(score: int) -> FitVerdict:
    return FitVerdict(
        score=score,
        verdict="apply",
        matched=["Python"],
        gaps=["LangGraph"],
        reason="Bonne adéquation.",
        language="fr",
    )


@pytest.fixture
def populated(stores, tmp_path):
    """Trois offres : une rejetée tôt, une notée, une avec dossier en attente."""
    seen, jobs = stores
    rejected = make_job("a", company="Société Générale", title="Data Analyst")
    scored = make_job("b", company="Mistral AI", location="Paris", contract="Stage")
    pending = make_job("c", company="Acme", description="Construire des agents.")

    seen.record(rejected)
    seen.transition(rejected.id, JobState.REJECTED, reason="titre_hors_sujet")

    seen.record(scored)
    seen.transition(scored.id, JobState.PRESCREENED)
    seen.transition(scored.id, JobState.FIT_OK, reason="fit:apply(80)")
    jobs.save(scored, fit=fit(80), lexical_score=0.2)

    directory = tmp_path / "outbox" / "acme"
    directory.mkdir(parents=True)
    (directory / "lettre.md").write_text(
        "<!-- en-tête -->\nMadame, Monsieur, version corrigée.", encoding="utf-8"
    )
    seen.record(pending)
    for state in (
        JobState.PRESCREENED,
        JobState.FIT_OK,
        JobState.DRAFTED,
        JobState.REVIEWED,
        JobState.AWAITING_USER,
    ):
        seen.transition(pending.id, state)
    jobs.save(
        pending,
        fit=fit(60),
        letter=Letter(text="Version du modèle.", language="fr"),
        outbox=str(directory),
    )
    return {"rejected": rejected, "scored": scored, "pending": pending}


class TestDashboard:
    def test_empty_install_renders(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "Aucune offre" in response.text

    def test_counts_and_pending(self, client, populated):
        text = client.get("/").text
        assert "offres connues" in text
        assert "/offres?state=awaiting_user" in text
        assert "Mistral AI" in text


class TestHistory:
    def test_lists_rejected_offers_with_reason(self, client, populated):
        text = client.get("/offres").text
        assert "Data Analyst" in text
        assert "titre_hors_sujet" in text
        assert "3 offres sur 3" in text

    def test_search_by_company_ignores_case_and_accents(self, client, populated):
        text = client.get("/offres", params={"q": "societe"}).text
        assert "Société Générale" in text
        assert "Mistral AI" not in text

    def test_search_matches_title(self, client, populated):
        text = client.get("/offres", params={"q": "analyst"}).text
        assert "1 offre sur 3" in text

    def test_filter_by_state(self, client, populated):
        text = client.get("/offres", params={"state": "awaiting_user"}).text
        assert "Acme" in text
        assert "Mistral AI" not in text

    def test_sort_by_score_puts_best_first(self, client, populated):
        text = client.get("/offres", params={"sort": "score"}).text
        assert text.index("Mistral AI") < text.index("Acme") < text.index("Data Analyst")

    def test_period_excludes_old_activity(self, client, populated, config):
        # Vieillit la dernière ligne de l'offre rejetée, en réécrivant le journal.
        path = config.paths.seen_file
        seen = SeenStore(path)
        old = seen.get(populated["rejected"].id)
        old = old.model_copy(
            update={"last_state_change": old.last_state_change - timedelta(days=40)}
        )
        with path.open("a", encoding="utf-8") as handle:
            handle.write(old.model_dump_json() + "\n")

        text = client.get("/offres", params={"days": 30}).text
        assert "Data Analyst" not in text

    def test_invalid_filters_are_ignored(self, client, populated):
        response = client.get("/offres", params={"state": "zzz", "days": "abc"})
        assert response.status_code == 200
        assert "3 offres sur 3" in response.text

    def test_htmx_request_returns_only_rows(self, client, populated):
        response = client.get(
            "/offres", params={"q": "mistral"}, headers={"HX-Request": "true"}
        )
        assert "<html" not in response.text
        assert "Mistral AI" in response.text

    def test_sees_writes_made_while_running(self, client, populated, config):
        client.get("/offres")
        # Une passe CLI lancée à côté du serveur.
        SeenStore(config.paths.seen_file).record(make_job("d", company="Nouvelle"))
        assert "Nouvelle" in client.get("/offres").text


class TestDetail:
    def test_unknown_offer_is_404(self, client, populated):
        assert client.get("/offres/inconnue").status_code == 404

    def test_early_rejection_shows_timeline_without_description(
        self, client, populated
    ):
        text = client.get(f"/offres/{populated['rejected'].id}").text
        assert "Description non récupérée" in text
        assert "titre_hors_sujet" in text
        assert "https://example.com/jobs/a" in text

    def test_scored_offer_shows_fit(self, client, populated):
        text = client.get(f"/offres/{populated['scored'].id}").text
        assert "Bonne adéquation." in text
        assert "LangGraph" in text
        assert "fit:apply(80)" in text

    def test_letter_is_read_from_outbox(self, client, populated):
        text = client.get(f"/offres/{populated['pending'].id}").text
        assert "version corrigée" in text
        assert "Version du modèle." not in text
        assert "Construire des agents." in text
        # En attente : les réserves sont calculées, dont le CV absent du dossier.
        assert "aucun CV joint" in text

    def test_timeline_follows_every_transition(self, client, populated):
        text = client.get(f"/offres/{populated['pending'].id}").text
        timeline = text[text.index('class="timeline"') :]
        assert timeline.count("<li>") == 6


class TestCvVariants:
    @pytest.fixture
    def multi_client(self, config):
        from agent_emploi.config import CvVariant

        config.profile.variants = [
            CvVariant(id="dev", label="Développeur logiciel", markdown="dev.md"),
            CvVariant(id="ml", label="Machine learning", markdown="ml.md"),
        ]
        return TestClient(create_app(config), base_url="http://127.0.0.1")

    @pytest.fixture
    def classified(self, stores):
        seen, jobs = stores
        for slug, company, cv in (("d", "Backendco", "dev"), ("m", "Visionco", "ml")):
            job = make_job(slug, company=company)
            seen.record(job)
            seen.transition(job.id, JobState.PRESCREENED)
            seen.transition(job.id, JobState.FIT_OK)
            jobs.save(job, fit=fit(80).model_copy(update={"cv": cv}))
        return seen, jobs

    def test_single_cv_mode_shows_no_cv_column(self, client, classified):
        text = client.get("/offres").text
        assert "<th>CV</th>" not in text
        assert "Tous les CV" not in text

    def test_offers_show_their_variant(self, multi_client, classified):
        text = multi_client.get("/offres").text
        assert "<th>CV</th>" in text
        assert "Développeur logiciel" in text
        assert "Machine learning" in text

    def test_filter_by_variant(self, multi_client, classified):
        text = multi_client.get("/offres", params={"cv": "ml"}).text
        assert "Visionco" in text
        assert "Backendco" not in text

    def test_unknown_variant_filter_is_ignored(self, multi_client, classified):
        text = multi_client.get("/offres", params={"cv": "zz"}).text
        assert "Visionco" in text and "Backendco" in text

    def test_detail_names_the_variant(self, multi_client, classified):
        seen, _ = classified
        job_id = next(e.id for e in seen.entries() if e.company == "Visionco")
        text = multi_client.get(f"/offres/{job_id}").text
        assert "CV Machine learning" in text
