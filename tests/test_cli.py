import os

import pytest
import yaml

from agent_emploi.cli import main
from tests.conftest import BASE_CONFIG


def write_setup(
    tmp_path,
    *,
    cv_content: str | None = "Python, PyTorch, LLM, RAG",
    sources: list[str] | None = None,
):
    """Écrit un config.yaml et un profil minimal dans tmp_path."""
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    if cv_content is not None:
        (profile_dir / "cv.md").write_text(cv_content, encoding="utf-8")

    data = {
        **BASE_CONFIG,
        "profile": {
            "cv_markdown": str(profile_dir / "cv.md"),
            "cv_fr": str(profile_dir / "cv_fr.pdf"),
            "cv_en": str(profile_dir / "cv_en.pdf"),
            "voice": str(profile_dir / "voice.md"),
            "banned_phrases": str(profile_dir / "banned.txt"),
            "identity": str(profile_dir / "identity.yaml"),
        },
        "paths": {
            "data": str(tmp_path / "data"),
            "outbox": str(tmp_path / "outbox"),
            "applications": str(tmp_path / "applications"),
        },
    }
    if sources is not None:
        data["search"] = {**data["search"], "sources": sources}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return config_path


class TestDoctor:
    def test_succeeds_with_filled_cv(self, tmp_path, capsys):
        config_path = write_setup(tmp_path)
        assert main(["--config", str(config_path), "doctor"]) == 0
        assert "Socle opérationnel" in capsys.readouterr().out

    def test_fails_without_cv(self, tmp_path, capsys):
        config_path = write_setup(tmp_path, cv_content=None)
        assert main(["--config", str(config_path), "doctor"]) == 1
        assert "absent" in capsys.readouterr().out

    def test_fails_on_unfilled_template(self, tmp_path, capsys):
        # Un gabarit créé mais laissé tel quel passerait sinon inaperçu.
        config_path = write_setup(tmp_path, cv_content="# CV\n\nÀ compléter.\n")
        assert main(["--config", str(config_path), "doctor"]) == 1
        assert "gabarit non rempli" in capsys.readouterr().out

    def test_lists_sources_and_warns_on_missing_credentials(
        self, tmp_path, capsys, monkeypatch
    ):
        from agent_emploi.sources.france_travail import ENV_CLIENT_ID, ENV_CLIENT_SECRET

        monkeypatch.delenv(ENV_CLIENT_ID, raising=False)
        monkeypatch.delenv(ENV_CLIENT_SECRET, raising=False)
        config_path = write_setup(tmp_path, sources=["wttj", "france_travail"])

        # Une source non configurée n'est pas bloquante : elle sera ignorée.
        assert main(["--config", str(config_path), "doctor"]) == 0
        out = capsys.readouterr().out
        assert "Sources" in out
        assert ENV_CLIENT_ID in out
        assert "source ignorée" in out

    def test_fails_on_unknown_source(self, tmp_path, capsys):
        # Une faute de frappe dans `config.yaml` que rien ne rattrapera ensuite.
        config_path = write_setup(tmp_path, sources=["france_travial"])
        assert main(["--config", str(config_path), "doctor"]) == 1
        assert "source inconnue" in capsys.readouterr().out

    def test_never_prints_api_key_values(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-secret-value")
        config_path = write_setup(tmp_path)
        main(["--config", str(config_path), "doctor"])
        assert "gsk-secret-value" not in capsys.readouterr().out

    def test_creates_working_directories(self, tmp_path):
        config_path = write_setup(tmp_path)
        main(["--config", str(config_path), "doctor"])
        assert (tmp_path / "data").is_dir()
        assert (tmp_path / "outbox").is_dir()


class TestSourceCredentialPrompt:
    """L'invite d'identifiants de source, en début de passe."""

    def config_with_ft(self, tmp_path, monkeypatch):
        from agent_emploi.sources.france_travail import ENV_CLIENT_ID, ENV_CLIENT_SECRET

        monkeypatch.delenv(ENV_CLIENT_ID, raising=False)
        monkeypatch.delenv(ENV_CLIENT_SECRET, raising=False)
        return write_setup(tmp_path, sources=["france_travail"])

    def test_prompts_on_a_terminal(self, tmp_path, monkeypatch, capsys):
        from agent_emploi.sources.france_travail import ENV_CLIENT_ID, ENV_CLIENT_SECRET

        config_path = self.config_with_ft(tmp_path, monkeypatch)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda prompt="": "identifiant-saisi")
        monkeypatch.setattr(
            "agent_emploi.sources.base.getpass", lambda prompt="": "secret-saisi"
        )

        main(["--config", str(config_path), "search", "--dry-run"])

        assert os.environ[ENV_CLIENT_ID] == "identifiant-saisi"
        assert os.environ[ENV_CLIENT_SECRET] == "secret-saisi"
        assert "secret-saisi" not in capsys.readouterr().out

    def test_no_prompt_outside_a_terminal(self, tmp_path, monkeypatch):
        """Une tâche planifiée ne doit pas se bloquer sur une question."""
        config_path = self.config_with_ft(tmp_path, monkeypatch)
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        monkeypatch.setattr(
            "builtins.input", lambda prompt="": pytest.fail("invite inattendue")
        )

        assert main(["--config", str(config_path), "search", "--dry-run"]) == 1

    def test_no_ask_disables_the_prompt(self, tmp_path, monkeypatch, capsys):
        config_path = self.config_with_ft(tmp_path, monkeypatch)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr(
            "builtins.input", lambda prompt="": pytest.fail("invite inattendue")
        )

        main(["--config", str(config_path), "--no-ask", "search", "--dry-run"])
        assert "variables absentes" in capsys.readouterr().out

    def test_giving_up_leaves_the_source_ignored(self, tmp_path, monkeypatch, capsys):
        config_path = self.config_with_ft(tmp_path, monkeypatch)
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda prompt="": "")
        monkeypatch.setattr("agent_emploi.sources.base.getpass", lambda prompt="": "")

        assert main(["--config", str(config_path), "search", "--dry-run"]) == 1
        assert "ignorée, variables absentes" in capsys.readouterr().out


class TestConfigErrors:
    def test_missing_config_file(self, tmp_path, capsys):
        assert main(["--config", str(tmp_path / "absent.yaml"), "doctor"]) == 1
        assert "introuvable" in capsys.readouterr().err

    def test_invalid_config_file(self, tmp_path, capsys):
        bad = tmp_path / "config.yaml"
        bad.write_text("profile: {}\n", encoding="utf-8")
        assert main(["--config", str(bad), "doctor"]) == 1
        assert "invalide" in capsys.readouterr().err


def test_status_on_empty_store(tmp_path, capsys):
    config_path = write_setup(tmp_path)
    assert main(["--config", str(config_path), "status"]) == 0
    out = capsys.readouterr().out
    assert "Offres connues: 0" in out


class TestDraft:
    def test_reports_when_there_is_nothing_to_draft(self, tmp_path, capsys):
        config_path = write_setup(tmp_path)
        assert main(["--config", str(config_path), "draft"]) == 0
        assert "Aucun dossier préparé" in capsys.readouterr().out

    def test_fails_without_cv(self, tmp_path, capsys):
        config_path = write_setup(tmp_path, cv_content=None)
        assert main(["--config", str(config_path), "draft"]) == 1
        assert "CV texte introuvable" in capsys.readouterr().err

    def test_warns_when_the_voice_file_is_missing(self, tmp_path, capsys):
        # Sans échantillons d'écriture, les lettres sonnent générées : c'est
        # l'avertissement le plus utile de l'étape 4.
        config_path = write_setup(tmp_path)
        main(["--config", str(config_path), "draft"])
        assert "sonneront génériques" in capsys.readouterr().out


class TestReview:
    """La commande où le système rend la main : rien n'avance sans décision."""

    def stage_bundle(self, tmp_path, config_path):
        """Écrit un dossier prêt à valider et retourne (offre, dossier)."""
        from agent_emploi.config import load_config
        from agent_emploi.models import FitVerdict, Job, JobState, Letter, ReviewVerdict
        from agent_emploi.outbox import write_bundle
        from agent_emploi.store.jobs import JobStore
        from agent_emploi.store.seen import SeenStore

        config = load_config(config_path)
        config.paths.ensure()
        job = Job.build(
            source="fake",
            url="https://example.com/jobs/1",
            title="Ingénieur IA",
            company="Acme",
            description="Agents LLM.",
        )
        seen = SeenStore(config.paths.seen_file)
        seen.record(job)
        for target in (
            JobState.PRESCREENED,
            JobState.FIT_OK,
            JobState.DRAFTED,
            JobState.REVIEWED,
            JobState.AWAITING_USER,
        ):
            seen.advance(job.id, target)

        store = JobStore(config.paths.jobs_file)
        record = store.save(
            job,
            fit=FitVerdict(
                score=80, verdict="apply", reason="ok", language="fr"
            ),
            letter=Letter(text=" ".join(["mot"] * 170), language="fr"),
            review=ReviewVerdict(approved=True),
        )
        directory = write_bundle(config.paths.outbox, record, None)
        store.save(job, outbox=str(directory))
        return job, directory

    def test_reports_when_there_is_nothing_to_review(self, tmp_path, capsys):
        config_path = write_setup(tmp_path)
        assert main(["--config", str(config_path), "review"]) == 0
        assert "Aucun dossier en attente" in capsys.readouterr().out

    def test_list_decides_nothing(self, tmp_path, capsys):
        config_path = write_setup(tmp_path)
        job, _ = self.stage_bundle(tmp_path, config_path)
        assert main(["--config", str(config_path), "review", "--list"]) == 0

        from agent_emploi.config import load_config
        from agent_emploi.models import JobState
        from agent_emploi.store.seen import SeenStore

        config = load_config(config_path)
        assert "Acme" in capsys.readouterr().out
        assert SeenStore(config.paths.seen_file).get(job.id).state is (
            JobState.AWAITING_USER
        )

    def test_approve_without_the_menu(self, tmp_path, capsys):
        config_path = write_setup(tmp_path)
        job, _ = self.stage_bundle(tmp_path, config_path)
        code = main(
            ["--config", str(config_path), "review", job.id[:8], "--approve"]
        )
        assert code == 0

        from agent_emploi.config import load_config
        from agent_emploi.models import JobState
        from agent_emploi.store.seen import SeenStore

        config = load_config(config_path)
        assert SeenStore(config.paths.seen_file).get(job.id).state is JobState.APPROVED
        assert "approuvé" in capsys.readouterr().out

    def test_reject_carries_the_note(self, tmp_path):
        config_path = write_setup(tmp_path)
        job, _ = self.stage_bundle(tmp_path, config_path)
        main(
            [
                "--config", str(config_path), "review", job.id[:8],
                "--reject", "--note", "trop loin",
            ]
        )

        from agent_emploi.config import load_config
        from agent_emploi.models import JobState
        from agent_emploi.store.seen import SeenStore

        entry = SeenStore(load_config(config_path).paths.seen_file).get(job.id)
        assert entry.state is JobState.REJECTED
        assert "trop loin" in entry.reason

    def test_unknown_reference_fails_loudly(self, tmp_path, capsys):
        config_path = write_setup(tmp_path)
        self.stage_bundle(tmp_path, config_path)
        code = main(["--config", str(config_path), "review", "zzzz", "--approve"])
        assert code == 1
        assert "aucun dossier" in capsys.readouterr().err

    def test_approve_and_reject_are_exclusive(self, tmp_path, capsys):
        config_path = write_setup(tmp_path)
        code = main(
            ["--config", str(config_path), "review", "abcd", "--approve", "--reject"]
        )
        assert code == 1
        assert "s'excluent" in capsys.readouterr().err

    def test_status_separates_waiting_from_approved(self, tmp_path, capsys):
        config_path = write_setup(tmp_path)
        job, _ = self.stage_bundle(tmp_path, config_path)
        main(["--config", str(config_path), "status"])
        assert "en attente de validation" in capsys.readouterr().out

        main(["--config", str(config_path), "review", job.id[:8], "--approve"])
        capsys.readouterr()
        main(["--config", str(config_path), "status"])
        assert "approuvés" in capsys.readouterr().out


class TestApply:
    """La commande de candidature, sur les chemins qui n'ouvrent aucun navigateur."""

    def test_init_identity_ecrit_le_gabarit(self, tmp_path, capsys):
        config_path = write_setup(tmp_path)
        assert main(["--config", str(config_path), "apply", "--init-identity"]) == 0

        out = capsys.readouterr().out
        assert "gabarit écrit" in out
        assert (tmp_path / "profile" / "identity.yaml").exists()

    def test_etat_civil_absent_arrete_avant_le_navigateur(self, tmp_path, capsys):
        """Aucun Chromium n'est lancé : l'erreur tombe avant."""
        config_path = write_setup(tmp_path)
        assert main(["--config", str(config_path), "apply"]) == 1
        assert "état civil introuvable" in capsys.readouterr().err

    def test_rien_a_faire_quand_aucun_dossier_nest_approuve(self, tmp_path, capsys):
        config_path = write_setup(tmp_path)
        main(["--config", str(config_path), "apply", "--init-identity"])
        capsys.readouterr()

        assert main(["--config", str(config_path), "apply"]) == 0
        assert "Aucun dossier approuvé" in capsys.readouterr().out

    def test_doctor_signale_letat_civil_a_completer(self, tmp_path, capsys):
        config_path = write_setup(tmp_path)
        main(["--config", str(config_path), "apply", "--init-identity"])
        capsys.readouterr()

        main(["--config", str(config_path), "doctor"])
        assert "état civil" in capsys.readouterr().out

    def test_marquer_envoye_exige_une_candidature_preparee(self, tmp_path, capsys):
        config_path = write_setup(tmp_path)
        job, _ = TestReview.stage_bundle(self, tmp_path, config_path)
        main(["--config", str(config_path), "review", job.id[:8], "--approve"])
        capsys.readouterr()

        # Aucun formulaire n'a été rempli : il n'y a rien à déclarer envoyé.
        assert main(["--config", str(config_path), "apply", job.id[:8], "--sent"]) == 1
        assert "aucune candidature préparée" in capsys.readouterr().err

    def test_marquer_envoye_apres_remplissage(self, tmp_path, capsys):
        from agent_emploi.config import load_config
        from agent_emploi.models import ApplyOutcome, JobState
        from agent_emploi.store.jobs import JobStore
        from agent_emploi.store.seen import SeenStore

        config_path = write_setup(tmp_path)
        job, _ = TestReview.stage_bundle(self, tmp_path, config_path)
        main(["--config", str(config_path), "review", job.id[:8], "--approve"])
        capsys.readouterr()

        config = load_config(config_path)
        store = JobStore(config.paths.jobs_file)
        store.save(
            job,
            application=ApplyOutcome(status="prefilled", apply_url="https://ats/x"),
        )
        SeenStore(config.paths.seen_file).transition(job.id, JobState.PREFILLED)

        assert main(["--config", str(config_path), "apply", job.id[:8], "--sent"]) == 0
        assert "envoi enregistré" in capsys.readouterr().out
        assert SeenStore(config.paths.seen_file).get(job.id).state is JobState.SUBMITTED


class TestArchive:
    """Le classement des candidatures closes, une fois l'envoi déclaré."""

    def sent_bundle(self, tmp_path, config_path):
        """Une candidature menée jusqu'à l'envoi déclaré par l'utilisateur."""
        from agent_emploi.config import load_config
        from agent_emploi.models import ApplyOutcome, JobState
        from agent_emploi.store.jobs import JobStore
        from agent_emploi.store.seen import SeenStore

        job, directory = TestReview.stage_bundle(self, tmp_path, config_path)
        main(["--config", str(config_path), "review", job.id[:8], "--approve"])

        config = load_config(config_path)
        JobStore(config.paths.jobs_file).save(
            job,
            application=ApplyOutcome(
                status="prefilled", apply_url="https://ats/x", submitted=True
            ),
        )
        seen = SeenStore(config.paths.seen_file)
        seen.transition(job.id, JobState.PREFILLED)
        seen.transition(job.id, JobState.SUBMITTED, "utilisateur:envoyé")
        return job, directory

    def test_rien_a_archiver_sans_envoi_declare(self, tmp_path, capsys):
        config_path = write_setup(tmp_path)
        TestReview.stage_bundle(self, tmp_path, config_path)
        capsys.readouterr()

        assert main(["--config", str(config_path), "archive"]) == 0
        assert "Aucune candidature à archiver" in capsys.readouterr().out

    def test_classe_la_candidature_envoyee(self, tmp_path, capsys):
        config_path = write_setup(tmp_path)
        _, directory = self.sent_bundle(tmp_path, config_path)
        capsys.readouterr()

        assert main(["--config", str(config_path), "archive"]) == 0
        out = capsys.readouterr().out
        assert "1 candidature(s) archivée(s)" in out
        assert not directory.exists()

        archived = list((tmp_path / "applications").iterdir())
        assert len(archived) == 1
        assert (archived[0] / "README.md").exists()
        assert (archived[0] / "lettre.md").exists()

    def test_passe_a_blanc_ne_deplace_rien(self, tmp_path, capsys):
        config_path = write_setup(tmp_path)
        _, directory = self.sent_bundle(tmp_path, config_path)
        capsys.readouterr()

        assert main(["--config", str(config_path), "archive", "--dry-run"]) == 0
        assert "rien n'a été déplacé" in capsys.readouterr().out
        assert directory.exists()
        assert not any((tmp_path / "applications").iterdir())


class TestRun:
    """La boucle bout en bout : elle s'arrête à la validation, sans navigateur."""

    def test_fails_without_cv(self, tmp_path, capsys):
        config_path = write_setup(tmp_path, cv_content=None)
        assert main(["--config", str(config_path), "run"]) == 1
        assert "CV texte introuvable" in capsys.readouterr().err

    def test_source_injoignable_ne_casse_pas_la_boucle(self, tmp_path, capsys,
                                                       monkeypatch):
        """La source échoue, la boucle va au bout et le signale — code 1."""
        monkeypatch.setattr("agent_emploi.search.REGISTRY", {})
        config_path = write_setup(tmp_path)

        assert main(["--config", str(config_path), "run"]) == 1

        out = capsys.readouterr().out
        assert "source inconnue" in out
        assert "Aucun dossier en attente" in out
        assert "Consommation LLM" in out

    def test_annonce_les_dossiers_a_valider(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setattr("agent_emploi.search.REGISTRY", {})
        config_path = write_setup(tmp_path)
        TestReview.stage_bundle(self, tmp_path, config_path)
        capsys.readouterr()

        main(["--config", str(config_path), "run"])

        out = capsys.readouterr().out
        assert "1 dossier(s) attendent votre validation" in out
        assert "python -m agent_emploi review" in out


class TestLoadDotenv:
    """Les clés d'API vivent dans `.env` ; `config.yaml` n'en contient aucune."""

    def test_loads_pairs(self, tmp_path, monkeypatch):
        from agent_emploi.config import load_dotenv

        env = tmp_path / ".env"
        env.write_text(
            "# commentaire\nGROQ_API_KEY=abc\nexport GEMINI_API_KEY='def'\nvide=\n",
            encoding="utf-8",
        )
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        assert load_dotenv(env) == ["GROQ_API_KEY", "GEMINI_API_KEY"]
        assert os.environ["GROQ_API_KEY"] == "abc"
        assert os.environ["GEMINI_API_KEY"] == "def"

    def test_existing_environment_wins(self, tmp_path, monkeypatch):
        from agent_emploi.config import load_dotenv

        env = tmp_path / ".env"
        env.write_text("GROQ_API_KEY=depuis_le_fichier\n", encoding="utf-8")
        monkeypatch.setenv("GROQ_API_KEY", "deja_exporte")

        assert load_dotenv(env) == []
        assert os.environ["GROQ_API_KEY"] == "deja_exporte"

    def test_missing_file_is_not_an_error(self, tmp_path):
        from agent_emploi.config import load_dotenv

        assert load_dotenv(tmp_path / "absent") == []
