import pytest

from agent_emploi.profile import BannedPhrases, Profile, fold


class TestFold:
    def test_removes_case_and_accents(self):
        assert fold("Fort de mon Expérience") == "fort de mon experience"

    def test_unifies_apostrophes(self):
        assert fold("n’hésitez") == fold("n'hésitez")

    def test_collapses_line_breaks(self):
        assert fold("je suis\n  convaincu") == "je suis convaincu"


class TestBannedPhrases:
    @pytest.fixture
    def phrases(self, tmp_path):
        path = tmp_path / "banned.txt"
        path.write_text(
            "# commentaire\n\nfort de mon expérience\nn'hésitez pas\n",
            encoding="utf-8",
        )
        return BannedPhrases.load(path)

    def test_ignores_comments_and_blank_lines(self, phrases):
        assert len(phrases) == 2

    def test_finds_a_phrase_whatever_its_spelling(self, phrases):
        hits = phrases.find("Fort de mon experience, je vous écris. N’hésitez pas !")
        assert hits == ["fort de mon expérience", "n'hésitez pas"]

    def test_finds_a_phrase_split_across_lines(self, phrases):
        assert phrases.find("fort de\nmon expérience") == ["fort de mon expérience"]

    def test_returns_the_original_spelling_for_display(self, phrases):
        assert phrases.find("FORT DE MON EXPERIENCE") == ["fort de mon expérience"]

    def test_clean_letter_has_no_hit(self, phrases):
        assert phrases.find("J'ai implémenté un perceptron multicouche.") == []

    def test_missing_file_is_not_an_error(self, tmp_path):
        assert len(BannedPhrases.load(tmp_path / "absent.txt")) == 0


class TestProfileLoad:
    @pytest.fixture
    def profile_dir(self, tmp_path, config):
        directory = tmp_path / "profile"
        directory.mkdir()
        (directory / "cv.md").write_text("Python, Tensorflow.", encoding="utf-8")
        (directory / "voice.md").write_text(
            "## Échantillons\n\n```\nJe vouvoie toujours.\n```\n", encoding="utf-8"
        )
        (directory / "banned.txt").write_text("passionné par\n", encoding="utf-8")
        config.profile.cv_markdown = directory / "cv.md"
        config.profile.voice = directory / "voice.md"
        config.profile.banned_phrases = directory / "banned.txt"
        config.profile.cv_fr = directory / "cv_fr.pdf"
        config.profile.cv_en = directory / "cv_en.pdf"
        return config, directory

    def test_loads_every_piece(self, profile_dir):
        config, _ = profile_dir
        profile = Profile.load(config)
        assert "Tensorflow" in profile.cv_text
        assert profile.has_voice
        assert len(profile.banned) == 1

    def test_missing_cv_is_fatal(self, profile_dir):
        config, directory = profile_dir
        (directory / "cv.md").unlink()
        with pytest.raises(FileNotFoundError, match="CV texte"):
            Profile.load(config)

    def test_missing_voice_is_survivable(self, profile_dir):
        config, directory = profile_dir
        (directory / "voice.md").unlink()
        assert not Profile.load(config).has_voice

    def test_unfilled_voice_template_does_not_count(self, profile_dir):
        config, directory = profile_dir
        (directory / "voice.md").write_text(
            "## Échantillons\n\n<!-- colle ici tes textes -->\n```\n```\n"
            "\nÀ compléter.\n",
            encoding="utf-8",
        )
        assert not Profile.load(config).has_voice

    def test_a_partly_filled_voice_counts(self, profile_dir):
        # Une ligne de gabarit oubliée en bas ne doit pas annuler les
        # échantillons déjà écrits plus haut.
        config, directory = profile_dir
        (directory / "voice.md").write_text(
            "## Comment j'écris\n\n```\nPhrases courtes.\n```\n\n"
            "## Échantillons\n\n```\n```\n\nÀ compléter.\n",
            encoding="utf-8",
        )
        profile = Profile.load(config)
        assert profile.has_voice
        assert "colle ici" not in profile.voice

    def test_template_instructions_stay_out_of_the_prompt(self, profile_dir):
        config, directory = profile_dir
        (directory / "voice.md").write_text(
            "> À remplir avant l'étape 4.\n\n<!-- Décris ton registre -->\n"
            "```\nPhrases courtes.\n```\n",
            encoding="utf-8",
        )
        voice = Profile.load(config).voice
        assert "Décris ton registre" not in voice
        assert "À remplir avant" not in voice
        assert "Phrases courtes." in voice

    def test_cv_pdf_follows_the_offer_language(self, profile_dir):
        config, directory = profile_dir
        (directory / "cv_fr.pdf").write_bytes(b"%PDF-fr")
        (directory / "cv_en.pdf").write_bytes(b"%PDF-en")
        profile = Profile.load(config)
        assert profile.cv_pdf("fr").name == "cv_fr.pdf"
        assert profile.cv_pdf("en").name == "cv_en.pdf"

    def test_missing_cv_pdf_returns_none(self, profile_dir):
        config, _ = profile_dir
        assert Profile.load(config).cv_pdf("en") is None
