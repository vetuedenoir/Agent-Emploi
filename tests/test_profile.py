import pytest

from agent_emploi.config import CvVariant, ProfileConfig
from agent_emploi.profile import (
    BannedPhrases,
    CvText,
    Profile,
    fold,
    load_cvs,
    strip_data_uris,
)


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


class TestStripDataUris:
    """Une photo d'identité en base64 pèse plus que tout le reste du CV.

    Le CV part dans chaque prompt : la laisser passer suffit à dépasser la
    limite de tokens par minute du tier gratuit et à faire échouer la passe.
    """

    def test_removes_a_markdown_image(self):
        text = "## CV\n\n![](data:image/png;base64,%s)\n\nPython." % ("A" * 5000)
        cleaned = strip_data_uris(text)
        assert "base64" not in cleaned
        assert cleaned.startswith("## CV")
        assert cleaned.endswith("Python.")

    def test_removes_an_html_image(self):
        cleaned = strip_data_uris('<img src="data:image/png;base64,AAAA"/>\nPython.')
        assert cleaned == "Python."

    def test_removes_a_link_reference(self):
        cleaned = strip_data_uris("[photo]: data:image/png;base64,AAAA\nPython.")
        assert cleaned == "Python."

    def test_keeps_ordinary_images_and_links(self):
        text = "![photo](photo.png)\n[site](https://example.com)"
        assert strip_data_uris(text) == text


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

    def test_cv_is_stripped_of_encoded_images(self, profile_dir):
        config, directory = profile_dir
        (directory / "cv.md").write_text(
            "![](data:image/png;base64,%s)\n\nPython, Tensorflow." % ("A" * 20000),
            encoding="utf-8",
        )
        profile = Profile.load(config)
        assert profile.cv_text == "Python, Tensorflow."

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


class TestVariants:
    @pytest.fixture
    def variants_config(self, tmp_path, config):
        directory = tmp_path / "profile"
        directory.mkdir()
        for name, text in (("dev.md", "Backend Python."), ("ml.md", "Keras, vision.")):
            (directory / name).write_text(text, encoding="utf-8")
        config.profile.variants = [
            CvVariant(id="dev", label="Développeur", markdown=directory / "dev.md"),
            CvVariant(id="ml", label="Machine learning", markdown=directory / "ml.md"),
        ]
        config.profile.cv_markdown = None
        config.profile.voice = directory / "voice.md"
        config.profile.banned_phrases = directory / "banned.txt"
        return config

    def test_single_cv_mode_has_no_identifier(self, config, tmp_path):
        config.profile.cv_markdown = tmp_path / "cv.md"
        config.profile.cv_markdown.write_text("Python.", encoding="utf-8")
        assert load_cvs(config) == [CvText(None, None, "Python.")]

    def test_loads_every_variant(self, variants_config):
        cvs = load_cvs(variants_config)
        assert [(cv.id, cv.text) for cv in cvs] == [
            ("dev", "Backend Python."),
            ("ml", "Keras, vision."),
        ]

    def test_missing_variant_file_fails(self, variants_config):
        variants_config.profile.variants[1].markdown.unlink()
        with pytest.raises(FileNotFoundError):
            load_cvs(variants_config)

    def test_for_cv_switches_the_cv_text(self, variants_config):
        profile = Profile.load(variants_config)
        assert profile.cv_text == "Backend Python."
        assert profile.for_cv("ml").cv_text == "Keras, vision."
        assert profile.for_cv(None).cv_text == "Backend Python."
        assert profile.for_cv("retirée").cv_text == "Backend Python."
        assert profile.label("ml") == "Machine learning"

    def test_config_requires_a_cv(self):
        with pytest.raises(ValueError):
            ProfileConfig(cv_fr="a", cv_en="b", voice="c", banned_phrases="d")

    def test_config_rejects_duplicate_ids(self):
        variant = {"id": "dev", "label": "Dev", "markdown": "dev.md"}
        with pytest.raises(ValueError):
            ProfileConfig(
                cv_fr="a", cv_en="b", voice="c", banned_phrases="d",
                variants=[variant, variant],
            )
