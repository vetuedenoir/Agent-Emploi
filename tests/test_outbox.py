from datetime import datetime, timezone

import pytest

from agent_emploi.models import FitVerdict, Job, JobRecord, Letter, ReviewVerdict
from agent_emploi.outbox import bundle_path, slugify, write_bundle

AT = datetime(2026, 8, 9, tzinfo=timezone.utc)


def make_record(**fields) -> JobRecord:
    job = Job.build(
        source="wttj",
        url="https://example.com/jobs/1",
        title="Ingénieur IA — Vision",
        company="Acme & Cie",
        location="Paris",
        contract="stage",
        description="Vision par ordinateur en Python.",
        apply_url="https://boards.greenhouse.io/acme/jobs/1",
        ats="greenhouse",
    )
    base = {
        "job": job,
        "fit": FitVerdict(
            score=78,
            verdict="apply",
            matched=["Python"],
            gaps=["Kubernetes"],
            reason="Bon recouvrement.",
            language="fr",
        ),
        "letter": Letter(text="Madame, Monsieur,\n\nJ'ai écrit un MLP.", language="fr"),
        "review": ReviewVerdict(approved=True),
    }
    return JobRecord(**{**base, **fields})


class TestSlugify:
    def test_folds_accents_and_separators(self):
        assert slugify("Ingénieur IA — Vision") == "ingenieur-ia-vision"

    def test_cuts_on_a_word_boundary(self):
        slug = slugify("ingenieur machine learning et vision par ordinateur senior")
        assert len(slug) <= 40
        assert not slug.endswith("-")
        assert slug.split("-")[-1] in {"vision", "par", "ordinateur", "et"}

    def test_never_returns_empty(self):
        assert slugify("!!!") == "sans-titre"


class TestBundlePath:
    def test_names_the_directory_by_date_company_and_title(self, tmp_path):
        path = bundle_path(tmp_path, make_record(), AT)
        assert path.name == "2026-08-09_acme-cie_ingenieur-ia-vision"


class TestWriteBundle:
    @pytest.fixture
    def cv(self, tmp_path):
        path = tmp_path / "cv_fr.pdf"
        path.write_bytes(b"%PDF-fr")
        return path

    def test_writes_every_file(self, tmp_path, cv):
        directory = write_bundle(tmp_path / "outbox", make_record(), cv, at=AT)
        written = {path.name for path in directory.iterdir()}
        assert written == {
            "offre.md",
            "lettre.md",
            "cv.pdf",
            "fit.json",
            "review.json",
            "preview.html",
        }

    def test_letter_file_holds_the_text_and_a_hidden_header(self, tmp_path, cv):
        directory = write_bundle(tmp_path / "outbox", make_record(), cv, at=AT)
        content = (directory / "lettre.md").read_text(encoding="utf-8")
        assert content.startswith("<!--")
        assert "J'ai écrit un MLP." in content

    def test_offer_keeps_the_apply_url(self, tmp_path, cv):
        directory = write_bundle(tmp_path / "outbox", make_record(), cv, at=AT)
        offer = (directory / "offre.md").read_text(encoding="utf-8")
        assert "boards.greenhouse.io" in offer
        assert "Vision par ordinateur" in offer

    def test_cv_is_copied_not_linked(self, tmp_path, cv):
        directory = write_bundle(tmp_path / "outbox", make_record(), cv, at=AT)
        cv.unlink()
        assert (directory / "cv.pdf").read_bytes() == b"%PDF-fr"

    def test_missing_cv_is_reported_in_the_preview(self, tmp_path):
        directory = write_bundle(tmp_path / "outbox", make_record(), None, at=AT)
        assert not (directory / "cv.pdf").exists()
        assert "Aucun CV joint" in (directory / "preview.html").read_text(encoding="utf-8")

    def test_preview_shows_warnings(self, tmp_path, cv):
        record = make_record(
            letter=Letter(
                text="Fort de mon expérience.",
                language="fr",
                banned_hits=["fort de mon expérience"],
            ),
            review=ReviewVerdict(
                approved=False, unsupported_claims=["cinq ans d'expérience"]
            ),
        )
        directory = write_bundle(tmp_path / "outbox", record, cv, at=AT)
        preview = (directory / "preview.html").read_text(encoding="utf-8")
        assert "Formules interdites" in preview
        # `html.escape` remplace l'apostrophe par une entité : le texte source
        # diffère, le rendu non.
        assert "cinq ans d" in preview
        assert "refusé cette lettre" in preview

    def test_preview_escapes_html_from_the_offer(self, tmp_path, cv):
        record = make_record()
        record = record.model_copy(
            update={"job": record.job.model_copy(update={"company": "<script>x</script>"})}
        )
        preview = (
            write_bundle(tmp_path / "outbox", record, cv, at=AT) / "preview.html"
        ).read_text(encoding="utf-8")
        assert "<script>x</script>" not in preview
        assert "&lt;script&gt;" in preview

    def test_a_hand_edited_letter_is_kept_aside(self, tmp_path, cv):
        root = tmp_path / "outbox"
        directory = write_bundle(root, make_record(), cv, at=AT)
        (directory / "lettre.md").write_text("ma version corrigée", encoding="utf-8")

        write_bundle(root, make_record(), cv, at=AT)
        assert (directory / "lettre.originale.md").read_text(
            encoding="utf-8"
        ) == "ma version corrigée"
        assert "J'ai écrit un MLP." in (directory / "lettre.md").read_text(encoding="utf-8")

    def test_rewriting_an_identical_bundle_leaves_no_copy(self, tmp_path, cv):
        root = tmp_path / "outbox"
        directory = write_bundle(root, make_record(), cv, at=AT)
        write_bundle(root, make_record(), cv, at=AT)
        assert not (directory / "lettre.originale.md").exists()

    def test_refuses_a_record_without_a_letter(self, tmp_path, cv):
        with pytest.raises(ValueError, match="aucune lettre"):
            write_bundle(tmp_path / "outbox", make_record(letter=None), cv, at=AT)
