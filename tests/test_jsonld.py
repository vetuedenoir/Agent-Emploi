from pathlib import Path

import httpx
import pytest
import respx

from agent_emploi.web import jsonld

FIXTURE = Path(__file__).parent / "fixtures" / "jobposting.html"


class TestParse:
    def test_reads_job_posting_inside_graph(self):
        found = jsonld.parse(FIXTURE.read_text(encoding="utf-8"))
        assert found.structured
        assert found.title == "Stage IA générative & agents"
        assert found.company == "Acme"
        assert found.location == "Paris, France"
        assert found.contract == "Stage, FULL_TIME"
        assert found.apply_url == "https://boards.example.com/acme/apply/42"
        assert found.posted_at.year == 2026
        assert found.notes == []

    def test_description_becomes_text_with_list_items(self):
        found = jsonld.parse(FIXTURE.read_text(encoding="utf-8"))
        assert found.description == "Vous rejoignez l'équipe \"agents\".\n\n- Python\n- RAG"

    def test_falls_back_to_page_title(self):
        found = jsonld.parse("<html><title> Offre  Acme </title></html>")
        assert not found.structured
        assert found.title == "Offre Acme"
        assert found.notes

    def test_falls_back_to_opengraph(self):
        page = (
            "<title>Senior Data Scientist | Artefact | Greenhouse</title>"
            '<meta content="Senior Data Scientist" property="og:title">'
            "<meta property='og:site_name' content='Artefact &amp; Co'>"
        )
        found = jsonld.parse(page)
        assert (found.title, found.company) == ("Senior Data Scientist", "Artefact & Co")
        assert not found.structured

    def test_broken_block_is_skipped(self):
        page = (
            '<script type="application/ld+json">{oops</script>'
            '<script type="application/ld+json">{"@type": "JobPosting", '
            '"title": "ML", "hiringOrganization": "Beta"}</script>'
        )
        found = jsonld.parse(page)
        assert (found.title, found.company) == ("ML", "Beta")
        assert "description" in found.notes[0]


class TestFetch:
    @respx.mock
    def test_prefill_downloads_page(self):
        respx.get("https://jobs.example.com/42").mock(
            return_value=httpx.Response(200, text=FIXTURE.read_text(encoding="utf-8"))
        )
        assert jsonld.prefill("https://jobs.example.com/42").company == "Acme"

    @respx.mock
    def test_http_error_is_reported(self):
        respx.get("https://jobs.example.com/42").mock(return_value=httpx.Response(403))
        with pytest.raises(jsonld.PrefillError, match="403"):
            jsonld.prefill("https://jobs.example.com/42")

    def test_only_http_urls(self):
        with pytest.raises(jsonld.PrefillError):
            jsonld.prefill("file:///etc/passwd")
