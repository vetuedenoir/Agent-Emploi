"""Tests de la source Welcome to the Jungle.

Les tests unitaires rejouent des réponses enregistrées (`tests/fixtures/`) : ils
sont déterministes et sans réseau. Le test marqué `live` interroge le vrai
service — c'est lui qui détecte une rupture du contrat non documenté sur lequel
repose cette source.
"""

import json
from pathlib import Path

import httpx
import pytest

from agent_emploi.sources.base import SearchQuery, SourceError
from agent_emploi.sources.wttj import WttjSource, html_to_text

FIXTURES = Path(__file__).parent / "fixtures"
SEARCH = json.loads((FIXTURES / "wttj_search.json").read_text(encoding="utf-8"))
DETAIL = json.loads((FIXTURES / "wttj_detail.json").read_text(encoding="utf-8"))


def fake_source(handler) -> WttjSource:
    """Source branchée sur un transport simulé, sans réseau ni pause."""
    client = httpx.Client(transport=httpx.MockTransport(handler), timeout=5)
    return WttjSource(
        client=client, app_id="APP", api_key="KEY", request_delay=0.0
    )


def route(*, search=SEARCH, detail=DETAIL, search_status=200, detail_status=200):
    """Routeur simulé : Algolia, API v3, et le point d'entrée /api/env."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "algolia.net" in url:
            return httpx.Response(search_status, json=search)
        if "/api/v3/organizations/" in url:
            return httpx.Response(detail_status, json=detail)
        if "/api/env" in url:
            return httpx.Response(202, text="<html>défi anti-bot</html>")
        return httpx.Response(404)

    return handler


class TestHtmlToText:
    def test_strips_tags_and_keeps_structure(self):
        html = "<h2>Missions</h2><ul><li>Entraîner</li><li>Déployer</li></ul>"
        text = html_to_text(html)
        assert "<" not in text
        assert "- Entraîner" in text
        assert "- Déployer" in text

    def test_unescapes_entities(self):
        assert html_to_text("<p>R&amp;D &gt; 50&nbsp;%</p>").startswith("R&D > 50")

    def test_line_breaks_become_newlines(self):
        assert html_to_text("a<br/>b") == "a\nb"

    def test_collapses_excess_blank_lines(self):
        assert "\n\n\n" not in html_to_text("<p>a</p><p></p><p></p><p>b</p>")

    def test_empty_input(self):
        assert html_to_text("") == ""


class TestSearch:
    def test_returns_normalised_jobs(self):
        jobs = fake_source(route()).search(SearchQuery(text="ml", limit=10))
        assert jobs
        job = jobs[0]
        assert job.source == "wttj"
        assert job.title
        assert job.company
        assert str(job.url).startswith("https://www.welcometothejungle.com/fr/companies/")
        assert job.source_ref["org_slug"] and job.source_ref["job_slug"]

    def test_deduplicates_partner_job_boards(self):
        """La même offre est indexée une fois par job board qui la diffuse.

        Les hits ont des objectID distincts mais les mêmes slugs : sans
        dédoublonnage, une seule offre remplirait toute la page de résultats.
        """
        jobs = fake_source(route()).search(SearchQuery(text="ml", limit=50))
        raw_ids = {hit["objectID"] for hit in SEARCH["hits"]}
        slugs = {(h["organization"]["slug"], h["slug"]) for h in SEARCH["hits"]}

        assert len(raw_ids) > len(slugs), "la fixture doit contenir des doublons"
        assert len(jobs) == len(slugs)
        assert len({job.id for job in jobs}) == len(jobs)

    def test_respects_limit(self):
        jobs = fake_source(route()).search(SearchQuery(text="ml", limit=1))
        assert len(jobs) == 1

    def test_skips_hits_without_slugs(self):
        broken = {**SEARCH, "hits": [{"objectID": "1", "name": "Sans slug"}]}
        assert fake_source(route(search=broken)).search(SearchQuery(text="ml")) == []

    def test_filters_out_stale_offers(self):
        aged = {
            **SEARCH,
            "hits": [{**SEARCH["hits"][0], "published_at": "2020-01-01T00:00:00Z"}],
        }
        source = fake_source(route(search=aged))
        assert source.search(SearchQuery(text="ml", max_age_days=30)) == []
        assert source.search(SearchQuery(text="ml")) != []

    def test_http_error_raises_source_error(self):
        source = fake_source(route(search_status=403))
        with pytest.raises(SourceError, match="refusée"):
            source.search(SearchQuery(text="ml"))

    def test_malformed_payload_raises_source_error(self):
        source = fake_source(route(search={"unexpected": True}))
        with pytest.raises(SourceError, match="hits"):
            source.search(SearchQuery(text="ml"))


class TestFacetFilters:
    def test_maps_french_contract_names(self):
        source = fake_source(route())
        filters = source._facet_filters(SearchQuery(text="", contracts=["CDI", "stage"]))
        flat = [value for group in filters for value in group]
        assert "contract_type:FULL_TIME" in flat
        assert "contract_type:INTERNSHIP" in flat

    def test_unknown_contract_is_ignored_not_fatal(self, caplog):
        source = fake_source(route())
        filters = source._facet_filters(SearchQuery(text="", contracts=["CDI", "portage"]))
        flat = [value for group in filters for value in group]
        assert flat == ["contract_type:FULL_TIME"]
        assert "portage" in caplog.text

    def test_country_and_remote(self):
        source = fake_source(route())
        filters = source._facet_filters(
            SearchQuery(text="", countries=["France"], remote=["full"])
        )
        flat = [value for group in filters for value in group]
        assert "offices.country:France" in flat
        assert "remote:full" in flat

    def test_no_criteria_means_no_filters(self):
        assert fake_source(route())._facet_filters(SearchQuery(text="")) == []


class TestEnrich:
    def test_adds_description_and_apply_url(self):
        source = fake_source(route())
        job = source.search(SearchQuery(text="ml", limit=1))[0]
        assert not job.is_enriched

        enriched = source.enrich(job)
        assert enriched.is_enriched
        assert "<" not in enriched.description
        assert enriched.apply_url == DETAIL["job"]["apply_url"]
        assert enriched.ats == DETAIL["job"]["ats"]
        assert enriched.id == job.id

    def test_failure_returns_job_unchanged(self):
        """Une offre sans détail ne doit pas faire échouer toute la passe."""
        source = fake_source(route(detail_status=500))
        job = source.search(SearchQuery(text="ml", limit=1))[0]

        enriched = source.enrich(job)
        assert enriched == job

    def test_job_without_slugs_is_returned_as_is(self):
        source = fake_source(route())
        job = source.search(SearchQuery(text="ml", limit=1))[0]
        stripped = job.model_copy(update={"source_ref": {}})
        assert source.enrich(stripped) == stripped


class TestCredentials:
    def test_explicit_credentials_skip_network(self):
        def refuse(request: httpx.Request) -> httpx.Response:
            raise AssertionError(f"aucun appel attendu, reçu {request.url}")

        source = WttjSource(
            client=httpx.Client(transport=httpx.MockTransport(refuse)),
            app_id="APP",
            api_key="KEY",
        )
        assert source._credentials() == ("APP", "KEY")

    def test_falls_back_when_env_is_challenged(self):
        """Le pare-feu du site sert un défi anti-bot : le repli doit être silencieux."""
        from agent_emploi.sources.wttj import FALLBACK_API_KEY, FALLBACK_APP_ID

        source = WttjSource(
            client=httpx.Client(transport=httpx.MockTransport(route())),
            request_delay=0.0,
        )
        assert source._credentials() == (FALLBACK_APP_ID, FALLBACK_API_KEY)

    def test_reads_env_when_available(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                text='window.env = {"PUBLIC_ALGOLIA_APPLICATION_ID":"NEW_APP",'
                '"PUBLIC_ALGOLIA_API_KEY_CLIENT":"NEW_KEY"};',
            )

        source = WttjSource(client=httpx.Client(transport=httpx.MockTransport(handler)))
        assert source._credentials() == ("NEW_APP", "NEW_KEY")


@pytest.mark.live
class TestLiveContract:
    """Détecte une rupture du contrat non documenté sur lequel repose la source.

    Ces points d'entrée ne sont garantis par personne : clés renouvelées, index
    renommé, endpoint v3 retiré. En cas d'échec ici, ne pas contourner — relever
    les nouvelles valeurs et mettre à jour `wttj.py`.
    """

    def test_search_and_enrich_end_to_end(self):
        with WttjSource() as source:
            jobs = source.search(
                SearchQuery(text="machine learning", countries=["France"], limit=3)
            )
            assert jobs, "l'index Algolia ne renvoie plus de résultats"
            assert len({job.id for job in jobs}) == len(jobs)

            enriched = source.enrich(jobs[0])
            assert enriched.is_enriched, "l'API v3 ne renvoie plus de description"
            assert len(enriched.description) > 200
