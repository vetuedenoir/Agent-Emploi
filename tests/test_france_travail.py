"""Tests de la source France Travail.

Les tests unitaires rejouent des réponses enregistrées : ni réseau, ni
identifiants. Le test marqué `live` interroge la vraie API — il vérifie que le
jeton s'obtient toujours et que les codes de contrat employés existent encore
dans les référentiels ; c'est lui qui détecterait une rupture de contrat.
"""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs

import httpx
import pytest

from agent_emploi.sources.base import (
    SearchQuery,
    SourceError,
    missing_env,
    prompt_missing_env,
)
from agent_emploi.sources.france_travail import (
    ENV_CLIENT_ID,
    ENV_CLIENT_SECRET,
    FranceTravailSource,
    published_since,
)

FIXTURES = Path(__file__).parent / "fixtures"
SEARCH = json.loads((FIXTURES / "france_travail_search.json").read_text(encoding="utf-8"))

#: Référentiels recopiés de l'API réelle, et non inventés. Deux détails y sont
#: contre-intuitifs et portent la moitié des tests de ce fichier : les libellés
#: sont abrégés (« Cont. professionnalisation »), et **aucune nature « stage »
#: n'existe**. Le test `live` vérifie que ces deux faits tiennent toujours.
TYPES_CONTRATS = [
    {"code": "CCE", "libelle": "Profession commerciale"},
    {"code": "CDD", "libelle": "Contrat à durée déterminée"},
    {"code": "CDI", "libelle": "Contrat à durée indéterminée"},
    {"code": "DDI", "libelle": "Contrat durée déterminée insertion"},
    {"code": "DIN", "libelle": "CDI Intérimaire"},
    {"code": "LIB", "libelle": "Profession libérale"},
    {"code": "MIS", "libelle": "Mission intérimaire"},
    {"code": "SAI", "libelle": "Contrat travail saisonnier"},
]
NATURES_CONTRATS = [
    {"code": "CC", "libelle": "CDI de chantier ou d'opération"},
    {"code": "E1", "libelle": "Contrat travail"},
    {"code": "E2", "libelle": "Contrat apprentissage"},
    {"code": "FS", "libelle": "Cont. professionnalisation"},
    {"code": "FT", "libelle": "CUI - CAE"},
    {"code": "PS", "libelle": "Portage salarial"},
]


class Recorder:
    """Routeur simulé : jeton, référentiels, recherche, détail.

    Enregistre les requêtes reçues pour qu'un test puisse vérifier ce qui a été
    demandé — et surtout ce qui ne l'a pas été (jeton non renouvelé, référentiel
    chargé une seule fois).
    """

    def __init__(
        self,
        *,
        search=SEARCH,
        search_status=200,
        token_status=200,
        referentiel_status=200,
        expires_in=1499,
        first_call_401=False,
    ) -> None:
        self.search = search
        self.search_status = search_status
        self.token_status = token_status
        self.referentiel_status = referentiel_status
        self.expires_in = expires_in
        self.first_call_401 = first_call_401
        self.requests: list[httpx.Request] = []

    @property
    def paths(self) -> list[str]:
        return [request.url.path for request in self.requests]

    def params(self, needle: str) -> dict:
        """Paramètres de la dernière requête dont le chemin contient `needle`."""
        for request in reversed(self.requests):
            if needle in request.url.path:
                return dict(request.url.params)
        raise AssertionError(f"aucune requête vers {needle}")

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = str(request.url)

        if "access_token" in url:
            if self.token_status != 200:
                return httpx.Response(self.token_status, json={"error": "invalid_client"})
            return httpx.Response(
                200,
                json={
                    "access_token": f"jeton-{sum('access_token' in str(r.url) for r in self.requests)}",
                    "token_type": "Bearer",
                    "expires_in": self.expires_in,
                },
            )

        if "/referentiel/typesContrats" in url:
            return httpx.Response(self.referentiel_status, json=TYPES_CONTRATS)
        if "/referentiel/naturesContrats" in url:
            return httpx.Response(self.referentiel_status, json=NATURES_CONTRATS)

        if "/offres/search" in url:
            if self.first_call_401 and not any(
                r.url.path.endswith("/search") and r is not request for r in self.requests
            ):
                return httpx.Response(401, json={"error": "expired"})
            if self.search_status == 204:
                return httpx.Response(204)
            return httpx.Response(self.search_status, json=self.search)

        if "/offres/" in url:
            return httpx.Response(200, json=self.search["resultats"][0])

        return httpx.Response(404)


def fake_source(recorder: Recorder | None = None) -> tuple[FranceTravailSource, Recorder]:
    """Source branchée sur un transport simulé, sans réseau ni pause."""
    recorder = recorder or Recorder()
    client = httpx.Client(transport=httpx.MockTransport(recorder), timeout=5)
    return (
        FranceTravailSource(
            client=client, client_id="ID", client_secret="SECRET", request_delay=0.0
        ),
        recorder,
    )


class TestCredentials:
    def test_missing_credentials_are_named(self, monkeypatch):
        monkeypatch.delenv(ENV_CLIENT_ID, raising=False)
        monkeypatch.delenv(ENV_CLIENT_SECRET, raising=False)
        source = FranceTravailSource(
            client=httpx.Client(transport=httpx.MockTransport(Recorder()))
        )
        with pytest.raises(SourceError) as excinfo:
            source.search(SearchQuery(text="ia"))
        assert ENV_CLIENT_ID in str(excinfo.value)
        assert ENV_CLIENT_SECRET in str(excinfo.value)

    def test_credentials_come_from_environment(self, monkeypatch):
        monkeypatch.setenv(ENV_CLIENT_ID, "depuis-env")
        monkeypatch.setenv(ENV_CLIENT_SECRET, "secret-env")
        recorder = Recorder()
        source = FranceTravailSource(
            client=httpx.Client(transport=httpx.MockTransport(recorder)),
            request_delay=0.0,
        )
        source.search(SearchQuery(text="ia", limit=1))
        sent = parse_qs(recorder.requests[0].content.decode())
        assert sent["client_id"] == ["depuis-env"]

    def test_required_env_drives_missing_env(self, monkeypatch):
        monkeypatch.delenv(ENV_CLIENT_ID, raising=False)
        monkeypatch.setenv(ENV_CLIENT_SECRET, "secret")
        assert missing_env(FranceTravailSource) == [ENV_CLIENT_ID]

    def test_auth_failure_is_reported_not_swallowed(self):
        source, _ = fake_source(Recorder(token_status=400))
        with pytest.raises(SourceError, match="authentification refusée"):
            source.search(SearchQuery(text="ia"))

    def test_token_is_reused_across_requests(self):
        """Un jeton dure vingt minutes : le renouveler par requête serait absurde."""
        source, recorder = fake_source()
        source.search(SearchQuery(text="ia", limit=2))
        source.search(SearchQuery(text="ml", limit=2))
        assert sum("access_token" in path for path in recorder.paths) == 1

    def test_expired_token_is_renewed(self):
        source, recorder = fake_source(Recorder(expires_in=0))
        source.search(SearchQuery(text="ia", limit=2))
        source.search(SearchQuery(text="ml", limit=2))
        assert sum("access_token" in path for path in recorder.paths) == 2

    def test_rejected_token_triggers_one_retry(self):
        source, recorder = fake_source(Recorder(first_call_401=True))
        jobs = source.search(SearchQuery(text="ia", limit=1))
        assert jobs
        assert sum("access_token" in path for path in recorder.paths) == 2


class TestInteractiveCredentials:
    """Saisie des identifiants à l'invite, comme `apply --login` pour les sites.

    Le principe : ne pas obliger à écrire un secret dans un fichier. Ce qui est
    saisi ne vit que le temps de la commande.
    """

    def test_prompts_only_for_what_is_missing(self, monkeypatch):
        monkeypatch.setenv(ENV_CLIENT_ID, "déjà-là")
        monkeypatch.delenv(ENV_CLIENT_SECRET, raising=False)
        asked: list[str] = []

        filled = prompt_missing_env(
            FranceTravailSource,
            ask=lambda prompt: asked.append(("clair", prompt)) or "x",
            ask_secret=lambda prompt: asked.append(("masqué", prompt)) or "secret-saisi",
        )

        assert filled == [ENV_CLIENT_SECRET]
        assert os.environ[ENV_CLIENT_SECRET] == "secret-saisi"
        assert os.environ[ENV_CLIENT_ID] == "déjà-là"
        assert [mode for mode, _ in asked] == ["masqué"]

    def test_secret_is_read_masked(self, monkeypatch):
        """Une clé secrète ne doit pas s'afficher dans le terminal."""
        monkeypatch.delenv(ENV_CLIENT_ID, raising=False)
        monkeypatch.delenv(ENV_CLIENT_SECRET, raising=False)
        modes: dict[str, str] = {}

        prompt_missing_env(
            FranceTravailSource,
            ask=lambda prompt: modes.setdefault(prompt.split()[0], "clair") or "id",
            ask_secret=lambda prompt: modes.setdefault(prompt.split()[0], "masqué")
            or "clé",
        )

        assert modes[ENV_CLIENT_ID] == "clair"
        assert modes[ENV_CLIENT_SECRET] == "masqué"

    def test_empty_answer_leaves_the_source_unconfigured(self, monkeypatch):
        """Renoncer à l'invite doit rester possible, sans variable vide posée."""
        monkeypatch.delenv(ENV_CLIENT_ID, raising=False)
        monkeypatch.delenv(ENV_CLIENT_SECRET, raising=False)

        filled = prompt_missing_env(
            FranceTravailSource, ask=lambda _: "  ", ask_secret=lambda _: ""
        )

        assert filled == []
        assert missing_env(FranceTravailSource) == [ENV_CLIENT_ID, ENV_CLIENT_SECRET]

    def test_values_are_never_written_to_disk(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv(ENV_CLIENT_ID, raising=False)
        monkeypatch.delenv(ENV_CLIENT_SECRET, raising=False)

        prompt_missing_env(
            FranceTravailSource, ask=lambda _: "identifiant", ask_secret=lambda _: "secret"
        )

        assert list(tmp_path.iterdir()) == []


class TestSearch:
    def test_returns_normalised_jobs(self):
        source, _ = fake_source()
        jobs = source.search(SearchQuery(text="machine learning", limit=10))
        assert len(jobs) == len(SEARCH["resultats"])
        job = jobs[0]
        assert job.source == "france_travail"
        assert job.title == "Ingénieur Machine Learning (H/F)"
        assert job.company == "ACME AI"
        assert job.location == "75 - Paris (Dept.)"
        assert job.language == "fr"
        assert str(job.url).endswith("/187MLKQ")
        assert job.source_ref["offer_id"] == "187MLKQ"

    def test_description_comes_with_the_search(self):
        """Contrairement à WTTJ, aucune requête de détail n'est nécessaire."""
        source, _ = fake_source()
        job = source.search(SearchQuery(text="ia", limit=1))[0]
        assert job.is_enriched
        assert "machine learning" in job.description

    def test_apply_url_prefers_the_real_form(self):
        source, _ = fake_source()
        jobs = source.search(SearchQuery(text="ia", limit=10))
        assert jobs[0].apply_url == "https://recrutement.acme-ai.example/offres/187MLKQ"
        # À défaut d'URL de postulation, l'URL d'origine du diffuseur.
        assert jobs[1].apply_url == "https://www.partenaire.example/offre/187MLKR"
        # À défaut des deux, la page France Travail, qui reste candidatable.
        assert jobs[2].apply_url.endswith("/187MLKS")

    def test_partner_becomes_the_ats(self):
        source, _ = fake_source()
        jobs = source.search(SearchQuery(text="ia", limit=10))
        assert jobs[1].ats == "APEC"
        assert jobs[0].ats == "france_travail"

    def test_hidden_company_gets_a_readable_placeholder(self):
        source, _ = fake_source()
        jobs = source.search(SearchQuery(text="ia", limit=10))
        assert jobs[1].company == "APEC"
        assert all(job.company for job in jobs)

    def test_limit_is_respected(self):
        source, _ = fake_source()
        assert len(source.search(SearchQuery(text="ia", limit=2))) == 2

    def test_no_result_is_not_an_error(self):
        source, _ = fake_source(Recorder(search_status=204))
        assert source.search(SearchQuery(text="licorne", limit=5)) == []

    def test_partial_content_is_accepted(self):
        """L'API répond 206 dès que la plage demandée dépasse le nombre d'offres."""
        source, _ = fake_source(Recorder(search_status=206))
        assert source.search(SearchQuery(text="ia", limit=10))

    def test_http_error_is_raised(self):
        source, _ = fake_source(Recorder(search_status=500))
        with pytest.raises(SourceError, match="HTTP 500"):
            source.search(SearchQuery(text="ia"))

    def test_stale_offers_are_dropped(self):
        source, _ = fake_source()
        jobs = source.search(SearchQuery(text="ia", limit=10, max_age_days=1))
        # Le jeu d'essai est daté d'août 2026 : tout est hors délai aujourd'hui.
        recent = [
            job
            for job in jobs
            if job.posted_at
            and job.posted_at > datetime.now(timezone.utc) - timedelta(days=1)
        ]
        assert jobs == recent

    def test_offer_without_id_is_ignored(self):
        source, _ = fake_source(
            Recorder(search={"resultats": [{"intitule": "sans identifiant"}]})
        )
        assert source.search(SearchQuery(text="ia", limit=5)) == []


class TestParams:
    def test_keywords_are_sanitised(self):
        source, recorder = fake_source()
        source.search(SearchQuery(text="ingénieur IA / LLM (H/F)", limit=1))
        assert recorder.params("/offres/search")["motsCles"] == "ingénieur IA LLM H F"

    def test_range_follows_the_limit(self):
        source, recorder = fake_source()
        source.search(SearchQuery(text="ia", limit=3))
        assert recorder.params("/offres/search")["range"] == "0-2"

    def test_published_since_never_filters_harder_than_asked(self):
        assert published_since(1) == 1
        assert published_since(10) == 14
        assert published_since(30) == 31
        assert published_since(90) == 31
        assert published_since(None) is None

    def test_published_since_is_sent(self):
        source, recorder = fake_source()
        source.search(SearchQuery(text="ia", limit=1, max_age_days=30))
        assert recorder.params("/offres/search")["publieeDepuis"] == "31"


class TestContractFilter:
    def test_codes_come_from_the_referentiel(self):
        source, recorder = fake_source()
        source.search(SearchQuery(text="ia", limit=1, contracts=["CDI", "CDD"]))
        assert recorder.params("/offres/search")["typeContrat"] == "CDD,CDI"

    def test_lookalike_labels_are_excluded(self):
        """« CDD insertion » contient « durée déterminée » sans être un CDD."""
        source, recorder = fake_source()
        source.search(SearchQuery(text="ia", limit=1, contracts=["CDD"]))
        assert recorder.params("/offres/search")["typeContrat"] == "CDD"

    def test_interim_does_not_pull_in_permanent_contracts(self):
        source, recorder = fake_source()
        source.search(SearchQuery(text="ia", limit=1, contracts=["interim"]))
        assert recorder.params("/offres/search")["typeContrat"] == "MIS"

    def test_apprenticeship_uses_the_nature_referentiel(self):
        source, recorder = fake_source()
        source.search(SearchQuery(text="ia", limit=1, contracts=["alternance"]))
        assert recorder.params("/offres/search")["natureContrat"] == "E2,FS"

    def test_internship_has_no_server_side_filter(self, caplog):
        """L'API n'expose aucune nature « stage » : on ne peut pas filtrer dessus."""
        source, recorder = fake_source()
        source.search(SearchQuery(text="ia", limit=1, contracts=["stage"]))
        params = recorder.params("/offres/search")
        assert "typeContrat" not in params
        assert "natureContrat" not in params

    def test_mixed_families_send_no_contract_filter(self, caplog):
        """`typeContrat` et `natureContrat` se combinent en ET : « CDI et stage »
        ne renvoie rien. Mieux vaut ne pas filtrer et laisser le filtre local."""
        source, recorder = fake_source()
        source.search(SearchQuery(text="ia", limit=1, contracts=["CDI", "stage"]))
        params = recorder.params("/offres/search")
        assert "typeContrat" not in params
        assert "natureContrat" not in params

    def test_untranslatable_contract_sends_no_filter(self, caplog):
        source, recorder = fake_source()
        source.search(SearchQuery(text="ia", limit=1, contracts=["CDI", "portage"]))
        assert "typeContrat" not in recorder.params("/offres/search")
        assert "portage" in caplog.text

    def test_referentiel_is_loaded_once(self):
        source, recorder = fake_source()
        source.search(SearchQuery(text="ia", limit=1, contracts=["CDI"]))
        source.search(SearchQuery(text="ml", limit=1, contracts=["CDI"]))
        assert sum("typesContrats" in path for path in recorder.paths) == 1

    def test_unavailable_referentiel_falls_back_to_stable_codes(self):
        source, recorder = fake_source(Recorder(referentiel_status=500))
        source.search(SearchQuery(text="ia", limit=1, contracts=["CDI"]))
        assert recorder.params("/offres/search")["typeContrat"] == "CDI"

    def test_unavailable_referentiel_without_fallback_drops_the_filter(self):
        """Aucun code figé pour les natures de contrat : on préfère ne pas filtrer.

        Inventer `E2` de mémoire ne ferait pas échouer la requête — elle la
        viderait, ce qui passerait pour « aucune offre » au lieu d'une panne.
        """
        source, recorder = fake_source(Recorder(referentiel_status=500))
        source.search(SearchQuery(text="ia", limit=1, contracts=["alternance"]))
        assert "natureContrat" not in recorder.params("/offres/search")


class TestContractLabel:
    """Le libellé doit parler le vocabulaire de `config.yaml`, sinon le
    pré-filtrage local rejette des offres qu'il est censé accepter."""

    def test_labels_match_the_configured_vocabulary(self):
        from agent_emploi.config import FiltersConfig, SearchConfig
        from agent_emploi.filters import screen_metadata

        source, _ = fake_source()
        jobs = source.search(SearchQuery(text="ia", limit=10))
        assert [job.contract for job in jobs[:3]] == ["CDI", "Stage", "Alternance"]

        search = SearchConfig(queries=["ia"], contracts=["CDI", "stage", "alternance"])
        filters = FiltersConfig(required_any=[], excluded_any=[], require_in_title=False)
        for job in jobs[:3]:
            screening = screen_metadata(job, filters, search, now=job.posted_at)
            assert screening.passed, f"{job.contract}: {screening.reason}"

    def test_internship_is_read_from_the_title(self):
        """Les champs de contrat mentent sur les stages, l'intitulé non.

        Le jeu d'essai reprend une offre réelle : « Stage - Data Scientist NLP »
        publiée en `CDI` / `Contrat travail`. Se fier au champ classerait ce
        stage en CDI.
        """
        source, _ = fake_source()
        stage = source.search(SearchQuery(text="ia", limit=10))[1]
        assert stage.raw["typeContrat"] == "CDI"
        assert stage.contract == "Stage"

    def test_alternance_wins_over_a_title_saying_stage(self):
        """« Stage de césure » en contrat d'apprentissage est une alternance."""
        source, _ = fake_source(
            Recorder(
                search={
                    "resultats": [
                        {
                            "id": "X2",
                            "intitule": "Stage de césure Développement (H/F)",
                            "typeContrat": "CDD",
                            "natureContrat": "Contrat apprentissage",
                            "alternance": True,
                        }
                    ]
                }
            )
        )
        assert source.search(SearchQuery(text="ia", limit=1))[0].contract == "Alternance"

    def test_unknown_code_keeps_the_source_label(self):
        source, _ = fake_source(
            Recorder(
                search={
                    "resultats": [
                        {
                            "id": "X1",
                            "intitule": "Offre",
                            "typeContrat": "ZZZ",
                            "typeContratLibelle": "Contrat exotique",
                        }
                    ]
                }
            )
        )
        assert source.search(SearchQuery(text="ia", limit=1))[0].contract == "Contrat exotique"


class TestEnrich:
    def test_enriched_job_costs_no_request(self):
        source, recorder = fake_source()
        job = source.search(SearchQuery(text="ia", limit=1))[0]
        before = len(recorder.requests)
        assert source.enrich(job) == job
        assert len(recorder.requests) == before

    def test_truncated_job_is_completed_by_the_detail(self):
        source, recorder = fake_source()
        job = source.search(SearchQuery(text="ia", limit=1))[0]
        stripped = job.model_copy(update={"description": ""})

        enriched = source.enrich(stripped)
        assert enriched.is_enriched
        assert enriched.id == job.id
        assert any(path.endswith("/offres/187MLKQ") for path in recorder.paths)

    def test_job_without_identifier_is_returned_as_is(self):
        source, _ = fake_source()
        job = source.search(SearchQuery(text="ia", limit=1))[0]
        stripped = job.model_copy(update={"description": "", "source_ref": {}})
        assert source.enrich(stripped) == stripped

    def test_detail_failure_returns_job_unchanged(self):
        source, _ = fake_source(Recorder(token_status=400))
        job = FranceTravailSource._to_job(
            fake_source()[0], SEARCH["resultats"][0]
        ).model_copy(update={"description": ""})
        assert source.enrich(job) == job


@pytest.mark.live
class TestLiveContract:
    """Interroge la vraie API. Sans identifiants, le test est ignoré.

    En cas d'échec, ne pas contourner : relever les valeurs actuelles dans les
    référentiels et mettre à jour `france_travail.py`.
    """

    @pytest.fixture(autouse=True)
    def credentials(self):
        if not (os.environ.get(ENV_CLIENT_ID) and os.environ.get(ENV_CLIENT_SECRET)):
            pytest.skip(f"{ENV_CLIENT_ID}/{ENV_CLIENT_SECRET} absentes")

    def test_search_returns_offers(self):
        with FranceTravailSource() as source:
            jobs = source.search(
                SearchQuery(text="machine learning", limit=5, max_age_days=31)
            )
            assert jobs, "l'API ne renvoie plus d'offres"
            assert all(job.is_enriched for job in jobs), "plus de description en recherche"
            assert len({job.id for job in jobs}) == len(jobs)

    def test_contract_referentials_still_carry_the_expected_labels(self):
        with FranceTravailSource() as source:
            assert source._resolve_contract("cdi") == ("typesContrats", ["CDI"])
            assert source._resolve_contract("cdd") == ("typesContrats", ["CDD"])

            referentiel, codes = source._resolve_contract("alternance")
            assert referentiel == "naturesContrats"
            assert sorted(codes) == ["E2", "FS"], (
                "les natures apprentissage / professionnalisation ont changé de "
                "code ou de libellé"
            )

    def test_no_internship_nature_exists(self):
        """Le jour où France Travail en ajoutera une, ce test le dira.

        Toute la stratégie « stage » en dépend : pas de filtre serveur, et un
        libellé déduit de l'intitulé. Si une nature « stage » apparaît, elle
        devient plus fiable que l'heuristique et il faut revenir ici.
        """
        with FranceTravailSource() as source:
            natures = source._referentiel("naturesContrats")
            assert natures, "référentiel des natures de contrat vide ou inaccessible"
            assert not [n for n in natures if "stage" in n["libelle"].lower()]
