"""Source Welcome to the Jungle.

Deux services publics sont utilisés, aucun ne demandant de compte :

* **Index Algolia** `wk_cms_jobs_production` — la recherche du site. Il contient
  les métadonnées (titre, entreprise, lieu, contrat, secteurs) mais **pas** la
  description de l'offre.
* **API v3** `/api/v3/organizations/{org}/jobs/{slug}` — le détail d'une offre :
  description complète, et surtout `apply_url` / `ats`, car une majorité des
  offres redirigent vers un ATS externe (Greenhouse, Lever, Workday…).

La page publique d'une offre, elle, est protégée par un pare-feu applicatif qui
renvoie un défi JavaScript : elle est inutilisable en HTTP simple, et l'API v3
la remplace avantageusement.

Aucun de ces points d'entrée n'est contractuel. Les identifiants Algolia sont
donc lus à l'exécution depuis `/api/env` (le site les y publie pour son propre
front) et ne sont figés dans le code qu'en dernier recours. Le test marqué
`live` échoue bruyamment si le contrat change.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from html import unescape
from typing import Any

import httpx

from agent_emploi.models import Job
from agent_emploi.sources.base import SearchQuery, SourceError

logger = logging.getLogger(__name__)

SITE = "https://www.welcometothejungle.com"
API = "https://api.welcometothejungle.com"
ENV_URL = f"{SITE}/api/env"
JOBS_INDEX = "wk_cms_jobs_production"

#: Identifiants publics de recherche, relevés le 2026-08-09 sur `/api/env`, où
#: le site les publie pour son propre front. Ils peuvent être renouvelés ; c'est
#: le test marqué `live` qui le détectera, en échouant sur un 403.
FALLBACK_APP_ID = "CSEKHVMS53"
FALLBACK_API_KEY = "4bd8f6215d0cc52b26430765769e65a0"

_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

#: Algolia restreint sa clé publique par référent : sans cet en-tête, tout est 403.
_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Referer": f"{SITE}/",
    "Origin": SITE,
    "Accept": "application/json",
}

#: Le pare-feu du site sert un défi JavaScript (HTTP 202) aux requêtes qui ne
#: ressemblent pas à une navigation. `/api/env` doit donc être demandé avec un
#: `Accept` de navigateur, pas `application/json`.
_ENV_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
}

TIMEOUT = httpx.Timeout(30.0, connect=10.0)

#: Une même offre est indexée une fois par job board partenaire diffusant
#: l'annonce (`website.reference`), avec un `objectID` différent mais les mêmes
#: slugs. Sans sur-échantillonnage, une seule offre peut occuper toute la page
#: de résultats. On demande donc plus de résultats que nécessaire, puis on
#: dédoublonne sur (org, offre).
_OVERSAMPLE = 5
_MAX_HITS_PER_PAGE = 100
_MAX_PAGES = 10

#: Correspondance entre les contrats nommés dans `config.yaml` et les valeurs
#: de la facette `contract_type`.
CONTRACT_FACETS = {
    "cdi": "FULL_TIME",
    "full_time": "FULL_TIME",
    "cdd": "TEMPORARY",
    "temporaire": "TEMPORARY",
    "temporary": "TEMPORARY",
    "stage": "INTERNSHIP",
    "internship": "INTERNSHIP",
    "alternance": "APPRENTICESHIP",
    "apprenticeship": "APPRENTICESHIP",
    "freelance": "FREELANCE",
}

_TAG = re.compile(r"<[^>]+>")
_SPACES = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def html_to_text(html: str) -> str:
    """Convertit la description HTML de l'offre en texte lisible.

    Les descriptions WTTJ sont du HTML simple (titres, paragraphes, listes).
    On insère un saut de ligne aux balises de bloc pour que la structure
    survive, sinon les phrases se collent et le score lexical se dégrade.
    """
    if not html:
        return ""
    text = re.sub(r"(?i)<br\s*/?>", "\n", html)
    text = re.sub(r"(?i)</(p|div|li|h[1-6]|tr|ul|ol|blockquote)>", "\n", text)
    text = re.sub(r"(?i)<li[^>]*>", "- ", text)
    text = _TAG.sub("", text)
    text = unescape(text)
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _BLANK_LINES.sub("\n\n", text).strip()


def _first(mapping: Any, *keys: str) -> Any:
    """Première valeur non vide parmi `keys` dans un dictionnaire éventuel."""
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        value = mapping.get(key)
        if value:
            return value
    return None


class WttjSource:
    """Recherche et détail des offres Welcome to the Jungle."""

    name = "wttj"

    #: Aucun compte, aucune clé : les deux services utilisés sont publics.
    required_env: tuple[str, ...] = ()

    def __init__(
        self,
        client: httpx.Client | None = None,
        *,
        app_id: str | None = None,
        api_key: str | None = None,
        request_delay: float = 0.3,
    ) -> None:
        self._client = client or httpx.Client(timeout=TIMEOUT, headers=_HEADERS)
        self._app_id = app_id
        self._api_key = api_key
        #: Pause entre requêtes de détail — on interroge un service gratuit,
        #: autant ne pas le marteler.
        self._request_delay = request_delay

    # --------------------------------------------------------- identifiants

    def _credentials(self) -> tuple[str, str]:
        """Identifiants Algolia : lecture dynamique au mieux, sinon constantes.

        Le site publie ses clés publiques sur `/api/env` pour son propre front ;
        les relire à l'exécution permettrait de survivre à un renouvellement.
        Mais le domaine du site est protégé par un pare-feu applicatif qui sert
        un défi JavaScript (HTTP 202) aux clients qu'il ne reconnaît pas — ce que
        nous sommes. L'échec est donc **attendu**, pas anormal : on le journalise
        en debug et on retombe sur les constantes.

        Ce repli n'est pas fragile en pratique : l'API Algolia, elle, n'est pas
        protégée, et un renouvellement des clés se traduirait par un 403 franc
        que le test `live` détecte immédiatement.
        """
        if self._app_id and self._api_key:
            return self._app_id, self._api_key

        app_id, api_key = FALLBACK_APP_ID, FALLBACK_API_KEY
        try:
            response = self._client.get(ENV_URL, headers=_ENV_HEADERS)
            response.raise_for_status()
            # Réponse attendue : le fragment JS `window.env = {...};`
            payload = re.search(r"\{.*\}", response.text, re.DOTALL)
            if payload is None:
                raise ValueError("réponse sans objet JSON (défi anti-bot ?)")
            env = json.loads(payload.group(0))
            app_id = env["PUBLIC_ALGOLIA_APPLICATION_ID"]
            api_key = env["PUBLIC_ALGOLIA_API_KEY_CLIENT"]
            logger.debug("wttj: identifiants Algolia lus depuis %s", ENV_URL)
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            logger.debug(
                "wttj: %s illisible (%s), utilisation des identifiants figés",
                ENV_URL,
                exc,
            )

        self._app_id, self._api_key = app_id, api_key
        return app_id, api_key

    # -------------------------------------------------------------- recherche

    def _facet_filters(self, query: SearchQuery) -> list[list[str]]:
        """Traduit les critères en facettes Algolia.

        Une liste imbriquée signifie OU entre ses éléments ; deux listes
        distinctes signifient ET.
        """
        filters: list[list[str]] = []

        if query.countries:
            filters.append([f"offices.country:{c}" for c in query.countries])

        if query.contracts:
            facets = {
                CONTRACT_FACETS[c.lower()]
                for c in query.contracts
                if c.lower() in CONTRACT_FACETS
            }
            unknown = [c for c in query.contracts if c.lower() not in CONTRACT_FACETS]
            if unknown:
                logger.warning("wttj: contrats ignorés (inconnus): %s", unknown)
            if facets:
                filters.append([f"contract_type:{f}" for f in sorted(facets)])

        if query.remote:
            filters.append([f"remote:{r}" for r in query.remote])

        return filters

    def _query_page(self, query: SearchQuery, page: int) -> dict:
        """Une page de résultats Algolia."""
        app_id, api_key = self._credentials()
        payload: dict[str, Any] = {
            "query": query.text,
            "hitsPerPage": min(query.limit * _OVERSAMPLE, _MAX_HITS_PER_PAGE),
            "page": page,
            "attributesToHighlight": [],
        }
        facet_filters = self._facet_filters(query)
        if facet_filters:
            payload["facetFilters"] = facet_filters

        try:
            response = self._client.post(
                f"https://{app_id}-dsn.algolia.net/1/indexes/{JOBS_INDEX}/query",
                headers={
                    "X-Algolia-API-Key": api_key,
                    "X-Algolia-Application-Id": app_id,
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            raise SourceError(
                f"wttj: recherche refusée (HTTP {exc.response.status_code}) — "
                f"{exc.response.text[:200]}"
            ) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise SourceError(f"wttj: recherche impossible — {exc}") from exc

    def search(self, query: SearchQuery) -> list[Job]:
        """Interroge l'index Algolia et renvoie des offres distinctes, sans description.

        Parcourt autant de pages que nécessaire pour atteindre `query.limit`
        offres *distinctes* : la duplication par job board partenaire fait qu'une
        page de résultats bruts peut ne contenir qu'une poignée d'offres réelles.
        """
        cutoff = None
        if query.max_age_days is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(days=query.max_age_days)

        jobs: list[Job] = []
        seen_ids: set[str] = set()

        for page in range(_MAX_PAGES):
            data = self._query_page(query, page)
            hits = data.get("hits")
            if hits is None:
                raise SourceError(f"wttj: réponse sans 'hits' — {str(data)[:200]}")

            for hit in hits:
                job = self._to_job(hit)
                if job is None or job.id in seen_ids:
                    continue
                if cutoff and job.posted_at and job.posted_at < cutoff:
                    continue
                seen_ids.add(job.id)
                jobs.append(job)
                if len(jobs) >= query.limit:
                    return jobs

            if page + 1 >= data.get("nbPages", 0):
                break

        return jobs

    def _to_job(self, hit: dict) -> Job | None:
        """Normalise un résultat Algolia. Retourne None si inexploitable."""
        org = hit.get("organization") or {}
        org_slug = org.get("slug")
        job_slug = hit.get("slug")
        if not org_slug or not job_slug:
            # Sans les deux slugs on ne peut ni construire l'URL ni enrichir.
            logger.debug("wttj: résultat sans slug, ignoré (%s)", hit.get("objectID"))
            return None

        office = hit.get("office") or {}
        location = ", ".join(
            part for part in (office.get("city"), office.get("country")) if part
        )

        return Job.build(
            source=self.name,
            url=f"{SITE}/fr/companies/{org_slug}/jobs/{job_slug}",
            title=hit.get("name") or "(sans titre)",
            company=org.get("name") or org_slug,
            location=location or None,
            contract=_first(hit.get("contract_type_names"), "fr", "en")
            or hit.get("contract_type"),
            remote=hit.get("remote"),
            salary=self._salary(hit),
            posted_at=self._parse_date(hit.get("published_at")),
            language=hit.get("language"),
            sectors=self._sectors(hit),
            source_ref={"org_slug": org_slug, "job_slug": job_slug},
            raw=hit,
        )

    @staticmethod
    def _salary(hit: dict) -> str | None:
        low, high = hit.get("salary_minimum"), hit.get("salary_maximum")
        if not low and not high:
            return None
        currency = hit.get("salary_currency") or ""
        period = hit.get("salary_period") or ""
        span = f"{low}–{high}" if low and high else str(low or high)
        return " ".join(part for part in (span, currency, period) if part).strip()

    @staticmethod
    def _sectors(hit: dict) -> list[str]:
        names: list[str] = []
        for sector in hit.get("sectors") or []:
            name = _first(sector.get("name") if isinstance(sector, dict) else None, "fr", "en")
            if name:
                names.append(name)
        return names

    @staticmethod
    def _parse_date(value: Any) -> datetime | None:
        if not isinstance(value, str) or not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            logger.debug("wttj: date illisible: %r", value)
            return None
        # Une date sans fuseau est traitée comme UTC pour rester comparable.
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    # ---------------------------------------------------------------- détail

    def enrich(self, job: Job) -> Job:
        """Récupère la description complète et l'URL réelle de candidature.

        En cas d'échec, l'offre est renvoyée inchangée plutôt que de faire
        échouer la passe : une description manquante fera simplement échouer le
        pré-filtrage suivant, sans perdre les autres offres.
        """
        org_slug = job.source_ref.get("org_slug")
        job_slug = job.source_ref.get("job_slug")
        if not org_slug or not job_slug:
            logger.warning("wttj: offre %s sans slugs, enrichissement impossible", job.id)
            return job

        if self._request_delay:
            time.sleep(self._request_delay)

        try:
            response = self._client.get(
                f"{API}/api/v3/organizations/{org_slug}/jobs/{job_slug}"
            )
            response.raise_for_status()
            detail = (response.json() or {}).get("job") or {}
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("wttj: détail indisponible pour %s — %s", job.url, exc)
            return job

        description = html_to_text(detail.get("description") or "")
        summary = (detail.get("summary") or "").strip()
        if summary and summary not in description:
            description = f"{summary}\n\n{description}".strip()

        return job.model_copy(
            update={
                "description": description,
                "apply_url": detail.get("apply_url"),
                "ats": detail.get("ats"),
                "language": detail.get("language") or job.language,
                "salary": job.salary or self._detail_salary(detail),
            }
        )

    @staticmethod
    def _detail_salary(detail: dict) -> str | None:
        low, high = detail.get("salary_min"), detail.get("salary_max")
        if not low and not high:
            return None
        currency = detail.get("salary_currency") or ""
        period = detail.get("salary_period") or ""
        span = f"{low}–{high}" if low and high else str(low or high)
        return " ".join(part for part in (span, currency, period) if part).strip()

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> WttjSource:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
