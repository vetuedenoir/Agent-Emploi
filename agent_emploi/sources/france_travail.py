"""Source France Travail — l'API officielle « Offres d'emploi v2 ».

Contrairement à Welcome to the Jungle, tout est ici contractuel et documenté :
un compte développeur sur `francetravail.io`, une application déclarée, deux
identifiants, et un jeton OAuth2 `client_credentials` valable une vingtaine de
minutes. C'est le filet de sécurité du projet : si l'index Algolia de WTTJ ferme
un jour, cette source-ci continue de fonctionner.

Trois différences avec WTTJ, qui expliquent la conception :

1. **La recherche renvoie déjà la description.** Le découpage en deux temps
   (`search` puis `enrich`) n'a donc rien à économiser : `enrich()` ne fait
   aucune requête quand l'offre est déjà complète. L'interface reste la même,
   c'est le reste du système qui n'a pas à savoir.
2. **Les codes de contrat sont des référentiels**, pas des chaînes libres. Ils
   sont résolus à l'exécution depuis `/referentiel/...` en appariant les
   libellés, plutôt que devinés : un code inventé filtrerait en silence.
3. **`typeContrat` et `natureContrat` se combinent en ET.** Demander « CDI ou
   stage » dans une seule requête ne renvoie donc rien. Quand les contrats
   demandés relèvent des deux familles, aucun filtre n'est envoyé au serveur et
   c'est le pré-filtrage local qui tranche : plus d'offres transitent, mais
   aucune n'est perdue en silence.

Points d'entrée : jeton sur `entreprise.francetravail.fr`, données sur
`api.francetravail.io`. Le test marqué `live` vérifie les deux.
"""

from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, NamedTuple

import httpx

from agent_emploi.filters import contract_from_title, fold
from agent_emploi.models import Job
from agent_emploi.sources.base import SearchQuery, SourceError

logger = logging.getLogger(__name__)

AUTH_URL = "https://entreprise.francetravail.fr/connexion/oauth2/access_token"
AUTH_REALM = "/partenaire"
API = "https://api.francetravail.io/partenaire/offresdemploi/v2"
SITE = "https://candidat.francetravail.fr/offres/recherche/detail"

#: Portée demandée au jeton. `api_offresdemploiv2` ouvre l'API, `o2dsoffre`
#: donne accès aux offres elles-mêmes ; les deux sont nécessaires.
SCOPE = "api_offresdemploiv2 o2dsoffre"

#: Identifiants de l'application déclarée sur francetravail.io. Ils ne sont pas
#: dans `config.yaml` : ce sont des secrets, ils vivent dans `.env`.
ENV_CLIENT_ID = "FRANCE_TRAVAIL_CLIENT_ID"
ENV_CLIENT_SECRET = "FRANCE_TRAVAIL_CLIENT_SECRET"

_USER_AGENT = "agent-emploi/0.1 (recherche d'emploi personnelle)"

TIMEOUT = httpx.Timeout(30.0, connect=10.0)

#: Nombre maximal d'offres par requête, imposé par le paramètre `range`.
_PAGE_SIZE = 150
#: Au-delà, l'API refuse la plage. Une passe de recherche n'en approche jamais.
_MAX_OFFSET = 3000
#: Marge de renouvellement du jeton : on ne part pas en requête avec un jeton
#: qui expire dans la seconde.
_TOKEN_MARGIN = timedelta(seconds=60)

#: Valeurs acceptées par `publieeDepuis`. On choisit toujours la plus large
#: au-dessus de l'ancienneté demandée : filtrer plus strictement que demandé
#: côté serveur perdrait des offres que le filtre local aurait gardées.
_PUBLISHED_SINCE = (1, 3, 7, 14, 31)

#: Caractères acceptés par `motsCles`. Le reste est remplacé par une espace :
#: une apostrophe typographique ou un slash suffit à faire échouer la requête.
_MOTS_CLES = re.compile(r"[^0-9A-Za-zÀ-ÿ' -]+")


#: Traduction des contrats nommés dans `config.yaml` vers les référentiels de
#: France Travail. Pour chaque terme : le référentiel concerné, les libellés à
#: apparier (comparaison repliée, par sous-chaîne) et un code de repli quand
#: celui-ci est stable et documenté.
#:
#: Les libellés priment sur les codes : le référentiel est interrogé à
#: l'exécution, ce qui évite d'inventer un code — un code inconnu ne fait pas
#: échouer la requête, il la vide.
class ContractTerm(NamedTuple):
    """Comment traduire un contrat de `config.yaml` en codes de référentiel."""

    referentiel: str
    #: Libellés qui désignent ce contrat, cherchés par sous-chaîne.
    needles: tuple[str, ...]
    #: Libellés à écarter malgré une correspondance. Les référentiels contiennent
    #: des variantes dont le libellé contient celui du contrat courant sans être
    #: le même contrat : « Contrat durée déterminée **insertion** » n'est pas ce
    #: que demande quelqu'un qui écrit « CDD ».
    excludes: tuple[str, ...] = ()
    #: Code employé si le référentiel est injoignable. Seulement là où il est
    #: stable et documenté ; ailleurs, mieux vaut renoncer au filtre.
    fallback: str | None = None


CONTRACT_TERMS: dict[str, ContractTerm] = {
    "cdi": ContractTerm("typesContrats", ("duree indeterminee",), fallback="CDI"),
    "cdd": ContractTerm(
        "typesContrats", ("duree determinee",), ("insertion",), fallback="CDD"
    ),
    # « intérim » ne doit pas ramener le CDI intérimaire, qui est un CDI.
    "interim": ContractTerm(
        "typesContrats", ("mission interimaire",), ("cdi",), fallback="MIS"
    ),
    "freelance": ContractTerm("typesContrats", ("profession liberale",), fallback="LIB"),
    "liberal": ContractTerm("typesContrats", ("profession liberale",), fallback="LIB"),
    "saisonnier": ContractTerm("typesContrats", ("saisonnier",), fallback="SAI"),
    "alternance": ContractTerm(
        "naturesContrats", ("apprentissage", "professionnalisation")
    ),
    "apprentissage": ContractTerm("naturesContrats", ("apprentissage",)),
}

#: Contrats que l'API ne sait pas filtrer, quoi qu'on lui demande.
#:
#: Le référentiel des natures de contrat **ne contient aucun « stage »** — c'est
#: vérifié par un test `live`. Et les offres relayent l'anomalie : « Stage :
#: Ingénieur en informatique » est publiée avec `typeContrat: CDI`,
#: `natureContrat: Contrat travail`. Sur cette source, le stage ne se lit pas
#: dans les champs de contrat, seulement dans l'intitulé — d'où le repli de
#: `_contract_label`, et l'absence de filtre serveur ici.
UNFILTERABLE = frozenset({"stage"})


#: Nom du paramètre de recherche associé à chaque référentiel.
_FILTER_PARAM = {"typesContrats": "typeContrat", "naturesContrats": "natureContrat"}

#: Libellé court d'un contrat, pour que le pré-filtrage local compare la même
#: chose que ce qui est écrit dans `config.yaml` (« CDI », « stage »). Le
#: libellé complet du référentiel — « Contrat à durée indéterminée » — ne
#: contient pas la chaîne « CDI ».
_CONTRACT_LABELS = {
    "CDI": "CDI",
    "CDD": "CDD",
    "MIS": "Intérim",
    "SAI": "Saisonnier",
    "LIB": "Freelance",
    "DDI": "CDD",
    "DIN": "CDI",
}


def _first(mapping: Any, *keys: str) -> Any:
    """Première valeur non vide parmi `keys` dans un dictionnaire éventuel."""
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        value = mapping.get(key)
        if value:
            return value
    return None


def published_since(max_age_days: int | None) -> int | None:
    """Valeur de `publieeDepuis` couvrant au moins l'ancienneté demandée."""
    if max_age_days is None:
        return None
    for value in _PUBLISHED_SINCE:
        if value >= max_age_days:
            return value
    return _PUBLISHED_SINCE[-1]


class FranceTravailSource:
    """Recherche d'offres sur l'API officielle France Travail."""

    name = "france_travail"

    #: Variables d'environnement sans lesquelles la source ne peut rien faire.
    #: `doctor` et la passe de recherche s'en servent pour l'annoncer une fois,
    #: proprement, plutôt que d'échouer à chaque requête.
    required_env = (ENV_CLIENT_ID, ENV_CLIENT_SECRET)

    def __init__(
        self,
        client: httpx.Client | None = None,
        *,
        client_id: str | None = None,
        client_secret: str | None = None,
        request_delay: float = 0.2,
    ) -> None:
        self._client = client or httpx.Client(
            timeout=TIMEOUT, headers={"User-Agent": _USER_AGENT}
        )
        self._client_id = client_id
        self._client_secret = client_secret
        self._request_delay = request_delay
        self._token: str | None = None
        self._token_expiry: datetime | None = None
        self._referentiels: dict[str, list[dict]] = {}

    # ---------------------------------------------------------------- jeton

    def _credentials(self) -> tuple[str, str]:
        """Identifiants de l'application, lus dans l'environnement.

        Ils ne sont jamais journalisés ni recopiés ailleurs : seule leur absence
        est signalée, avec le nom exact de la variable à définir.
        """
        client_id = self._client_id or os.environ.get(ENV_CLIENT_ID)
        client_secret = self._client_secret or os.environ.get(ENV_CLIENT_SECRET)
        if not client_id or not client_secret:
            raise SourceError(
                "france_travail: identifiants absents — définir "
                f"{ENV_CLIENT_ID} et {ENV_CLIENT_SECRET} (voir .env.example ; "
                "compte gratuit sur https://francetravail.io)"
            )
        return client_id, client_secret

    def _access_token(self, *, force: bool = False) -> str:
        """Jeton OAuth2, renouvelé seulement quand il approche de l'expiration.

        Un jeton dure une vingtaine de minutes et couvre donc toute une passe :
        le renouveler à chaque requête ferait un aller-retour de plus par offre.
        """
        now = datetime.now(timezone.utc)
        if (
            not force
            and self._token
            and self._token_expiry
            and now + _TOKEN_MARGIN < self._token_expiry
        ):
            return self._token

        client_id, client_secret = self._credentials()
        try:
            response = self._client.post(
                AUTH_URL,
                params={"realm": AUTH_REALM},
                data={
                    "grant_type": "client_credentials",
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "scope": SCOPE,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            # Le corps d'une erreur d'authentification ne contient pas le secret
            # envoyé, seulement un code (`invalid_client`) : il est sûr à citer.
            raise SourceError(
                f"france_travail: authentification refusée "
                f"(HTTP {exc.response.status_code}) — {exc.response.text[:200]}"
            ) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise SourceError(f"france_travail: jeton impossible à obtenir — {exc}") from exc

        token = payload.get("access_token")
        if not token:
            raise SourceError("france_travail: réponse d'authentification sans jeton")

        self._token = token
        self._token_expiry = now + timedelta(seconds=int(payload.get("expires_in", 0)))
        return token

    def _get(self, path: str, params: dict | None = None) -> httpx.Response:
        """Requête authentifiée, avec une seule reprise si le jeton est périmé."""
        for attempt in (1, 2):
            token = self._access_token(force=attempt == 2)
            try:
                response = self._client.get(
                    f"{API}{path}",
                    params=params,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Accept": "application/json",
                    },
                )
            except httpx.HTTPError as exc:
                raise SourceError(f"france_travail: {path} injoignable — {exc}") from exc

            if response.status_code == 401 and attempt == 1:
                # Jeton révoqué avant son expiration annoncée : une reprise, et
                # une seule, pour ne pas boucler sur des identifiants invalides.
                logger.debug("france_travail: jeton refusé, renouvellement")
                continue
            if response.status_code >= 400:
                raise SourceError(
                    f"france_travail: {path} refusé (HTTP {response.status_code}) — "
                    f"{response.text[:200]}"
                )
            return response

        raise SourceError("france_travail: authentification impossible")  # pragma: no cover

    # --------------------------------------------------------- référentiels

    def _referentiel(self, name: str) -> list[dict]:
        """Référentiel `{code, libelle}`, chargé une fois par passe.

        En cas d'échec, une liste vide : l'appelant retombe alors sur les codes
        de repli, et à défaut renonce au filtre serveur.
        """
        if name in self._referentiels:
            return self._referentiels[name]
        try:
            data = self._get(f"/referentiel/{name}").json()
        except (SourceError, ValueError) as exc:
            logger.debug("france_travail: référentiel %s illisible (%s)", name, exc)
            data = []
        entries = data if isinstance(data, list) else []
        self._referentiels[name] = entries
        return entries

    def _resolve_contract(self, term: str) -> tuple[str, list[str]] | None:
        """Traduit un contrat de `config.yaml` en `(référentiel, codes)`.

        Un terme peut couvrir plusieurs codes — « alternance » vaut apprentissage
        et professionnalisation. Le libellé fait foi : c'est lui qui est apparié
        dans le référentiel courant. Le code figé n'intervient que si le
        référentiel est indisponible, et seulement là où il est documenté.
        """
        entry = CONTRACT_TERMS.get(fold(term).strip())
        if entry is None:
            return None
        referentiel, needles, excludes, fallback = entry

        codes: list[str] = []
        for item in self._referentiel(referentiel):
            libelle = fold(str(item.get("libelle") or ""))
            code = str(item.get("code") or "")
            if not code or not any(needle in libelle for needle in needles):
                continue
            if any(exclude in libelle for exclude in excludes):
                continue
            codes.append(code)
        if codes:
            return referentiel, codes

        if fallback:
            logger.debug(
                "france_travail: référentiel %s indisponible, code figé %s pour %r",
                referentiel,
                fallback,
                term,
            )
            return referentiel, [fallback]
        return None

    def _contract_params(self, query: SearchQuery) -> dict[str, str]:
        """Filtre de contrat côté serveur, ou rien si l'on ne peut pas le garantir.

        Deux cas font renoncer au filtre, et dans les deux le pré-filtrage local
        prend le relais — on ramène plus d'offres, on n'en perd aucune :

        * un contrat demandé qu'on ne sait pas traduire ;
        * des contrats relevant des deux référentiels à la fois, car l'API les
          combine en ET (« CDI **et** stage » ne renvoie rien).
        """
        if not query.contracts:
            return {}

        codes: dict[str, list[str]] = {}
        for term in query.contracts:
            if fold(term).strip() in UNFILTERABLE:
                logger.info(
                    "france_travail: %r n'existe pas dans les référentiels de "
                    "l'API — filtre serveur abandonné, le pré-filtrage local "
                    "s'en charge",
                    term,
                )
                return {}
            resolved = self._resolve_contract(term)
            if resolved is None:
                logger.warning(
                    "france_travail: contrat %r non traduisible — filtre serveur "
                    "abandonné, le pré-filtrage local s'en charge",
                    term,
                )
                return {}
            referentiel, resolved_codes = resolved
            codes.setdefault(referentiel, []).extend(resolved_codes)

        if len(codes) > 1:
            logger.info(
                "france_travail: contrats à cheval sur deux référentiels (%s) — "
                "filtre serveur abandonné, le pré-filtrage local s'en charge",
                ", ".join(sorted(codes)),
            )
            return {}

        referentiel, values = next(iter(codes.items()))
        return {_FILTER_PARAM[referentiel]: ",".join(sorted(set(values)))}

    # ------------------------------------------------------------- recherche

    def _params(self, query: SearchQuery, offset: int, count: int) -> dict[str, str]:
        params: dict[str, str] = {"range": f"{offset}-{offset + count - 1}"}

        keywords = _MOTS_CLES.sub(" ", query.text).strip()
        keywords = " ".join(keywords.split())
        if keywords:
            params["motsCles"] = keywords

        since = published_since(query.max_age_days)
        if since is not None:
            params["publieeDepuis"] = str(since)

        params.update(self._contract_params(query))

        # `query.countries` n'a pas d'équivalent : l'API ne publie que des
        # offres en France. Un pays demandé autre que la France signalerait une
        # attente que cette source ne peut pas tenir.
        foreign = [c for c in query.countries if fold(c).strip() not in ("", "france")]
        if foreign:
            logger.debug("france_travail: pays ignorés (source française): %s", foreign)

        return params

    def search(self, query: SearchQuery) -> list[Job]:
        """Interroge l'API et renvoie des offres normalisées, description comprise."""
        cutoff = None
        if query.max_age_days is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(days=query.max_age_days)

        jobs: list[Job] = []
        seen_ids: set[str] = set()
        offset = 0

        while len(jobs) < query.limit and offset < _MAX_OFFSET:
            count = min(_PAGE_SIZE, query.limit - len(jobs))
            response = self._get("/offres/search", self._params(query, offset, count))

            # 204 : aucune offre ne correspond. Ce n'est pas une erreur.
            if response.status_code == 204 or not response.content:
                break
            try:
                results = (response.json() or {}).get("resultats") or []
            except ValueError as exc:
                raise SourceError(f"france_travail: réponse illisible — {exc}") from exc
            if not results:
                break

            for offer in results:
                job = self._to_job(offer)
                if job is None or job.id in seen_ids:
                    continue
                if cutoff and job.posted_at and job.posted_at < cutoff:
                    continue
                seen_ids.add(job.id)
                jobs.append(job)
                if len(jobs) >= query.limit:
                    return jobs

            if len(results) < count:
                break
            offset += count

        return jobs

    def _to_job(self, offer: dict) -> Job | None:
        """Normalise une offre. Retourne None si elle est inexploitable."""
        offer_id = offer.get("id")
        title = offer.get("intitule")
        if not offer_id or not title:
            logger.debug("france_travail: offre sans id ou intitulé, ignorée")
            return None

        origin = offer.get("origineOffre") or {}
        partners = origin.get("partenaires") or []
        partner = partners[0].get("nom") if partners and isinstance(partners[0], dict) else None

        return Job.build(
            source=self.name,
            url=f"{SITE}/{offer_id}",
            title=title,
            company=self._company(offer, partner),
            location=_first(offer.get("lieuTravail"), "libelle"),
            contract=self._contract_label(offer),
            salary=_first(offer.get("salaire"), "libelle"),
            description=(offer.get("description") or "").strip(),
            posted_at=self._parse_date(offer.get("dateCreation")),
            # L'API ne publie que des annonces en français ; l'indice est sûr,
            # contrairement à WTTJ où il vient d'un champ déclaratif.
            language="fr",
            sectors=[s for s in (offer.get("secteurActiviteLibelle"),) if s],
            apply_url=_first(offer.get("contact"), "urlPostulation")
            or origin.get("urlOrigine")
            or f"{SITE}/{offer_id}",
            # L'ATS est le site du partenaire diffuseur quand il y
            # en a un ; sinon la candidature se fait sur France Travail.
            ats=partner or self.name,
            source_ref={"offer_id": str(offer_id)},
            raw=offer,
        )

    @staticmethod
    def _company(offer: dict, partner: str | None) -> str:
        """Nom de l'entreprise, souvent masqué sur les offres France Travail."""
        return (
            _first(offer.get("entreprise"), "nom")
            or partner
            or "Entreprise non précisée"
        )

    @staticmethod
    def _contract_label(offer: dict) -> str | None:
        """Libellé court du contrat, dans le vocabulaire de `config.yaml`.

        Le pré-filtrage local compare cette valeur aux contrats configurés
        (« CDI », « stage », « alternance ») : lui donner « Contrat à durée
        indéterminée » ferait rejeter l'offre par un filtre censé l'accepter.

        L'ordre n'est pas anodin, et il vient de l'observation des offres
        réelles :

        1. **L'alternance d'abord**, car elle est bien typée (nature
           apprentissage ou professionnalisation, `alternance: true`) — y
           compris sur des annonces intitulées « Stage de césure ».
        2. **Le type de contrat ensuite**, pour obtenir un libellé court.
        3. **L'intitulé en dernier mot**, parce que les champs de contrat
           mentent sur les stages et les alternances : « Stage : Ingénieur en
           informatique » est publiée en `CDI` / `Contrat travail`. Entre un
           champ démenti par les faits et un titre explicite, `contract_from_title`
           retient le titre.
        """
        nature = fold(str(offer.get("natureContrat") or ""))
        if (
            "apprentissage" in nature
            or "professionnalisation" in nature
            or offer.get("alternance") is True
        ):
            return "Alternance"

        code = str(offer.get("typeContrat") or "")
        label = (
            _CONTRACT_LABELS.get(code) or offer.get("typeContratLibelle") or code or None
        )
        return contract_from_title(str(offer.get("intitule") or ""), label)

    @staticmethod
    def _parse_date(value: Any) -> datetime | None:
        if not isinstance(value, str) or not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            logger.debug("france_travail: date illisible: %r", value)
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    # ---------------------------------------------------------------- détail

    def enrich(self, job: Job) -> Job:
        """Complète une offre — et ne requête rien si elle l'est déjà.

        La recherche renvoyant la description, ce cas est le cas courant :
        `enrich()` n'existe ici que pour respecter l'interface commune et pour
        rattraper les offres tronquées.
        """
        if job.is_enriched:
            return job

        offer_id = job.source_ref.get("offer_id")
        if not offer_id:
            logger.warning("france_travail: offre %s sans identifiant, détail impossible", job.id)
            return job

        if self._request_delay:
            time.sleep(self._request_delay)

        try:
            offer = self._get(f"/offres/{offer_id}").json() or {}
        except (SourceError, ValueError) as exc:
            logger.warning("france_travail: détail indisponible pour %s — %s", job.url, exc)
            return job

        detailed = self._to_job(offer)
        if detailed is None:
            return job
        return job.model_copy(
            update={
                "description": detailed.description,
                "apply_url": detailed.apply_url or job.apply_url,
                "salary": job.salary or detailed.salary,
                "raw": offer,
            }
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> FranceTravailSource:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
