"""Pré-remplissage d'une offre depuis son URL, via le JSON-LD `JobPosting`.

Les sites d'emploi publient leurs offres en `schema.org/JobPosting` pour Google
for Jobs : LinkedIn public, Indeed, WTTJ, Greenhouse, Lever, Workable… Un seul
format pour tous, donc un seul lecteur, plutôt qu'un extracteur par site qui
casserait à chaque refonte. Quand la page n'en publie pas (connexion requise,
page rendue en JavaScript), le formulaire se remplit à la main : rien ici n'est
indispensable.

Aucune dépendance d'analyse HTML : les blocs `ld+json` se repèrent par
expression régulière, et la description se ramène à du texte en quelques
substitutions.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from datetime import datetime

import httpx

#: Au-delà, la page n'est pas lue en entier : les blocs JSON-LD sont dans
#: l'en-tête ou tôt dans le corps.
MAX_BYTES = 3_000_000

TIMEOUT = 15.0

#: Un navigateur ordinaire : plusieurs sites servent une page vide, ou un 403,
#: à un client qui s'annonce comme une bibliothèque.
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

_LD_BLOCK = re.compile(
    r"<script[^>]*type\s*=\s*[\"']application/ld\+json[\"'][^>]*>(.*?)</script>",
    re.IGNORECASE | re.DOTALL,
)
_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_META = re.compile(r"<meta\b[^>]*>", re.IGNORECASE)
_ATTR = re.compile(r"""(\w[\w:-]*)\s*=\s*(?:"([^"]*)"|'([^']*)')""")

#: `employmentType` de schema.org, traduit quand il désigne un contrat. Les
#: autres valeurs (FULL_TIME, PART_TIME…) parlent du temps de travail : elles
#: sont laissées telles quelles, à corriger dans le formulaire.
CONTRACTS = {
    "INTERN": "Stage",
    "INTERNSHIP": "Stage",
    "TEMPORARY": "CDD",
    "CONTRACTOR": "Freelance",
    "APPRENTICESHIP": "Alternance",
}


class PrefillError(RuntimeError):
    """La page n'a pas pu être téléchargée."""


@dataclass
class Prefill:
    """Ce qu'on a pu lire de l'offre. Tout champ peut rester vide."""

    title: str = ""
    company: str = ""
    location: str = ""
    contract: str = ""
    description: str = ""
    apply_url: str = ""
    remote: str = ""
    posted_at: datetime | None = None
    #: Vrai si un `JobPosting` a été trouvé ; faux si seul le `<title>` a servi.
    structured: bool = False
    notes: list[str] = field(default_factory=list)


def fetch(url: str, *, client: httpx.Client | None = None) -> str:
    """Télécharge la page. Lève `PrefillError` sur toute erreur réseau ou HTTP."""
    if not url.lower().startswith(("http://", "https://")):
        raise PrefillError("seules les adresses http(s) se pré-remplissent")
    own = client is None
    client = client or httpx.Client(
        timeout=TIMEOUT,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT, "Accept-Language": "fr,en;q=0.8"},
    )
    try:
        response = client.get(url)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise PrefillError(f"la page a répondu {exc.response.status_code}") from exc
    except httpx.HTTPError as exc:
        raise PrefillError(f"page injoignable ({exc.__class__.__name__})") from exc
    finally:
        if own:
            client.close()
    return response.text[:MAX_BYTES]


def prefill(url: str, *, client: httpx.Client | None = None) -> Prefill:
    """Télécharge l'offre et en extrait ce qu'elle publie."""
    return parse(fetch(url, client=client))


def parse(page: str) -> Prefill:
    """Extrait le premier `JobPosting` de la page, à défaut son `<title>`."""
    for posting in _postings(page):
        return _from_posting(posting)

    # Repli : les balises OpenGraph, que presque toutes les pages publient
    # pour les aperçus de lien. Plus précises que `<title>`, qui ajoute souvent
    # le nom du site ; mais sans description exploitable.
    result = Prefill()
    meta = _opengraph(page)
    result.title = meta.get("og:title", "")
    result.company = meta.get("og:site_name", "")
    if not result.title:
        match = _TITLE.search(page)
        if match:
            result.title = _clean(match.group(1))
    result.notes.append(
        "aucune donnée structurée sur cette page : complétez les champs à la main"
    )
    return result


def _postings(page: str):
    """Les objets `JobPosting` de tous les blocs JSON-LD, dans l'ordre."""
    for block in _LD_BLOCK.findall(page):
        data = _load_block(block.strip())
        if data is not None:
            yield from _walk(data)


def _load_block(block: str):
    """Le JSON d'un bloc, ou `None` s'il reste illisible.

    Le bloc brut d'abord : le déséchapper d'office casserait une chaîne qui
    contient `&quot;`. `strict=False` tolère les sauts de ligne bruts que
    certains sites laissent dans les chaînes ; le déséchappement ne sert qu'aux
    pages qui ont encodé tout le bloc en entités.
    """
    for candidate in (block, html.unescape(block)):
        try:
            return json.loads(candidate, strict=False)
        except json.JSONDecodeError:
            continue
    return None


def _opengraph(page: str) -> dict[str, str]:
    """Les balises `<meta property="og:…">`, quel que soit l'ordre des attributs."""
    found: dict[str, str] = {}
    for tag in _META.findall(page):
        attrs = {
            name.lower(): double or single for name, double, single in _ATTR.findall(tag)
        }
        key = attrs.get("property") or attrs.get("name")
        if key and key.startswith("og:") and attrs.get("content"):
            found.setdefault(key, _clean(attrs["content"]))
    return found


def _walk(data):
    if isinstance(data, list):
        for item in data:
            yield from _walk(item)
    elif isinstance(data, dict):
        kind = data.get("@type")
        kinds = kind if isinstance(kind, list) else [kind]
        if "JobPosting" in kinds:
            yield data
        for item in data.get("@graph", []) or []:
            yield from _walk(item)


def _from_posting(posting: dict) -> Prefill:
    result = Prefill(structured=True)
    result.title = _clean(_text(posting.get("title")))

    organization = posting.get("hiringOrganization")
    if isinstance(organization, dict):
        result.company = _clean(_text(organization.get("name")))
    else:
        result.company = _clean(_text(organization))

    result.location = _location(posting.get("jobLocation"))
    result.contract = _contract(posting.get("employmentType"))
    result.description = html_to_text(_text(posting.get("description")))

    location_type = _text(posting.get("jobLocationType")).upper()
    if "TELECOMMUTE" in location_type:
        result.remote = "full"

    apply_url = posting.get("directApplyUrl") or posting.get("url")
    if isinstance(apply_url, str) and apply_url.startswith(("http://", "https://")):
        result.apply_url = apply_url

    posted = _text(posting.get("datePosted"))
    if posted:
        try:
            result.posted_at = datetime.fromisoformat(posted.replace("Z", "+00:00"))
        except ValueError:
            pass

    missing = [
        label
        for label, value in (
            ("intitulé", result.title),
            ("entreprise", result.company),
            ("description", result.description),
        )
        if not value
    ]
    if missing:
        result.notes.append(f"absent de la page : {', '.join(missing)}")
    return result


def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(_text(item) for item in value if item)
    return str(value)


def _clean(text: str) -> str:
    """Espaces compactés, entités décodées : Indeed écrit `&amp;` jusque dans
    le titre."""
    return " ".join(html.unescape(text).split())


def _location(value) -> str:
    places = value if isinstance(value, list) else [value]
    labels: list[str] = []
    for place in places:
        if not isinstance(place, dict):
            continue
        address = place.get("address")
        if isinstance(address, str):
            labels.append(_clean(address))
            continue
        if not isinstance(address, dict):
            continue
        country = address.get("addressCountry")
        if isinstance(country, dict):
            country = country.get("name")
        parts = [
            _clean(_text(part))
            for part in (address.get("addressLocality"), country)
            if part
        ]
        if parts:
            labels.append(", ".join(parts))
    # Plusieurs sites listent le même lieu deux fois.
    return " / ".join(dict.fromkeys(labels))


def _contract(value) -> str:
    kinds = value if isinstance(value, list) else [value]
    labels = [
        CONTRACTS.get(str(kind).upper(), str(kind)) for kind in kinds if kind
    ]
    return ", ".join(dict.fromkeys(labels))


# `</li>` n'y est pas : chaque `<li>` ouvre déjà sa ligne, et fermer aussi
# sautait une ligne vide entre deux puces.
_BREAKS = re.compile(r"<\s*(br|/p|/div|/h\d|/ul|/ol)\b[^>]*>", re.IGNORECASE)
_ITEMS = re.compile(r"<\s*li\b[^>]*>", re.IGNORECASE)
_TAGS = re.compile(r"<[^>]+>")
_BLANK_LINES = re.compile(r"\n{3,}")


def html_to_text(fragment: str) -> str:
    """Ramène une description HTML à du texte lisible, listes et paragraphes gardés.

    Certaines pages échappent le HTML de la description (`&lt;p&gt;`) : il est
    déséchappé une première fois avant de retirer les balises.
    """
    if "&lt;" in fragment:
        fragment = html.unescape(fragment)
    text = _ITEMS.sub("\n- ", fragment)
    text = _BREAKS.sub("\n", text)
    text = html.unescape(_TAGS.sub("", text))
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()
