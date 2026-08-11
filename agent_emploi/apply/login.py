"""Connexion aux sites d'emploi, depuis le programme.

Le contexte Chromium est persistant : une fois connecté, on le reste d'une
passe à l'autre. Se connecter *depuis le programme* n'est donc pas une
nécessité technique, c'est un confort — celui de ne pas avoir à retrouver la
page de connexion à la main quand la session expire.

Ce que ce module s'interdit :

- **Le mot de passe ne touche jamais le disque.** Il vient d'une variable
  d'environnement (`.env`, hors dépôt) ou d'une saisie masquée, il vit dans une
  variable locale le temps de la connexion, et il n'apparaît dans aucun
  journal — les traces de ce module ne mentionnent jamais que le site et
  l'issue.
- **Aucun sélecteur codé en dur.** Les pages de connexion changent ; le champ
  mot de passe, lui, reste un `input[type=password]`. On lit le formulaire
  comme on lit un formulaire de candidature, avec le même extracteur.
- **Un captcha ou un code à deux facteurs arrête tout.** La fenêtre est
  visible : le module rend la main et laisse l'utilisateur finir lui-même, la
  session ainsi ouverte est conservée comme n'importe quelle autre.

Le bouton cliqué ici est celui d'un formulaire de connexion, sur une URL de
connexion connue d'avance. C'est sans rapport avec l'envoi d'une candidature,
qu'aucun chemin de code ne déclenche.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from getpass import getpass
from typing import Callable

from agent_emploi.apply.browser import BrowserUnavailable, PlaywrightBrowser
from agent_emploi.apply.fields import FormField
from agent_emploi.profile import fold

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Site:
    """Un site sur lequel on sait se connecter."""

    key: str
    label: str
    login_url: str
    #: Domaines dont les formulaires bénéficient de cette session.
    domains: tuple[str, ...] = ()

    @property
    def env_user(self) -> str:
        return f"{self.key.upper()}_EMAIL"

    @property
    def env_password(self) -> str:
        return f"{self.key.upper()}_PASSWORD"


#: Sites connus. La plupart des ATS (Greenhouse, Lever) acceptent une
#: candidature sans compte : y ajouter une connexion n'apporterait rien.
#:
#: Attention à ne pas confondre les deux comptes France Travail : celui-ci est
#: le **compte candidat** (celui de l'espace personnel), qui sert à postuler sur
#: `candidat.francetravail.fr`. Il n'a rien à voir avec les identifiants
#: d'application `FRANCE_TRAVAIL_CLIENT_ID` / `_CLIENT_SECRET`, qui ouvrent
#: l'API de recherche et viennent d'un compte développeur `francetravail.io`.
SITES: dict[str, Site] = {
    "wttj": Site(
        key="wttj",
        label="Welcome to the Jungle",
        login_url="https://www.welcometothejungle.com/fr/signin",
        domains=("welcometothejungle.com",),
    ),
    "france_travail": Site(
        key="france_travail",
        label="France Travail (espace personnel)",
        login_url="https://candidat.francetravail.fr/espacepersonnel/",
        domains=("francetravail.fr",),
    ),
}


class LoginError(RuntimeError):
    """Connexion impossible — le message est sûr à afficher."""


@dataclass(frozen=True)
class Credentials:
    """Identifiants d'un site, le temps d'une connexion."""

    user: str
    password: str

    def __repr__(self) -> str:  # pragma: no cover - trivial
        # Un `repr` par défaut ferait fuiter le mot de passe dans la première
        # trace venue : trace d'exception, `logger.debug`, session `pdb`.
        return f"Credentials(user={self.user!r}, password=***)"


def resolve_credentials(
    site: Site,
    *,
    user: str | None = None,
    ask: Callable[[str], str] = input,
    ask_secret: Callable[[str], str] = getpass,
) -> Credentials:
    """Identifiants du site : environnement d'abord, saisie ensuite.

    L'environnement l'emporte pour que la passe puisse tourner sans terminal
    interactif ; la saisie masquée existe pour ne pas obliger à écrire un mot de
    passe dans un fichier. Rien n'est mémorisé entre deux appels.
    """
    identifier = user or os.environ.get(site.env_user) or ""
    if not identifier:
        identifier = ask(f"Identifiant {site.label} : ").strip()
    if not identifier:
        raise LoginError(f"aucun identifiant fourni pour {site.label}")

    password = os.environ.get(site.env_password) or ""
    if not password:
        password = ask_secret(f"Mot de passe {site.label} (saisie masquée) : ")
    if not password:
        raise LoginError(
            f"aucun mot de passe fourni pour {site.label} — définissez "
            f"{site.env_password} dans .env ou saisissez-le à l'invite"
        )
    return Credentials(user=identifier, password=password)


def _find(fields: list[FormField], kinds: tuple[str, ...], needles: tuple[str, ...]):
    """Premier champ dont le type convient et dont le libellé mord."""
    for field_ in fields:
        if field_.kind in kinds and any(
            needle in field_.haystack for needle in needles
        ):
            return field_
    for field_ in fields:
        if field_.kind in kinds:
            return field_
    return None


def is_signed_in(page) -> bool:
    """Vrai si la page de connexion ne demande plus rien.

    Les sites redirigent un visiteur déjà connecté hors de leur page de
    connexion : l'absence de champ mot de passe est le signal le plus fiable, et
    le seul qui ne dépende pas de la mise en page du jour.
    """
    return not any(field_.kind == "password" for field_ in page.fields())


def sign_in(
    browser: PlaywrightBrowser,
    site: Site,
    credentials: Credentials,
    *,
    timeout_ms: int = 20000,
) -> str:
    """Ouvre la page de connexion et s'y connecte. Retourne un message d'issue.

    Lève `LoginError` si la page ne se laisse pas remplir — captcha, double
    authentification, formulaire méconnaissable. Dans ce cas la fenêtre reste
    ouverte à la bonne page : l'utilisateur finit à la main, et la session est
    conservée dans le profil persistant comme si le programme l'avait faite.
    """
    page = browser.open(site.login_url)

    if is_signed_in(page):
        return f"{site.label} : session déjà ouverte"

    blockers = page.blockers()
    captcha = [blocker for blocker in blockers if "captcha" in fold(blocker)]
    if captcha:
        raise LoginError(
            f"{site.label} : {captcha[0]} — connectez-vous dans la fenêtre "
            "ouverte, la session sera conservée pour les passes suivantes"
        )

    fields = page.fields()
    password_field = _find(fields, ("password",), ("password", "mot de passe"))
    user_field = _find(
        fields, ("text", "email"), ("email", "e-mail", "identifiant", "user", "login")
    )
    if password_field is None or user_field is None:
        raise LoginError(
            f"{site.label} : formulaire de connexion méconnaissable — "
            "connectez-vous dans la fenêtre ouverte, la session sera conservée"
        )

    page.fill(user_field, credentials.user)
    page.fill(password_field, credentials.password)

    # On travaille ici sur la page Playwright brute, délibérément hors du
    # protocole `Page` : celui-ci n'expose aucun moyen de valider un
    # formulaire, et c'est ce qui garantit qu'une candidature ne peut pas être
    # envoyée par mégarde. La validation ci-dessous est celle d'un formulaire
    # de connexion, sur une URL de connexion fixée dans `SITES`.
    raw = getattr(page, "page", None)
    if raw is None:  # pragma: no cover - seulement en test avec une page factice
        raise LoginError(f"{site.label} : page non pilotable")
    try:
        raw.locator(password_field.selector).first.press("Enter")
        raw.wait_for_load_state("networkidle", timeout=timeout_ms)
    except Exception as exc:  # noqa: BLE001 — l'issue est vérifiée juste après
        logger.debug("validation du formulaire de connexion %s: %s", site.key, exc)

    if is_signed_in(page):
        logger.info("connexion réussie: %s", site.key)
        return f"{site.label} : connecté"

    raise LoginError(
        f"{site.label} : la connexion n'a pas abouti (mot de passe refusé, "
        "code à deux facteurs, ou vérification supplémentaire) — terminez dans "
        "la fenêtre ouverte, la session sera conservée"
    )


def sign_in_sites(
    browser: PlaywrightBrowser,
    keys: list[str],
    *,
    user: str | None = None,
    ask: Callable[[str], str] = input,
    ask_secret: Callable[[str], str] = getpass,
) -> tuple[list[str], list[str]]:
    """Connecte plusieurs sites. Retourne (réussites, échecs) — jamais d'exception.

    Un site qui refuse la connexion ne doit pas empêcher la passe de
    candidature : au pire, les formulaires de ce site partiront en `handoff`,
    ce qu'ils auraient fait de toute façon.
    """
    done: list[str] = []
    failed: list[str] = []
    for key in keys:
        site = SITES.get(key)
        if site is None:
            failed.append(f"site inconnu : {key} (connus : {', '.join(sorted(SITES))})")
            continue
        try:
            credentials = resolve_credentials(
                site, user=user, ask=ask, ask_secret=ask_secret
            )
            done.append(sign_in(browser, site, credentials))
        except (LoginError, BrowserUnavailable) as exc:
            failed.append(str(exc))
        except Exception as exc:  # noqa: BLE001 — le message ne porte aucun secret
            failed.append(f"{site.label} : connexion en échec ({type(exc).__name__})")
    return done, failed
