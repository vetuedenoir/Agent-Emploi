"""Connexion depuis le programme : identifiants, issues, et non-divulgation.

Les tests vérifient surtout ce que le module ne doit *pas* faire — écrire le
mot de passe quelque part, le laisser filer dans une trace, ou prétendre qu'une
connexion a réussi quand la page dit le contraire.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from agent_emploi.apply.fields import FormField
from agent_emploi.apply.login import (
    SITES,
    Credentials,
    LoginError,
    resolve_credentials,
    sign_in,
    sign_in_sites,
)

SITE = SITES["wttj"]


class FakeRawPage:
    """La page Playwright brute : seule la validation du formulaire y passe."""

    def __init__(self) -> None:
        self.pressed: list[str] = []

    def locator(self, selector):
        page = self

        class Locator:
            first = property(lambda self: self)

            def press(self, key):
                page.pressed.append(f"{selector}:{key}")

        return Locator().first

    def wait_for_load_state(self, *args, **kwargs):
        return None


class FakeLoginPage:
    """Page de connexion : les champs disparaissent une fois connecté."""

    def __init__(self, fields, blockers=None, succeeds=True):
        self.url = SITE.login_url
        self._fields = fields
        self._blockers = blockers or []
        self._succeeds = succeeds
        self.filled: dict[str, str] = {}
        self.page = FakeRawPage()

    def fields(self):
        if self._succeeds and self.page.pressed:
            return []
        return list(self._fields)

    def fill(self, field, value):
        self.filled[field.selector] = value

    def attach(self, field, path):  # pragma: no cover - inutilisé ici
        raise AssertionError("une page de connexion ne reçoit pas de pièce jointe")

    def screenshot(self, path):  # pragma: no cover - inutilisé ici
        return None

    def blockers(self):
        return list(self._blockers)


class FakeBrowser:
    def __init__(self, page):
        self._page = page
        self.opened: list[str] = []

    def open(self, url):
        self.opened.append(url)
        return self._page


def login_fields():
    return [
        FormField(selector="u", kind="email", label="Email"),
        FormField(selector="p", kind="password", label="Mot de passe"),
    ]


# ------------------------------------------------------------------ identifiants


def test_les_variables_denvironnement_priment(monkeypatch):
    monkeypatch.setenv("WTTJ_EMAIL", "moi@exemple.fr")
    monkeypatch.setenv("WTTJ_PASSWORD", "secret")

    def refuse(prompt):  # pragma: no cover - ne doit pas être appelé
        raise AssertionError("aucune saisie ne doit être demandée")

    credentials = resolve_credentials(SITE, ask=refuse, ask_secret=refuse)

    assert credentials == Credentials(user="moi@exemple.fr", password="secret")


def test_saisie_masquee_quand_lenvironnement_est_vide(monkeypatch):
    monkeypatch.delenv("WTTJ_EMAIL", raising=False)
    monkeypatch.delenv("WTTJ_PASSWORD", raising=False)
    asked: list[str] = []

    credentials = resolve_credentials(
        SITE,
        ask=lambda prompt: (asked.append(prompt), "moi@exemple.fr")[1],
        ask_secret=lambda prompt: (asked.append(prompt), "secret")[1],
    )

    assert credentials.password == "secret"
    assert any("masquée" in prompt for prompt in asked)


def test_mot_de_passe_absent_est_une_erreur_explicite(monkeypatch):
    monkeypatch.setenv("WTTJ_EMAIL", "moi@exemple.fr")
    monkeypatch.delenv("WTTJ_PASSWORD", raising=False)

    with pytest.raises(LoginError, match="WTTJ_PASSWORD"):
        resolve_credentials(SITE, ask=lambda _: "", ask_secret=lambda _: "")


def test_le_mot_de_passe_ne_figure_pas_dans_les_traces():
    """`repr` est ce qui fuit en premier : trace d'exception, log, pdb."""
    credentials = Credentials(user="moi@exemple.fr", password="tr3s-secret")

    assert "tr3s-secret" not in repr(credentials)
    assert "moi@exemple.fr" in repr(credentials)


# --------------------------------------------------------------------- connexion


def test_connexion_reussie():
    page = FakeLoginPage(login_fields())
    browser = FakeBrowser(page)

    message = sign_in(browser, SITE, Credentials("moi@exemple.fr", "secret"))

    assert browser.opened == [SITE.login_url]
    assert page.filled == {"u": "moi@exemple.fr", "p": "secret"}
    assert page.page.pressed == ["p:Enter"]
    assert "connecté" in message


def test_session_deja_ouverte_ne_ressaisit_rien():
    """Le contexte est persistant : une session ouverte doit être reconnue."""
    page = FakeLoginPage([])
    browser = FakeBrowser(page)

    message = sign_in(browser, SITE, Credentials("moi@exemple.fr", "secret"))

    assert page.filled == {}
    assert "déjà ouverte" in message


def test_captcha_rend_la_main_sans_saisir_le_mot_de_passe():
    page = FakeLoginPage(login_fields(), blockers=["captcha détecté sur la page"])

    with pytest.raises(LoginError, match="captcha"):
        sign_in(FakeBrowser(page), SITE, Credentials("moi@exemple.fr", "secret"))

    assert page.filled == {}


def test_connexion_refusee_est_signalee():
    """Mot de passe faux ou double authentification : on ne prétend pas avoir réussi."""
    page = FakeLoginPage(login_fields(), succeeds=False)

    with pytest.raises(LoginError, match="n'a pas abouti"):
        sign_in(FakeBrowser(page), SITE, Credentials("moi@exemple.fr", "faux"))


def test_formulaire_meconnaissable_renvoie_a_la_fenetre():
    """Un mot de passe attendu sans champ où mettre l'identifiant : on renonce."""
    page = FakeLoginPage([FormField(selector="p", kind="password", label="Mot de passe")])

    with pytest.raises(LoginError, match="méconnaissable"):
        sign_in(FakeBrowser(page), SITE, Credentials("moi@exemple.fr", "secret"))

    assert page.filled == {}


def test_un_site_en_echec_ninterrompt_pas_la_passe(monkeypatch):
    monkeypatch.setenv("WTTJ_EMAIL", "moi@exemple.fr")
    monkeypatch.setenv("WTTJ_PASSWORD", "faux")
    page = FakeLoginPage(login_fields(), succeeds=False)

    done, failed = sign_in_sites(FakeBrowser(page), ["wttj", "inconnu"])

    assert done == []
    assert len(failed) == 2
    assert "faux" not in " ".join(failed)


def test_aucun_mot_de_passe_nest_ecrit_sur_disque():
    """Le module n'ouvre ni n'écrit aucun fichier — le secret reste en mémoire.

    `browser.open()` ouvre une page, pas un fichier : le motif ne retient que
    les appels à `open` sans receveur.
    """
    source = Path("agent_emploi/apply/login.py").read_text(encoding="utf-8")

    assert "write_text" not in source
    assert "write_bytes" not in source
    assert re.search(r"(?<![.\w])open\(", source) is None
