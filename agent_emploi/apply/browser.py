"""Le navigateur — la seule partie de l'étape 6 qui touche une vraie page.

Playwright est importé à l'intérieur des fonctions, pas au sommet du module :
les étapes 1 à 5 n'ont aucune raison d'exiger un navigateur installé, et
`pytest` doit pouvoir dérouler toute la passe sans Chromium (voir le protocole
`Page` ci-dessous, que les tests implémentent en quelques lignes).

Le contexte est **persistant** : une connexion faite à la main sur un ATS reste
valable à la passe suivante. C'est ce qui distingue un `handoff` définitif d'un
`handoff` que l'utilisateur peut lever une fois pour toutes.

**Aucune fonction de ce module ne clique sur un bouton d'envoi.** Il n'existe
pas de méthode `submit`, et c'est la forme la plus solide de garantie : on ne
peut pas appeler ce qui n'est pas écrit.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

from agent_emploi.apply.fields import FormField
from agent_emploi.config import BrowserConfig

logger = logging.getLogger(__name__)


class BrowserUnavailable(RuntimeError):
    """Playwright absent, ou Chromium non installé."""


class Page(Protocol):
    """Ce que la passe de candidature attend d'une page ouverte.

    Volontairement étroit : lire les champs, remplir, joindre, photographier.
    Rien pour soumettre.
    """

    url: str

    def fields(self) -> list[FormField]: ...

    def fill(self, field: FormField, value: str) -> None: ...

    def attach(self, field: FormField, path: Path) -> None: ...

    def screenshot(self, path: Path) -> Path | None: ...

    def blockers(self) -> list[str]: ...


#: Extrait les champs visibles d'une page et les étiquette au passage.
#:
#: L'étiquetage (`data-ae-field`) est ce qui permet de rendre un sélecteur
#: fiable : les formulaires d'ATS sont pleins de champs sans `id` ni `name`
#: stable, et un sélecteur positionnel se périme dès qu'un champ conditionnel
#: apparaît. L'attribut, lui, reste collé à l'élément.
_COLLECT_JS = """
() => {
  const labelOf = (el) => {
    if (el.labels && el.labels.length) return el.labels[0].innerText;
    if (el.getAttribute('aria-label')) return el.getAttribute('aria-label');
    const by = el.getAttribute('aria-labelledby');
    if (by) {
      const ref = document.getElementById(by);
      if (ref) return ref.innerText;
    }
    const wrapper = el.closest('label');
    if (wrapper) return wrapper.innerText;
    return '';
  };
  const visible = (el) => {
    const style = getComputedStyle(el);
    if (style.display === 'none' || style.visibility === 'hidden') return false;
    // Les champs fichier sont couramment masqués derrière un bouton stylé :
    // invisibles à l'œil, ils restent la cible légitime d'un téléversement.
    if (el.type === 'file') return true;
    return el.getClientRects().length > 0;
  };
  const skip = new Set(['hidden', 'submit', 'button', 'image', 'reset']);
  const out = [];
  let index = 0;
  for (const el of document.querySelectorAll('input, textarea, select')) {
    const type = el.tagName === 'TEXTAREA' ? 'textarea'
               : el.tagName === 'SELECT' ? 'select'
               : (el.type || 'text').toLowerCase();
    if (skip.has(type) || el.disabled || el.readOnly) continue;
    if (!visible(el)) continue;
    el.setAttribute('data-ae-field', String(index));
    out.push({
      selector: `[data-ae-field="${index}"]`,
      kind: type,
      name: el.name || '',
      field_id: el.id || '',
      label: (labelOf(el) || '').trim().slice(0, 200),
      placeholder: el.placeholder || '',
      autocomplete: el.getAttribute('autocomplete') || '',
      required: el.required || el.getAttribute('aria-required') === 'true',
      value: type === 'file' ? '' : (el.value || ''),
      options: el.tagName === 'SELECT'
        ? Array.from(el.options).map((o) => o.text.trim()).slice(0, 40)
        : [],
    });
    index += 1;
  }
  return out;
}
"""

#: Marqueurs de captcha : leur seule présence suffit à rendre la main. Tenter
#: de les contourner serait à la fois vain et déloyal.
_CAPTCHA_MARKERS = ("recaptcha", "hcaptcha", "turnstile", "captcha")

#: Marqueurs d'une page qui demande d'abord de se connecter.
_LOGIN_MARKERS = ("sign in", "log in", "se connecter", "connexion", "identifiez-vous")


@dataclass
class PlaywrightPage:
    """Adaptateur d'une page Playwright vers le protocole `Page`."""

    page: object
    timeout_ms: int = 20000

    @property
    def url(self) -> str:
        return self.page.url

    def fields(self) -> list[FormField]:
        raw = self.page.evaluate(_COLLECT_JS)
        return [FormField(**item) for item in raw]

    def fill(self, field: FormField, value: str) -> None:
        locator = self.page.locator(field.selector).first
        locator.scroll_into_view_if_needed(timeout=self.timeout_ms)
        locator.fill(value, timeout=self.timeout_ms)

    def attach(self, field: FormField, path: Path) -> None:
        self.page.locator(field.selector).first.set_input_files(
            str(path), timeout=self.timeout_ms
        )

    def screenshot(self, path: Path) -> Path | None:
        """Photographie la page entière. Un échec ici n'arrête pas la passe."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.page.screenshot(path=str(path), full_page=True)
        except Exception as exc:  # noqa: BLE001 — une capture ratée reste un détail
            logger.warning("capture d'écran impossible (%s): %s", path, exc)
            return None
        return path

    def blockers(self) -> list[str]:
        """Ce qui empêche un remplissage automatique d'aboutir.

        Détecté avant le remplissage : inutile de poser un nom et un téléphone
        sur une page qui exige d'abord une connexion.
        """
        found: list[str] = []
        html = ""
        try:
            html = (self.page.content() or "").lower()
        except Exception as exc:  # noqa: BLE001
            logger.debug("contenu de page illisible: %s", exc)

        if any(marker in html for marker in _CAPTCHA_MARKERS):
            found.append("captcha détecté sur la page")

        if self.page.locator("input[type=password]").count() > 0:
            found.append("connexion requise (champ mot de passe)")
        else:
            visible_text = ""
            try:
                visible_text = (self.page.locator("body").inner_text() or "").lower()
            except Exception as exc:  # noqa: BLE001
                logger.debug("texte de page illisible: %s", exc)
            head = visible_text[:400]
            if any(marker in head for marker in _LOGIN_MARKERS):
                found.append("la page semble demander une connexion")

        return found


@dataclass
class PlaywrightBrowser:
    """Ouvre les pages de candidature, une par offre, dans un même contexte."""

    context: object
    config: BrowserConfig

    def open(self, url: str) -> PlaywrightPage:
        """Ouvre l'URL dans un nouvel onglet et attend la fin du chargement.

        Chaque offre a son onglet, et les onglets restent ouverts : à la fin de
        la passe l'utilisateur a devant lui tous les formulaires remplis, prêts
        à être vérifiés puis envoyés par ses soins.
        """
        page = self.context.new_page()
        page.set_default_timeout(self.config.timeout_ms)
        page.goto(url, wait_until="domcontentloaded", timeout=self.config.timeout_ms)
        try:
            page.wait_for_load_state("networkidle", timeout=self.config.timeout_ms)
        except Exception as exc:  # noqa: BLE001 — page bavarde, pas page cassée
            logger.debug("networkidle non atteint sur %s: %s", url, exc)
        return PlaywrightPage(page=page, timeout_ms=self.config.timeout_ms)


@contextmanager
def launch(config: BrowserConfig) -> Iterator[PlaywrightBrowser]:
    """Ouvre un Chromium à contexte persistant, et le referme à la sortie.

    Lève `BrowserUnavailable` avec la commande d'installation quand Playwright
    ou Chromium manque : c'est l'échec le plus probable au premier lancement de
    l'étape 6, il ne doit pas ressembler à un bogue.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise BrowserUnavailable(
            "Playwright n'est pas installé — `uv pip install -e \".[apply]\"` "
            "puis `playwright install chromium`"
        ) from exc

    config.user_data_dir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as driver:
        try:
            context = driver.chromium.launch_persistent_context(
                str(config.user_data_dir),
                headless=config.headless,
                slow_mo=config.slow_mo_ms,
                accept_downloads=False,
            )
        except Exception as exc:  # noqa: BLE001 — remonté tel quel à l'utilisateur
            raise BrowserUnavailable(
                f"Chromium n'a pas pu démarrer ({exc}) — "
                "essayez `playwright install chromium`"
            ) from exc
        try:
            yield PlaywrightBrowser(context=context, config=config)
        finally:
            context.close()
