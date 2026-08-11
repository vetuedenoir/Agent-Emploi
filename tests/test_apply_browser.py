"""Le navigateur, sur une page de formulaire figée en local.

Ces tests-là ouvrent un vrai Chromium : ils sont désactivés par défaut, comme
les tests réseau. Ils vérifient la seule chose qu'une page factice ne peut pas
vérifier — que l'extracteur JavaScript lit correctement un vrai DOM, libellés,
types et champs obligatoires compris.

    pytest -m browser        # après `playwright install chromium`
"""

from __future__ import annotations

import pytest

from agent_emploi.apply.browser import launch
from agent_emploi.apply.fields import Slot, build_plan
from agent_emploi.config import BrowserConfig

pytestmark = pytest.mark.browser

FORM = """<!doctype html>
<html lang="fr"><meta charset="utf-8"><title>Candidature</title>
<form>
  <label for="fn">Prénom</label><input id="fn" name="first_name" required>
  <label for="ln">Nom de famille</label><input id="ln" name="last_name" required>
  <label>Email <input type="email" name="email" required></label>
  <label for="cv">CV</label><input id="cv" type="file" name="resume" required>
  <label for="lm">Lettre de motivation</label><textarea id="lm" name="cover_letter"></textarea>
  <label for="rgpd"><input id="rgpd" type="checkbox" required> J'accepte</label>
  <input type="hidden" name="token" value="x">
  <input type="text" name="cache" style="display:none">
  <button type="submit">Envoyer</button>
</form>
</html>
"""


@pytest.fixture
def page(tmp_path):
    form = tmp_path / "form.html"
    form.write_text(FORM, encoding="utf-8")
    config = BrowserConfig(user_data_dir=tmp_path / "chromium", headless=True)
    with launch(config) as browser:
        yield browser.open(form.as_uri())


def test_extraction_des_champs(page):
    fields = {field.name: field for field in page.fields()}

    # Ni le champ caché, ni le champ masqué en CSS, ni le bouton d'envoi.
    assert set(fields) == {
        "first_name",
        "last_name",
        "email",
        "resume",
        "cover_letter",
        "",
    }
    assert fields["first_name"].label == "Prénom"
    assert fields["first_name"].required
    assert fields["email"].kind == "email"
    assert fields["resume"].kind == "file"
    assert fields["cover_letter"].kind == "textarea"


def test_remplissage_reel(page, tmp_path):
    cv = tmp_path / "cv.pdf"
    cv.write_bytes(b"%PDF-1.4\n")

    plan = build_plan(
        page.fields(),
        {
            Slot.FIRST_NAME: "Killian",
            Slot.LAST_NAME: "Scordel",
            Slot.EMAIL: "killian@exemple.fr",
            Slot.CV: str(cv),
            Slot.COVER_LETTER: "Madame, Monsieur,",
        },
    )
    assert plan.ok
    # La case RGPD est obligatoire mais reste à l'utilisateur.
    assert any("accepte" in todo for todo in plan.todo)

    for assignment in plan.assignments:
        if assignment.is_file:
            page.attach(assignment.field, cv)
        else:
            page.fill(assignment.field, assignment.value)

    values = page.page.evaluate(
        "() => ({fn: document.getElementById('fn').value,"
        " lm: document.getElementById('lm').value,"
        " cv: document.getElementById('cv').files.length})"
    )
    assert values == {"fn": "Killian", "lm": "Madame, Monsieur,", "cv": 1}

    # Le formulaire est rempli et n'a pas été soumis : la page n'a pas navigué.
    assert page.page.evaluate("() => document.getElementById('fn').value") == "Killian"

    shot = page.screenshot(tmp_path / "formulaire.png")
    assert shot is not None and shot.exists()
