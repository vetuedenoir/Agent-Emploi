"""L'état civil : chargement, gabarit, et ce qui manque."""

from __future__ import annotations

import pytest
import yaml

from agent_emploi.apply.fields import Slot
from agent_emploi.apply.identity import Identity


def test_chargement(tmp_path):
    path = tmp_path / "identity.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "first_name": "Killian",
                "last_name": "Scordel",
                "email": "killian@exemple.fr",
                "phone": "+33612345678",
                "linkedin": "https://linkedin.com/in/kscordel",
            }
        ),
        encoding="utf-8",
    )

    identity = Identity.load(path)

    assert identity.full_name == "Killian Scordel"
    assert identity.values()[Slot.EMAIL] == "killian@exemple.fr"
    assert identity.missing == []


def test_les_valeurs_vides_sont_omises_pas_rendues_vides(tmp_path):
    """Un champ sans valeur ne doit pas effacer ce que le site a pré-rempli."""
    path = tmp_path / "identity.yaml"
    path.write_text("first_name: Killian\nphone:\n", encoding="utf-8")

    values = Identity.load(path).values()

    assert Slot.PHONE not in values
    assert values[Slot.FIRST_NAME] == "Killian"


def test_fichier_absent_indique_la_commande(tmp_path):
    with pytest.raises(FileNotFoundError, match="--init-identity"):
        Identity.load(tmp_path / "absent.yaml")


def test_gabarit_ecrit_puis_protege(tmp_path):
    path = tmp_path / "profile" / "identity.yaml"

    Identity.write_template(path)
    assert "first_name" in path.read_text(encoding="utf-8")
    # Le gabarit rempli ne doit pas pouvoir être écrasé par un second appel.
    with pytest.raises(FileExistsError):
        Identity.write_template(path)


def test_gabarit_vide_se_charge_et_signale_ce_qui_manque(tmp_path):
    path = tmp_path / "identity.yaml"
    Identity.write_template(path)

    identity = Identity.load(path)

    assert identity.values() == {}
    assert identity.missing == ["first_name", "last_name", "email", "phone"]
