"""L'état civil du candidat — `profile/identity.yaml`.

Les étapes 1 à 5 n'en ont pas besoin : une lettre se rédige à partir du CV, pas
d'un numéro de téléphone. C'est le remplissage de formulaire qui exige ces
valeurs, et il vaut mieux qu'elles vivent dans un fichier que l'utilisateur
relit d'un coup d'œil plutôt que disséminées dans `config.yaml`.

Le fichier est dans `profile/`, hors dépôt : ce sont des données personnelles.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from agent_emploi.apply.fields import Slot

logger = logging.getLogger(__name__)

#: Gabarit écrit lorsque le fichier manque : l'utilisateur n'a plus qu'à
#: remplir. Les clés vides sont tolérées — un champ sans valeur est simplement
#: laissé à l'utilisateur dans le navigateur.
TEMPLATE = """\
# État civil et liens, utilisés pour remplir les formulaires de candidature.
# Ce fichier ne quitte jamais votre machine. Une valeur vide n'est pas une
# erreur : le champ correspondant sera simplement laissé à remplir à la main.

first_name: ""
last_name: ""
email: ""
phone: ""
# Ville ou région, telle que vous l'écririez dans un formulaire.
location: ""
linkedin: ""
github: ""
portfolio: ""
"""


class Identity(BaseModel):
    """Les valeurs que le remplissage peut poser dans un formulaire.

    Aucun champ n'est obligatoire : une valeur absente n'empêche pas la passe,
    elle apparaît dans les « à faire » du rapport. Ce qui bloquerait, c'est de
    poser une valeur fausse.
    """

    first_name: str = ""
    last_name: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    linkedin: str = ""
    github: str = ""
    portfolio: str = ""
    #: Valeurs supplémentaires, appariées par libellé si le besoin s'en fait
    #: sentir. Volontairement en marge : le tronc commun est au-dessus.
    extra: dict[str, str] = Field(default_factory=dict)

    @property
    def full_name(self) -> str:
        return " ".join(part for part in (self.first_name, self.last_name) if part)

    @classmethod
    def load(cls, path: Path) -> Identity:
        """Charge le fichier ; lève `FileNotFoundError` avec le gabarit en main.

        Le message porte le chemin attendu : c'est la seule erreur de
        configuration propre à l'étape 6, autant qu'elle se corrige sans ouvrir
        la documentation.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"état civil introuvable: {path} — créez-le avec "
                f"`python -m agent_emploi apply --init-identity`"
            )
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path}: un dictionnaire de clés est attendu")
        return cls.model_validate({k: v if v is not None else "" for k, v in data.items()})

    @classmethod
    def write_template(cls, path: Path) -> Path:
        """Dépose le gabarit, sans jamais écraser un fichier existant."""
        path = Path(path)
        if path.exists():
            raise FileExistsError(f"{path} existe déjà — il n'a pas été touché")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(TEMPLATE, encoding="utf-8")
        return path

    def values(self) -> dict[Slot, str]:
        """Les valeurs disponibles, indexées par emplacement de formulaire.

        Les emplacements sans valeur sont omis plutôt que rendus vides : le
        remplissage doit pouvoir distinguer « je sais quoi mettre » de « je ne
        sais pas », et ne jamais effacer un champ déjà pré-rempli par le site.
        """
        candidates = {
            Slot.FIRST_NAME: self.first_name,
            Slot.LAST_NAME: self.last_name,
            Slot.FULL_NAME: self.full_name,
            Slot.EMAIL: self.email,
            Slot.PHONE: self.phone,
            Slot.LOCATION: self.location,
            Slot.LINKEDIN: self.linkedin,
            Slot.GITHUB: self.github,
            Slot.PORTFOLIO: self.portfolio,
        }
        return {slot: value.strip() for slot, value in candidates.items() if value.strip()}

    @property
    def missing(self) -> list[str]:
        """Champs du tronc commun laissés vides — signalés par `doctor`."""
        return [
            name
            for name in ("first_name", "last_name", "email", "phone")
            if not getattr(self, name).strip()
        ]
