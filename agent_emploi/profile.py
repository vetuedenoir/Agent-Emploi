"""Le profil de l'utilisateur : CV, voix, formules interdites.

Tout ce que la rédaction consomme en entrée est réuni ici, chargé une fois par
passe. Deux raisons : ces fichiers forment le préfixe stable du prompt de
rédaction (celui que le cache de l'API réutilise d'une offre à l'autre), et
leur absence doit échouer *avant* le premier appel payant, pas au milieu.

La sélection du CV et la détection des formules interdites sont du Python pur —
aucun appel LLM. Ce sont deux vérifications que le modèle ferait mal et qui
doivent être reproductibles.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from agent_emploi.config import Config, ProfileConfig

logger = logging.getLogger(__name__)

#: Marqueur laissé dans les gabarits de `profile/` (voir `cli._is_template`).
TEMPLATE_MARKER = "À compléter"

#: Commentaires HTML du gabarit : ce sont des consignes adressées à
#: l'utilisateur, pas à un modèle. Les retirer allège le prompt et évite que le
#: rédacteur prenne « Décris ton registre » pour une instruction.
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)

#: Blocs entre triples accents graves : c'est là que le gabarit demande à
#: l'utilisateur d'écrire, et donc là que l'on vérifie qu'il a écrit.
_FENCED = re.compile(r"^```[^\n]*\n(.*?)^```", re.MULTILINE | re.DOTALL)


def clean_voice(text: str) -> str:
    """Retire les consignes du gabarit, garde ce que l'utilisateur a écrit."""
    stripped = _HTML_COMMENT.sub("", text)
    lines = [
        line
        for line in stripped.splitlines()
        if not line.lstrip().startswith(">") and line.strip() != f"{TEMPLATE_MARKER}."
    ]
    return "\n".join(lines).strip()


def voice_is_filled(text: str) -> bool:
    """Vrai si au moins un bloc du gabarit contient un échantillon d'écriture.

    Chercher `TEMPLATE_MARKER` ne suffit pas : le fichier est rempli section par
    section, et une ligne de gabarit oubliée en bas ne doit pas faire passer
    pour vide un fichier qui contient déjà trois échantillons.
    """
    return any(block.strip() for block in _FENCED.findall(text))


def fold(text: str) -> str:
    """Minuscules, sans accents, apostrophes et espaces uniformisés.

    Les formules interdites sont écrites au propre dans `banned_phrases.txt`,
    alors que le modèle produit indifféremment « n'hésitez » et « n’hésitez »,
    parfois coupés par un retour à la ligne. La comparaison se fait donc sur
    cette forme repliée, des deux côtés.
    """
    decomposed = unicodedata.normalize("NFKD", text.lower())
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    for quote in "’‘‛`´":
        stripped = stripped.replace(quote, "'")
    return " ".join(stripped.split())


class BannedPhrases:
    """Liste noire de formules, vérifiée après chaque génération.

    Un contrôle déterministe et gratuit : demander au modèle de ne pas écrire
    « fort de mon expérience » ne suffit pas, le vérifier après coup si.
    """

    def __init__(self, phrases: list[str]) -> None:
        #: (forme repliée pour la recherche, forme d'origine pour l'affichage)
        self._phrases = [(fold(p), p) for p in phrases if fold(p)]

    def __len__(self) -> int:
        return len(self._phrases)

    def __iter__(self):
        return iter(phrase for _, phrase in self._phrases)

    @classmethod
    def load(cls, path: Path) -> BannedPhrases:
        """Lit le fichier ; lignes vides et commentaires `#` ignorés.

        Un fichier absent donne une liste vide plutôt qu'une erreur : la liste
        noire durcit la sortie, elle n'est pas indispensable au fonctionnement.
        """
        if not Path(path).exists():
            logger.warning("liste de formules interdites absente: %s", path)
            return cls([])
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        return cls(
            [
                line.strip()
                for line in lines
                if line.strip() and not line.strip().startswith("#")
            ]
        )

    def find(self, text: str) -> list[str]:
        """Formules interdites présentes dans le texte, dans l'ordre du fichier."""
        folded = fold(text)
        return [phrase for needle, phrase in self._phrases if needle in folded]


@dataclass(frozen=True)
class Profile:
    """Le profil chargé, prêt à être injecté dans les prompts."""

    cv_text: str
    voice: str
    banned: BannedPhrases
    cv_fr: Path
    cv_en: Path

    @classmethod
    def load(cls, config: Config | ProfileConfig) -> Profile:
        """Charge le profil ; lève `FileNotFoundError` si le CV texte manque.

        Le CV texte est le seul fichier réellement bloquant : sans lui, la
        lettre n'aurait rien de concret à citer. L'absence de `voice.md` est un
        avertissement, pas une erreur — mais elle se paie en qualité, et c'est
        la raison la plus fréquente d'une lettre au style générique.
        """
        profile = config.profile if isinstance(config, Config) else config

        if not profile.cv_markdown.exists():
            raise FileNotFoundError(
                f"CV texte introuvable: {profile.cv_markdown} — requis pour rédiger"
            )
        cv_text = profile.cv_markdown.read_text(encoding="utf-8").strip()

        voice = ""
        if profile.voice.exists():
            voice = clean_voice(profile.voice.read_text(encoding="utf-8"))
        if not voice:
            logger.warning(
                "%s absent ou vide — les lettres sonneront génériques", profile.voice
            )

        return cls(
            cv_text=cv_text,
            voice=voice,
            banned=BannedPhrases.load(profile.banned_phrases),
            cv_fr=profile.cv_fr,
            cv_en=profile.cv_en,
        )

    @property
    def has_voice(self) -> bool:
        """Vrai si `voice.md` porte au moins un échantillon d'écriture."""
        return voice_is_filled(self.voice)

    def cv_pdf(self, language: str) -> Path | None:
        """CV à joindre pour la langue de l'annonce, ou `None` s'il manque.

        Pur Python : la langue vient déjà du fit-check, un appel LLM dédié pour
        choisir entre deux fichiers serait du gaspillage.
        """
        path = self.cv_en if language == "en" else self.cv_fr
        if not path.exists():
            logger.warning("CV %s introuvable: %s", language, path)
            return None
        return path
