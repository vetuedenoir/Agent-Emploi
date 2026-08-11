"""Appariement des champs d'un formulaire — du Python pur, sans navigateur.

Un formulaire de candidature est décrit ici par une liste de `FormField`
(libellé, nom, type, obligatoire). Savoir que le champ « Prénom » attend le
prénom ne demande pas un modèle de langage : c'est de la reconnaissance de
libellés, et un tableau de motifs le fait mieux — de façon reproductible, sans
appel réseau, et sans risque qu'un modèle décide de mettre l'adresse dans la
case du téléphone.

Le module ne connaît ni Playwright ni le disque : il transforme une liste de
champs en un plan de remplissage. C'est ce qui le rend testable, et c'est là
que se trouve toute la logique délicate de l'étape 6.

Deux règles de prudence :

- **Un champ jamais apparié n'est jamais rempli au hasard.** S'il est
  obligatoire et attend du texte ou un fichier, la candidature part en
  `handoff` : mieux vaut rendre la main que remplir de travers.
- **Les cases à cocher et les listes déroulantes ne sont pas cochées**, même
  reconnues. Consentement RGPD, disponibilité, autorisation de travail : ce
  sont des déclarations, elles appartiennent à l'utilisateur, qui est de toute
  façon devant l'écran.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from agent_emploi.profile import fold


class Slot(StrEnum):
    """Ce qu'un champ attend, indépendamment de son libellé et du site."""

    FIRST_NAME = "first_name"
    LAST_NAME = "last_name"
    FULL_NAME = "full_name"
    EMAIL = "email"
    PHONE = "phone"
    LOCATION = "location"
    LINKEDIN = "linkedin"
    GITHUB = "github"
    PORTFOLIO = "portfolio"
    #: Le CV en pièce jointe — c'est la copie du dossier `outbox/` qui part.
    CV = "cv"
    #: La lettre collée dans une zone de texte.
    COVER_LETTER = "cover_letter"
    #: La lettre en pièce jointe, quand le site n'offre pas de zone de texte.
    COVER_LETTER_FILE = "cover_letter_file"


#: Types de champ qui acceptent du texte libre.
TEXTUAL = frozenset({"text", "email", "tel", "url", "search", "textarea", "number"})

#: Types que l'on ne remplit jamais automatiquement : ce sont des déclarations
#: ou des choix, pas des données du profil.
DECLARATIVE = frozenset({"checkbox", "radio", "select"})


@dataclass(frozen=True)
class FormField:
    """Un champ de formulaire, tel que le navigateur l'a décrit.

    `selector` est opaque pour ce module : il ne sert qu'à redonner la main au
    navigateur au moment de remplir.
    """

    selector: str
    kind: str
    name: str = ""
    field_id: str = ""
    label: str = ""
    placeholder: str = ""
    autocomplete: str = ""
    required: bool = False
    #: Valeur déjà présente : un champ pré-rempli par le site n'est pas écrasé.
    value: str = ""
    options: list[str] = field(default_factory=list)

    @property
    def haystack(self) -> str:
        """Tout ce qui peut désigner le champ, replié pour la comparaison."""
        return fold(
            " ".join(
                (self.label, self.name, self.field_id, self.placeholder, self.autocomplete)
            )
        )

    @property
    def display(self) -> str:
        """Le nom du champ tel qu'affiché à l'utilisateur dans les rapports."""
        return self.label.strip() or self.name or self.field_id or self.selector


#: Motifs, du plus spécifique au plus général — l'ordre fait la décision.
#: « prenom » contient « nom » : le prénom doit donc être examiné avant le nom,
#: et le nom complet en dernier, quand ni l'un ni l'autre n'a mordu.
_TEXT_PATTERNS: tuple[tuple[Slot, tuple[str, ...]], ...] = (
    (Slot.FIRST_NAME, ("prenom", "first name", "firstname", "first_name", "given name")),
    (
        Slot.LAST_NAME,
        ("nom de famille", "last name", "lastname", "last_name", "surname", "family name"),
    ),
    (Slot.EMAIL, ("email", "e-mail", "courriel", "mail")),
    (
        Slot.PHONE,
        ("telephone", "phone", "portable", "mobile", "tel ", "numero de tel"),
    ),
    (Slot.LINKEDIN, ("linkedin",)),
    (Slot.GITHUB, ("github", "gitlab")),
    (
        Slot.PORTFOLIO,
        ("portfolio", "site web", "website", "personal site", "site personnel"),
    ),
    (
        Slot.LOCATION,
        ("ville", "city", "localisation", "location", "adresse", "address", "code postal"),
    ),
    (
        Slot.COVER_LETTER,
        (
            "lettre de motivation",
            "cover letter",
            "coverletter",
            "cover_letter",
            "motivation",
            "message",
            "pourquoi nous",
            "why do you want",
        ),
    ),
    (Slot.FULL_NAME, ("nom complet", "full name", "fullname", "votre nom", "name", "nom")),
)

#: Motifs des champs de type fichier. La lettre est examinée avant le CV : un
#: libellé « lettre de motivation (PDF) » ne doit pas partir dans la case CV.
_FILE_PATTERNS: tuple[tuple[Slot, tuple[str, ...]], ...] = (
    (
        Slot.COVER_LETTER_FILE,
        ("lettre de motivation", "cover letter", "coverletter", "cover_letter", "motivation"),
    ),
    (Slot.CV, ("cv", "curriculum", "resume", "resume/cv", "document")),
)


def _first_match(haystack: str, patterns) -> Slot | None:
    for slot, needles in patterns:
        if any(needle in haystack for needle in needles):
            return slot
    return None


def match(field_: FormField) -> Slot | None:
    """L'emplacement attendu par un champ, ou `None` si on ne sait pas.

    Le type prime sur le libellé : une zone de texte intitulée « CV » attend un
    texte, pas un fichier, et un champ fichier intitulé « votre message » reste
    un fichier. Se tromper de sens produirait un formulaire cassé plutôt qu'un
    formulaire incomplet.
    """
    if field_.kind == "file":
        return _first_match(field_.haystack, _FILE_PATTERNS)
    if field_.kind in DECLARATIVE:
        # Reconnu ou non, on ne coche ni ne choisit à la place de l'utilisateur.
        return None
    if field_.kind not in TEXTUAL:
        return None

    haystack = field_.haystack
    slot = _first_match(haystack, _TEXT_PATTERNS)
    if slot is None and field_.kind == "textarea":
        # Une zone de texte sans libellé reconnu sur un formulaire de
        # candidature est presque toujours la lettre — mais « presque » ne
        # suffit pas pour y coller 200 mots : on la laisse à l'utilisateur.
        return None
    if slot is Slot.COVER_LETTER and field_.kind not in ("textarea", "text"):
        return None
    return slot


@dataclass
class Assignment:
    """Un champ, sa valeur, et pourquoi elle a été retenue."""

    field: FormField
    slot: Slot
    value: str

    @property
    def is_file(self) -> bool:
        return self.field.kind == "file"


@dataclass
class Plan:
    """Le plan de remplissage d'un formulaire : ce qui part, ce qui reste.

    `blocking` est ce qui décide de l'issue : une liste non vide envoie la
    candidature en `handoff`. `todo` n'est qu'une liste de rappels — le
    formulaire est utilisable, il reste des cases à cocher.
    """

    assignments: list[Assignment] = field(default_factory=list)
    #: Champs laissés à l'utilisateur : cases, listes, questions libres.
    todo: list[str] = field(default_factory=list)
    #: Champs requis qu'on ne sait pas remplir : motif de rendu de la main.
    blocking: list[str] = field(default_factory=list)
    #: Champs ignorés parce que le site les a déjà remplis.
    prefilled: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.blocking


def build_plan(fields: list[FormField], values: dict[Slot, str]) -> Plan:
    """Apparie les champs aux valeurs disponibles et classe le reste.

    `values` porte ce que l'on sait poser : l'état civil, le chemin du CV et le
    texte de la lettre. Un emplacement reconnu mais sans valeur n'est pas une
    erreur silencieuse — il finit dans `todo`, ou dans `blocking` si le champ
    est obligatoire.
    """
    plan = Plan()

    for field_ in fields:
        slot = match(field_)

        if slot is None:
            label = field_.display
            if field_.kind in DECLARATIVE or not field_.required:
                plan.todo.append(label)
            else:
                plan.blocking.append(f"champ obligatoire non reconnu : {label}")
            continue

        value = values.get(slot, "")
        if not value:
            label = f"{field_.display} ({slot.value})"
            if field_.required:
                plan.blocking.append(f"champ obligatoire sans valeur connue : {label}")
            else:
                plan.todo.append(label)
            continue

        if field_.value.strip() and field_.kind != "file":
            # Le site a déjà rempli le champ (session connectée, brouillon) :
            # sa valeur peut être plus juste que la nôtre, on n'écrase pas.
            plan.prefilled.append(field_.display)
            continue

        plan.assignments.append(Assignment(field=field_, slot=slot, value=value))

    return plan
