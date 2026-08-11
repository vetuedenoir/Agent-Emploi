"""Dernier recours quand un libellé n'est pas reconnu — tâche `form_mapping`.

Les motifs de `fields.py` couvrent les formulaires courants ; ils ne couvriront
jamais « Comment devons-nous vous appeler ? ». Un appel LLM du tier gratuit
tranche ces cas, avec un rôle strictement borné :

**le modèle choisit un emplacement, jamais une valeur.** Il répond « ce champ
attend l'adresse électronique » ; c'est le code qui va chercher l'adresse dans
`identity.yaml` et qui la pose. Un modèle qui hallucine ne peut donc pas
inventer un numéro de téléphone — au pire il se trompe de case, et une case mal
remplie se voit dans le rapport comme dans le navigateur, que vous relisez
avant d'envoyer.

Deux catégories de champs ne passent jamais par ici :

- les **fichiers**, dont l'appariement est déjà tranché par le type ;
- les **champs déclaratifs** (cases, listes, boutons radio), qui ne sont
  remplis automatiquement en aucune circonstance.

Ce recours est une commodité, pas une dépendance : sans clé d'API, sans budget
ou sans réseau, la passe se déroule exactement comme avant — les champs
inconnus repartent en `handoff`.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from agent_emploi.apply.fields import DECLARATIVE, FormField, Slot, match
from agent_emploi.llm.budget import BudgetExceeded
from agent_emploi.llm.router import LlmError, Router

logger = logging.getLogger(__name__)

TASK = "form_mapping"

#: Au-delà, ce n'est plus un formulaire de candidature mais une page de
#: recherche ou un tableau de bord : on n'engage pas l'appel.
MAX_FIELDS = 25

SYSTEM = (
    "Tu apparies les champs d'un formulaire de candidature à une liste "
    "d'informations connues sur le candidat. Tu ne produis aucune valeur : tu "
    "dis seulement quelle information chaque champ attend. Dans le doute, "
    'réponds "unknown" — un champ non apparié sera rempli à la main par le '
    "candidat, ce qui est sans conséquence, alors qu'un champ mal apparié "
    "envoie une donnée fausse à un recruteur."
)

#: Ce que chaque emplacement désigne, pour que le modèle n'ait pas à deviner
#: le sens d'un identifiant technique.
_DESCRIPTIONS: dict[Slot, str] = {
    Slot.FIRST_NAME: "le prénom du candidat",
    Slot.LAST_NAME: "le nom de famille du candidat",
    Slot.FULL_NAME: "le nom complet (prénom et nom)",
    Slot.EMAIL: "l'adresse électronique",
    Slot.PHONE: "le numéro de téléphone",
    Slot.LOCATION: "la ville ou la région où habite le candidat",
    Slot.LINKEDIN: "l'adresse du profil LinkedIn",
    Slot.GITHUB: "l'adresse du profil GitHub",
    Slot.PORTFOLIO: "l'adresse d'un site personnel ou portfolio",
    Slot.COVER_LETTER: "le texte de la lettre de motivation",
}

#: Réponse du modèle pour un champ qu'il n'a pas su rattacher.
UNKNOWN = "unknown"


class FieldGuess(BaseModel):
    """L'emplacement supposé d'un champ, désigné par son numéro."""

    index: int
    slot: str


class Mapping(BaseModel):
    guesses: list[FieldGuess] = Field(default_factory=list)


def eligible_fields(fields: list[FormField]) -> list[FormField]:
    """Champs qu'un appariement LLM a le droit de trancher.

    Un champ déjà reconnu par les motifs n'est pas soumis au modèle : le
    déterministe l'emporte toujours, il est gratuit et reproductible.
    """
    return [
        field_
        for field_ in fields
        if field_.kind not in DECLARATIVE
        and field_.kind != "file"
        and match(field_) is None
    ]


def build_prompt(fields: list[FormField], available: list[Slot]) -> str:
    """Décrit les champs non reconnus et les informations encore disponibles."""
    lines = ["# Informations connues sur le candidat", ""]
    lines += [
        f"- `{slot.value}` : {_DESCRIPTIONS.get(slot, slot.value)}"
        for slot in available
    ]
    lines += [
        f"- `{UNKNOWN}` : aucune des informations ci-dessus ne convient",
        "",
        "# Champs du formulaire, non reconnus automatiquement",
        "",
    ]
    for index, field_ in enumerate(fields):
        details = [f"type={field_.kind}"]
        if field_.required:
            details.append("obligatoire")
        if field_.placeholder:
            details.append(f"exemple={field_.placeholder!r}")
        lines.append(f"{index}. « {field_.display} » ({', '.join(details)})")

    lines += [
        "",
        "Pour chaque champ, donne son numéro et l'information attendue, en "
        "reprenant exactement l'un des identifiants ci-dessus. N'invente aucune "
        "valeur : tu ne fais que désigner.",
    ]
    return "\n".join(lines)


def guess_slots(
    router: Router,
    fields: list[FormField],
    available: list[Slot],
    *,
    job_id: str | None = None,
) -> dict[str, Slot]:
    """Apparie les champs non reconnus ; retourne `{sélecteur: emplacement}`.

    Ne lève jamais : un échec d'appel, un plafond de budget atteint, une route
    absente de `config.yaml` ou une sortie invalide donnent un appariement
    vide, et les champs concernés suivent le chemin normal — à la main, ou en
    `handoff` s'ils sont obligatoires.
    """
    candidates = eligible_fields(fields)
    if not candidates or not available:
        return {}
    if len(candidates) > MAX_FIELDS:
        logger.info(
            "%d champs non reconnus — appariement LLM abandonné", len(candidates)
        )
        return {}

    try:
        mapping = router.structured(
            TASK,
            build_prompt(candidates, available),
            Mapping,
            system=SYSTEM,
            job_id=job_id,
        )
    except (LlmError, BudgetExceeded, KeyError) as exc:
        # `KeyError` : la tâche n'est pas déclarée dans `config.yaml`. C'est un
        # choix de configuration légitime — s'en passer, c'est revenir au
        # comportement purement déterministe.
        logger.warning("appariement LLM indisponible: %s", exc)
        return {}

    known = {slot.value: slot for slot in available}
    resolved: dict[str, Slot] = {}
    taken: set[Slot] = set()
    for guess in mapping.guesses:
        slot = known.get(guess.slot.strip().lower())
        if slot is None or not 0 <= guess.index < len(candidates):
            # `unknown`, emplacement inventé, numéro hors bornes : le modèle a
            # le droit de ne pas savoir, on ne le rattrape pas.
            continue
        if slot in taken:
            # Deux champs pour la même information : le second est sans doute
            # une question différente que le modèle a mal comprise.
            logger.info("emplacement %s proposé deux fois — second ignoré", slot.value)
            continue
        taken.add(slot)
        resolved[candidates[guess.index].selector] = slot

    if resolved:
        logger.info("%d champ(s) appariés par le modèle", len(resolved))
    return resolved
