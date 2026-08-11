"""Le recours d'appariement : ce qu'il rattrape, et ce qu'il ne peut pas casser.

L'enjeu de ces tests n'est pas la qualité des réponses du modèle — elle
échappe au code — mais le fait qu'une réponse fantaisiste ne puisse jamais
produire autre chose qu'un champ vide ou une case laissée à l'utilisateur.
"""

from __future__ import annotations

import json

import pytest

from agent_emploi.apply.fields import FormField, Slot, build_plan
from agent_emploi.apply.mapping import eligible_fields, guess_slots
from agent_emploi.llm.budget import BudgetExceeded
from agent_emploi.llm.router import LlmError


class FakeRouter:
    """Routeur scripté : rend une réponse, ou lève ce qu'on lui demande."""

    def __init__(self, guesses=None, raises=None):
        self._guesses = guesses
        self._raises = raises
        self.prompts: list[str] = []

    def structured(self, task, prompt, schema, **kwargs):
        self.prompts.append(prompt)
        if self._raises is not None:
            raise self._raises
        return schema.model_validate(json.loads(json.dumps({"guesses": self._guesses})))


def field(selector, label, kind="text", **kwargs):
    return FormField(selector=selector, kind=kind, label=label, **kwargs)


VALUES = {
    Slot.FIRST_NAME: "Killian",
    Slot.EMAIL: "killian@exemple.fr",
    Slot.PHONE: "+33612345678",
}


def test_seuls_les_champs_non_reconnus_sont_soumis():
    """Le déterministe l'emporte : on ne paie pas pour reconnaître « Prénom »."""
    fields = [
        field("s1", "Prénom"),
        field("s2", "Comment vous joindre ?"),
        field("s3", "CV", kind="file"),
        field("s4", "J'accepte", kind="checkbox"),
    ]

    assert [f.selector for f in eligible_fields(fields)] == ["s2"]


def test_appariement_rattrape_un_libelle_inattendu():
    router = FakeRouter(guesses=[{"index": 0, "slot": "email"}])
    fields = [field("s2", "Comment devons-nous vous joindre ?", required=True)]

    overrides = guess_slots(router, fields, [Slot.EMAIL, Slot.PHONE])

    assert overrides == {"s2": Slot.EMAIL}
    plan = build_plan(fields, VALUES, overrides)
    assert plan.ok
    assert plan.assignments[0].value == "killian@exemple.fr"


def test_le_prompt_ne_contient_aucune_valeur_personnelle():
    """Le modèle désigne un emplacement : il n'a pas à voir les données."""
    router = FakeRouter(guesses=[])
    fields = [field("s2", "Comment vous joindre ?")]

    guess_slots(router, fields, list(VALUES))

    prompt = router.prompts[0]
    for value in VALUES.values():
        assert value not in prompt


def test_reponse_unknown_laisse_le_champ_a_lutilisateur():
    router = FakeRouter(guesses=[{"index": 0, "slot": "unknown"}])
    fields = [field("s2", "Votre plat préféré", required=True)]

    overrides = guess_slots(router, fields, list(VALUES))

    assert overrides == {}
    assert not build_plan(fields, VALUES, overrides).ok


@pytest.mark.parametrize(
    "guesses",
    [
        [{"index": 9, "slot": "email"}],  # numéro hors bornes
        [{"index": 0, "slot": "numero_de_securite_sociale"}],  # emplacement inventé
        [{"index": 0, "slot": "cv"}],  # emplacement non proposé
    ],
)
def test_les_reponses_fantaisistes_sont_ignorees(guesses):
    router = FakeRouter(guesses=guesses)
    fields = [field("s2", "Votre plat préféré")]

    assert guess_slots(router, fields, [Slot.EMAIL, Slot.PHONE]) == {}


def test_un_emplacement_nest_pas_attribue_deux_fois():
    router = FakeRouter(
        guesses=[{"index": 0, "slot": "email"}, {"index": 1, "slot": "email"}]
    )
    fields = [field("s1", "Un champ"), field("s2", "Un autre champ")]

    assert guess_slots(router, fields, [Slot.EMAIL]) == {"s1": Slot.EMAIL}


def test_les_champs_declaratifs_echappent_au_modele():
    """Même appariée par le modèle, une case ne se coche pas toute seule."""
    fields = [field("s1", "Disponible immédiatement ?", kind="checkbox", required=True)]

    assert guess_slots(FakeRouter(guesses=[]), fields, list(VALUES)) == {}
    # Et si un appariement lui parvenait tout de même, `build_plan` l'écarte.
    plan = build_plan(fields, VALUES, {"s1": Slot.EMAIL})
    assert plan.assignments == []
    assert plan.todo == ["Disponible immédiatement ?"]


def test_le_modele_ne_reattribue_pas_un_champ_deja_reconnu():
    """Un motif qui a mordu fait foi, quoi que dise le modèle."""
    fields = [field("s1", "Prénom")]

    plan = build_plan(fields, VALUES, {"s1": Slot.EMAIL})

    assert plan.assignments[0].value == "Killian"


@pytest.mark.parametrize(
    "exception",
    [
        LlmError("route en panne"),
        BudgetExceeded("plafond atteint"),
        KeyError("tâche LLM inconnue: 'form_mapping'"),
    ],
)
def test_un_echec_dappel_ne_leve_pas(exception):
    """Sans clé, sans budget, sans réseau ou sans route déclarée, rien ne casse."""
    router = FakeRouter(raises=exception)

    assert guess_slots(router, [field("s2", "Un champ")], list(VALUES)) == {}


def test_aucun_appel_sans_champ_a_apparier():
    router = FakeRouter(guesses=[])

    assert guess_slots(router, [field("s1", "Prénom")], list(VALUES)) == {}
    assert router.prompts == []
