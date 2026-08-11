"""Appariement des champs : le cœur déterministe de l'étape 6, sans navigateur."""

from __future__ import annotations

import pytest

from agent_emploi.apply.fields import FormField, Slot, build_plan, match


def field(**kwargs) -> FormField:
    kwargs.setdefault("selector", "[data-ae-field='0']")
    kwargs.setdefault("kind", "text")
    return FormField(**kwargs)


@pytest.mark.parametrize(
    "label, expected",
    [
        ("Prénom", Slot.FIRST_NAME),
        ("First name", Slot.FIRST_NAME),
        ("Nom de famille", Slot.LAST_NAME),
        ("Last name *", Slot.LAST_NAME),
        ("Adresse e-mail", Slot.EMAIL),
        ("Téléphone", Slot.PHONE),
        ("Profil LinkedIn", Slot.LINKEDIN),
        ("GitHub", Slot.GITHUB),
        ("Ville", Slot.LOCATION),
        ("Nom complet", Slot.FULL_NAME),
        ("Votre couleur préférée", None),
    ],
)
def test_libelles_courants(label, expected):
    assert match(field(label=label)) is expected


def test_prenom_avant_nom():
    """« prenom » contient « nom » : l'ordre des motifs doit trancher."""
    assert match(field(label="Prénom")) is Slot.FIRST_NAME
    assert match(field(name="firstname")) is Slot.FIRST_NAME


def test_appariement_par_attribut_quand_le_libelle_manque():
    """Les ATS livrent souvent des champs sans `label`, mais avec un `name`."""
    assert match(field(name="job_application[email]")) is Slot.EMAIL
    assert match(field(field_id="last_name")) is Slot.LAST_NAME
    assert match(field(placeholder="+33 6 12 34 56 78", name="x")) is None


def test_le_type_prime_sur_le_libelle():
    """Un champ fichier intitulé « lettre » reste un fichier, et l'inverse."""
    assert match(field(kind="file", label="Lettre de motivation")) is Slot.COVER_LETTER_FILE
    assert match(field(kind="textarea", label="Lettre de motivation")) is Slot.COVER_LETTER
    assert match(field(kind="file", label="CV")) is Slot.CV


def test_les_champs_declaratifs_ne_sont_jamais_apparies():
    """Cocher une déclaration à la place de l'utilisateur est hors de question."""
    assert match(field(kind="checkbox", label="J'accepte la politique RGPD")) is None
    assert match(field(kind="select", label="Ville")) is None
    assert match(field(kind="radio", label="Email")) is None


def test_textarea_sans_libelle_reconnu_reste_a_lutilisateur():
    assert match(field(kind="textarea", label="Une question de notre part")) is None


def test_plan_remplit_ce_quil_connait():
    fields = [
        field(selector="s1", label="Prénom"),
        field(selector="s2", label="Nom de famille"),
        field(selector="s3", kind="file", label="CV"),
    ]
    plan = build_plan(
        fields,
        {Slot.FIRST_NAME: "Killian", Slot.LAST_NAME: "Scordel", Slot.CV: "/tmp/cv.pdf"},
    )

    assert plan.ok
    assert [a.value for a in plan.assignments] == ["Killian", "Scordel", "/tmp/cv.pdf"]
    assert plan.assignments[2].is_file


def test_champ_obligatoire_inconnu_bloque():
    fields = [field(label="Votre plat préféré", required=True)]
    plan = build_plan(fields, {})

    assert not plan.ok
    assert "Votre plat préféré" in plan.blocking[0]


def test_champ_facultatif_inconnu_ne_bloque_pas():
    plan = build_plan([field(label="Votre plat préféré")], {})

    assert plan.ok
    assert plan.todo == ["Votre plat préféré"]


def test_champ_reconnu_sans_valeur_bloque_sil_est_obligatoire():
    """Reconnaître « Téléphone » sans connaître le numéro n'aide personne."""
    plan = build_plan([field(label="Téléphone", required=True)], {})

    assert not plan.ok
    assert "sans valeur connue" in plan.blocking[0]


def test_case_a_cocher_obligatoire_ne_bloque_pas_mais_est_rappelee():
    """L'utilisateur est devant l'écran : une case à cocher est un rappel."""
    fields = [field(kind="checkbox", label="J'accepte les conditions", required=True)]
    plan = build_plan(fields, {})

    assert plan.ok
    assert plan.todo == ["J'accepte les conditions"]


def test_valeur_deja_presente_non_ecrasee():
    """Un site qui pré-remplit en sait peut-être plus que nous."""
    fields = [field(label="Email", value="deja@rempli.fr")]
    plan = build_plan(fields, {Slot.EMAIL: "moi@exemple.fr"})

    assert plan.assignments == []
    assert plan.prefilled == ["Email"]
