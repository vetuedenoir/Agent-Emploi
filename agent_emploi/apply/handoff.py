"""Rendre la main proprement — `candidature.md` dans le dossier de l'offre.

Le remplissage automatique échouera régulièrement : captcha, connexion,
formulaire maison, question inattendue. Ce n'est pas un incident, c'est le
fonctionnement normal d'un système qui refuse de deviner. Ce qui serait un
incident, c'est de laisser l'utilisateur devant un formulaire à moitié rempli
sans lui dire ce qui reste à faire.

Ce module écrit donc, dans le dossier `outbox/` déjà validé, une fiche qui
contient tout ce qu'il faut pour finir à la main : l'URL directe, les valeurs à
recopier, ce qui a été rempli, ce qui ne l'a pas été et pourquoi.
"""

from __future__ import annotations

from pathlib import Path

from agent_emploi.apply.fields import Plan, Slot
from agent_emploi.models import ApplyOutcome, JobRecord

#: Nom de la fiche déposée dans le dossier de l'offre.
HANDOFF_FILE = "candidature.md"

#: Nom de la capture d'écran du formulaire rempli.
SCREENSHOT_FILE = "formulaire.png"

#: Libellés lisibles des emplacements, pour la liste à recopier.
_LABELS: dict[Slot, str] = {
    Slot.FIRST_NAME: "Prénom",
    Slot.LAST_NAME: "Nom",
    Slot.FULL_NAME: "Nom complet",
    Slot.EMAIL: "Email",
    Slot.PHONE: "Téléphone",
    Slot.LOCATION: "Localisation",
    Slot.LINKEDIN: "LinkedIn",
    Slot.GITHUB: "GitHub",
    Slot.PORTFOLIO: "Portfolio",
}


def write_handoff(
    directory: Path,
    record: JobRecord,
    outcome: ApplyOutcome,
    *,
    plan: Plan | None = None,
    values: dict[Slot, str] | None = None,
) -> Path:
    """Écrit la fiche de reprise à la main et retourne son chemin.

    Elle est produite dans les deux cas, `prefilled` comme `handoff` : même
    quand le formulaire est rempli, l'utilisateur peut vouloir la liste de ce
    qui reste à cocher, ou recopier une valeur ailleurs.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    job = record.job

    prefilled = outcome.status == "prefilled"
    lines = [
        f"# Candidature — {job.company}",
        "",
        f"**{job.title}**",
        "",
        (
            "Le formulaire a été rempli dans le navigateur. **Rien n'a été "
            "envoyé** : vérifiez les champs, complétez ce qui reste, puis "
            "envoyez vous-même."
            if prefilled
            else "Le remplissage automatique s'est arrêté. Tout ce qu'il faut "
            "pour candidater à la main est ci-dessous."
        ),
        "",
        f"- Formulaire : {outcome.apply_url}",
    ]
    if outcome.ats:
        lines.append(f"- ATS : {outcome.ats}")
    lines.append(f"- Annonce : {job.url}")
    lines.append(f"- Dossier : `{directory}`")
    if outcome.screenshot:
        lines.append(f"- Capture : `{Path(outcome.screenshot).name}`")

    if outcome.blockers:
        lines += ["", "## Ce qui a bloqué", ""]
        lines += [f"- {blocker}" for blocker in outcome.blockers]

    if outcome.filled:
        lines += ["", "## Champs remplis automatiquement", ""]
        lines += [f"- {label}" for label in outcome.filled]

    if outcome.todo:
        lines += ["", "## À faire vous-même", ""]
        lines += [f"- [ ] {label}" for label in outcome.todo]

    if values:
        lines += ["", "## À recopier", ""]
        for slot, value in values.items():
            label = _LABELS.get(slot)
            if label:
                lines.append(f"- {label} : {value}")

    # La lettre reproduite est celle du dossier, transmise dans `values` : si
    # l'utilisateur a corrigé `lettre.md`, c'est sa version qu'il doit avoir
    # sous les yeux pour la recopier, pas celle qu'avait produite le modèle.
    letter_text = (values or {}).get(Slot.COVER_LETTER, "")
    if not letter_text and record.letter is not None:
        letter_text = record.letter.text
    if letter_text:
        lines += [
            "",
            "## Pièces",
            "",
            "- CV : `cv.pdf` (dans ce dossier)",
            "- Lettre : `lettre.md` (dans ce dossier) — reproduite ci-dessous "
            "pour un copier-coller direct",
            "",
            "---",
            "",
            letter_text.strip(),
            "",
        ]

    if plan is not None and plan.prefilled:
        lines += [
            "",
            "> Champs déjà remplis par le site et laissés intacts : "
            + ", ".join(plan.prefilled),
            "",
        ]

    path = directory / HANDOFF_FILE
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path
