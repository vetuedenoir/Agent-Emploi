"""Revue de la lettre — un appel LLM, tier gratuit.

Ce que la revue cherche, dans l'ordre de gravité :

1. **Une affirmation absente du CV.** C'est le seul défaut qui peut coûter cher
   au candidat en entretien : une technologie qu'il n'a jamais pratiquée, un
   diplôme ou un chiffre inventé. Le modèle de rédaction, à qui l'on demande
   d'être convaincant, a une pente naturelle à en ajouter.
2. Une lettre qui marcherait pour n'importe quelle entreprise.
3. Un ton qui sonne généré.

Le relecteur ne réécrit rien : il constate. La reprise, s'il y en a une, est
faite par le rédacteur avec ces reproches en entrée — un modèle gratuit qui
corrige un texte payant produirait un patchwork.
"""

from __future__ import annotations

import logging

from agent_emploi.llm.router import Router
from agent_emploi.models import Job, Letter, ReviewVerdict
from agent_emploi.profile import Profile

logger = logging.getLogger(__name__)

TASK = "review"

#: La revue n'a pas besoin de l'annonce entière : l'intitulé, l'entreprise et le
#: début de la description suffisent à juger si la lettre parle bien de ce
#: poste-là.
MAX_DESCRIPTION_CHARS = 2500

SYSTEM = (
    "Tu es un relecteur exigeant. Tu vérifies qu'une lettre de motivation ne "
    "contient aucune affirmation absente du CV, qu'elle est spécifique à "
    "l'offre, et qu'elle ne sonne pas comme un texte généré. Tu ne réécris "
    "rien : tu constates. Tu n'inventes pas de reproche pour paraître utile — "
    "une lettre correcte est approuvée sans réserve."
)

CRITERES = """\
Critères, du plus grave au moins grave :

1. `unsupported_claims` : toute affirmation de la lettre qui n'est pas
   soutenue par le CV — technologie jamais mentionnée, expérience, diplôme,
   durée ou chiffre absent. Cite l'affirmation telle qu'elle est écrite. C'est
   le point le plus important : le candidat devra défendre chaque phrase en
   entretien.
2. `issues` : lettre interchangeable (aucune raison propre à cette entreprise),
   ton artificiel ou flatteur, formule creuse, répétition, longueur
   déraisonnable, langue différente de celle de l'annonce.

`approved` : vrai seulement si `unsupported_claims` est vide ET qu'aucun point
d'`issues` ne justifie une réécriture. Une maladresse de style isolée ne suffit
pas à refuser.

Réponds en français, même si la lettre est en anglais.\
"""


def build_prompt(job: Job, letter: Letter, cv_text: str) -> str:
    """Assemble la revue : CV d'abord (préfixe stable), lettre en dernier."""
    description = job.description[:MAX_DESCRIPTION_CHARS]
    if len(job.description) > MAX_DESCRIPTION_CHARS:
        description += "\n[…description tronquée]"

    return "\n".join(
        [
            "# CV du candidat",
            cv_text.strip(),
            "",
            "# Offre",
            f"Intitulé : {job.title}",
            f"Entreprise : {job.company}",
            f"Langue de l'annonce : {letter.language}",
            "",
            "Description :",
            description,
            "",
            "# Lettre à relire",
            letter.text.strip(),
            "",
            CRITERES,
        ]
    )


class ReviewAgent:
    """Relit une lettre au regard du CV et de l'offre."""

    def __init__(self, router: Router, profile: Profile) -> None:
        self.router = router
        self.profile = profile

    def review(self, job: Job, letter: Letter) -> ReviewVerdict:
        """Un appel LLM, sortie validée. Propage `LlmError` et `BudgetExceeded`."""
        verdict = self.router.structured(
            TASK,
            build_prompt(job, letter, self.profile.cv_text),
            ReviewVerdict,
            system=SYSTEM,
            job_id=job.id,
        )
        logger.info(
            "revue %s — %s (%d invention(s), %d remarque(s))",
            job.title,
            "approuvée" if verdict.approved else "refusée",
            len(verdict.unsupported_claims),
            len(verdict.issues),
        )
        return verdict

    @staticmethod
    def feedback(verdict: ReviewVerdict) -> str:
        """Reproches de la revue, mis en forme pour la reprise du rédacteur."""
        lines = ["Ta version précédente a été refusée à la relecture."]
        if verdict.unsupported_claims:
            lines.append(
                "Affirmations absentes du CV, à retirer ou à remplacer par un "
                "fait vérifiable : "
                + ", ".join(f"« {claim} »" for claim in verdict.unsupported_claims)
            )
        if verdict.issues:
            lines.append("À corriger : " + " ; ".join(verdict.issues))
        lines.append("Réécris la lettre entière en tenant compte de ces points.")
        return "\n".join(lines)
