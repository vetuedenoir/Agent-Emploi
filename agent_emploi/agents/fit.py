"""Verdict d'adéquation offre ↔ CV — un appel LLM, tier gratuit.

C'est le premier étage de jugement : les filtres déterministes savent dire
« cette annonce ne parle pas d'IA », pas « ce candidat a le niveau attendu ».
Un seul appel suffit, et il produit au passage la langue de l'annonce, qui
servira à l'étape 4 pour choisir la version du CV — un appel dédié pour ça
serait du gaspillage.

Le modèle est tenu à un rôle étroit : évaluer, pas encourager. Les modèles du
tier gratuit ont une pente naturelle à l'optimisme ; les consignes de notation
sont donc chiffrées et la sortie contrainte par un schéma.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import create_model

from agent_emploi.config import FitConfig
from agent_emploi.llm.router import Router
from agent_emploi.models import FitVerdict, Job
from agent_emploi.profile import CvText

logger = logging.getLogger(__name__)

TASK = "fit_check"

#: Au-delà, la description n'apporte plus que la culture d'entreprise et les
#: mentions légales. Tronquer borne le coût et la latence de l'appel.
MAX_DESCRIPTION_CHARS = 6000

SYSTEM = (
    "Tu es un recruteur technique expérimenté. Tu évalues l'adéquation entre un "
    "candidat et une offre avec exigence et sans complaisance : ton rôle est "
    "d'éviter au candidat des candidatures perdues d'avance, pas de l'encourager. "
    "Tu ne juges que sur les éléments présents dans le CV et l'annonce, sans rien "
    "supposer."
)

BAREME = """\
Barème de notation (score sur 100) :
- 80-100 : le candidat couvre l'essentiel des exigences, y compris le niveau
  d'expérience et le type de contrat demandés.
- 60-79 : bon recouvrement technique, une ou deux exigences importantes non
  couvertes, candidature défendable.
- 40-59 : recouvrement partiel, exigences structurantes manquantes
  (années d'expérience, technologie centrale, diplôme imposé).
- 0-39 : hors périmètre.

Verdict :
- "apply" si le candidat a une chance réelle d'être reçu en entretien.
- "maybe" si la candidature n'est ni évidente ni absurde.
- "skip" si une exigence bloquante n'est pas couverte (années d'expérience très
  supérieures au profil, technologie centrale inconnue du CV, contrat
  incompatible).

Contraintes :
- `matched` : uniquement des compétences réellement présentes dans le CV ET
  demandées dans l'annonce.
- `gaps` : exigences de l'annonce que le CV ne couvre pas.
- `reason` : une phrase, factuelle, en français.
- `language` : la langue dans laquelle l'annonce est rédigée ("fr" ou "en"), pas
  celle du CV.\
"""


VARIANTES = """\
Choix du CV :
- Le candidat dispose de plusieurs variantes de son CV, d'une même base mais
  orientées vers des postes différents.
- `cv` : l'identifiant de la variante la plus adaptée à cette offre.
- `score`, `verdict`, `matched` et `gaps` se rapportent à cette variante seule.\
"""


def build_prompt(job: Job, cv_text: str | list[CvText]) -> str:
    """Assemble le prompt : CV d'abord, annonce ensuite, barème en dernier.

    Le CV est placé en tête parce qu'il est identique d'une offre à l'autre :
    c'est le préfixe que le cache de prompt peut réutiliser. Avec plusieurs
    variantes, chacune a son bloc, et le barème demande de choisir.
    """
    description = job.description[:MAX_DESCRIPTION_CHARS]
    if len(job.description) > MAX_DESCRIPTION_CHARS:
        description += "\n[…description tronquée]"

    if isinstance(cv_text, str):
        lines = ["# CV du candidat", cv_text.strip(), ""]
        bareme = BAREME
    else:
        lines = []
        for cv in cv_text:
            lines += [f"# CV « {cv.id} » — {cv.label}", cv.text.strip(), ""]
        bareme = BAREME + "\n\n" + VARIANTES

    lines += [
        "# Offre",
        f"Intitulé : {job.title}",
        f"Entreprise : {job.company}",
    ]
    for label, value in (
        ("Lieu", job.location),
        ("Contrat", job.contract),
        ("Télétravail", job.remote),
        ("Salaire", job.salary),
    ):
        if value:
            lines.append(f"{label} : {value}")
    lines += ["", "Description :", description, "", bareme]
    return "\n".join(lines)


class FitAgent:
    """Évalue une offre et tranche selon les seuils de `config.fit`."""

    def __init__(
        self, router: Router, cv_text: str | list[CvText], config: FitConfig
    ) -> None:
        self.router = router
        self.config = config
        # Une seule entrée sans identifiant : c'est le mode à un seul CV, dont
        # le prompt et le schéma restent ceux d'avant les variantes.
        if not isinstance(cv_text, str) and len(cv_text) == 1 and cv_text[0].id is None:
            cv_text = cv_text[0].text
        self.cv_text = cv_text
        self.schema = FitVerdict
        if not isinstance(cv_text, str):
            ids = tuple(cv.id for cv in cv_text)
            # Le schéma n'accepte que les variantes déclarées : un identifiant
            # inventé échoue à la validation et déclenche la relance du routeur.
            self.schema = create_model(
                "FitVerdictCv", __base__=FitVerdict, cv=(Literal[ids], ...)
            )

    def evaluate(self, job: Job) -> FitVerdict:
        """Un appel LLM, sortie validée. Propage `LlmError` et `BudgetExceeded`.

        Ces deux exceptions ne sont pas rattrapées ici : la première concerne une
        offre, la seconde doit arrêter toute la passe. C'est à l'orchestrateur de
        faire la différence.
        """
        verdict = self.router.structured(
            TASK,
            build_prompt(job, self.cv_text),
            self.schema,
            system=SYSTEM,
            job_id=job.id,
        )
        if type(verdict) is not FitVerdict:
            verdict = FitVerdict.model_validate(verdict.model_dump())
        logger.info(
            "fit %s — %s (%d%s) : %s",
            job.title,
            verdict.verdict,
            verdict.score,
            f", CV {verdict.cv}" if verdict.cv else "",
            verdict.reason,
        )
        return verdict

    def accepts(self, verdict: FitVerdict) -> bool:
        """Vrai si le verdict autorise le passage à la rédaction."""
        return (
            verdict.verdict in self.config.accept_verdicts
            and verdict.score >= self.config.min_score
        )

    def rejection_reason(self, verdict: FitVerdict) -> str:
        """Motif court persisté dans `seen.jsonl` pour une offre écartée."""
        if verdict.verdict not in self.config.accept_verdicts:
            return f"fit:{verdict.verdict}({verdict.score})"
        return f"fit:score {verdict.score}<{self.config.min_score}"
