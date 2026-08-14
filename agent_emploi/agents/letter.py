"""Rédaction de la lettre de motivation — un appel LLM, tier payant.

C'est la seule étape qui justifie un modèle payant : la lettre est le seul
livrable lu par un humain, et une lettre générique se repère à la première
phrase. Tout le reste du système existe pour qu'on n'en écrive que quelques-unes
par jour, sur des offres qui en valent la peine.

Deux choix structurent le module :

* **Le prompt est coupé en deux.** Les consignes, le CV, la voix et la liste
  noire vont dans le bloc système — identiques d'une offre à l'autre, donc mis
  en cache par le fournisseur. Seule l'annonce change d'un appel à l'autre.
* **La vérification est en Python, pas dans le prompt.** Demander « n'écris pas
  ces formules » réduit leur fréquence sans la mettre à zéro ; les compter après
  coup, si. Le module rend les défauts constatés (`Letter.banned_hits`, nombre
  de mots) et laisse l'appelant décider de régénérer — c'est lui qui tient le
  compte des reprises autorisées.
"""

from __future__ import annotations

import logging
import re

from agent_emploi.config import LetterConfig
from agent_emploi.llm.router import Router
from agent_emploi.models import FitVerdict, Job, Letter
from agent_emploi.profile import Profile

logger = logging.getLogger(__name__)

TASK = "letter"

#: Au-delà, l'annonce ne contient plus que la culture d'entreprise et les
#: mentions légales — rien dont la lettre puisse tirer un argument.
MAX_DESCRIPTION_CHARS = 8000

#: Marge confortable : une lettre de 200 mots tient largement dedans, et le
#: modèle peut réfléchir avant d'écrire.
MAX_TOKENS = 4096

#: Enrobage Markdown que le modèle ajoute parfois autour du texte demandé.
_FENCE = re.compile(r"^```[a-z]*\s*|\s*```$", re.MULTILINE)

CONSIGNES = """\
Tu rédiges une lettre de motivation à la première personne, pour le candidat
dont le CV suit. Tu écris comme lui, pas comme un modèle de lettre.

Règles :
- Longueur : entre {min_words} et {max_words} mots. Rien de plus.
- Langue : celle de l'annonce, indiquée dans la demande.
- La première phrase parle du poste ou de l'entreprise, jamais du statut du
  candidat. « Étudiant à X, je recherche un stage » est une ouverture morte :
  le lecteur le sait déjà par le CV.
- UN projet du CV, développé : le problème traité, une décision technique
  prise, le résultat. Pas deux projets survolés, pas un catalogue.
- Trois technologies nommées au maximum dans toute la lettre, et seulement
  celles que l'annonce demande. Une énumération séparée par des virgules
  (« Python, SQL et JavaScript, Docker, Nginx ») est un extrait de CV recopié :
  le lecteur a le CV sous les yeux.
- Jamais de compétence, de chiffre ou d'expérience absent du CV.
- Une raison propre à CETTE offre, appuyée sur un élément que seule cette
  annonce contient — une contrainte du poste, un produit, une façon de
  travailler. Dire que le périmètre est large ou le sujet intéressant ne vaut
  rien : n'importe quel candidat peut l'écrire.
- Pas de flatterie, pas d'adjectifs sur soi-même (« rigoureux », « motivé »,
  « autonome », « curieux », « passionné »). Des faits, le lecteur jugera.
- Pas de phrases de raccord qui affirment la correspondance sans la montrer
  (« ce qui correspond à », « ce qui rejoint directement », « en lien avec mon
  parcours »). Poser le fait suffit ; le rapprochement se voit.
- Structure : accroche sur l'offre, le projet qui prouve la capacité attendue,
  ce qui manque assumé sans excuse s'il y a lieu, formule de politesse brève.
- Tu réponds avec le texte de la lettre, et rien d'autre : ni objet, ni
  en-tête, ni commentaire, ni balise de code.\
"""

VOIX_ABSENTE = """\
Aucun échantillon d'écriture n'a été fourni. Écris dans un français sobre et
direct, phrases courtes, sans emphase.\
"""


def _system_prompt(profile: Profile, config: LetterConfig) -> str:
    """Bloc stable du prompt : consignes, CV, voix, liste noire.

    Ce texte est identique pour toutes les offres d'une passe : c'est lui que le
    cache de prompt réutilise, et c'est ce qui rend la lettre payante presque
    gratuite à partir de la deuxième offre.
    """
    sections = [
        CONSIGNES.format(min_words=config.min_words, max_words=config.max_words),
        "# CV du candidat\n" + profile.cv_text,
    ]

    if profile.has_voice:
        sections.append(
            "# La voix du candidat\n"
            "Ces échantillons sont de lui. Tu en tires sa MANIÈRE d'écrire — "
            "vocabulaire, longueur de phrase, niveau de formalité — et rien "
            "d'autre.\n"
            "Tu n'en reprends ni les formules, ni le plan, ni les tournures. "
            "Certains échantillons sont d'anciennes lettres de candidature : "
            "elles contiennent précisément les tics que les règles ci-dessus "
            "interdisent. Les y retrouver ne les autorise pas — c'est le signe "
            "qu'il faut écrire la phrase autrement.\n\n" + profile.voice
        )
    else:
        sections.append("# La voix du candidat\n" + VOIX_ABSENTE)

    if len(profile.banned):
        sections.append(
            "# Formules interdites\n"
            "Ces tournures sont refusées, y compris sous une forme voisine. "
            "Une seule suffit à faire rejeter la lettre.\n"
            + "\n".join(f"- {phrase}" for phrase in profile.banned)
        )

    return "\n\n".join(sections)


def build_prompt(job: Job, fit: FitVerdict, feedback: str | None = None) -> str:
    """Partie variable du prompt : l'annonce, ce qui colle, ce qui manque.

    `matched` et `gaps` viennent du fit-check : ils orientent la lettre vers les
    arguments réellement défendables, et évitent que le modèle mette en avant
    une compétence que l'annonce ne demande pas.
    """
    description = job.description[:MAX_DESCRIPTION_CHARS]
    if len(job.description) > MAX_DESCRIPTION_CHARS:
        description += "\n[…description tronquée]"

    langue = "français" if fit.language == "fr" else "anglais"
    lines = [
        "# Offre",
        f"Intitulé : {job.title}",
        f"Entreprise : {job.company}",
    ]
    for label, value in (
        ("Lieu", job.location),
        ("Contrat", job.contract),
        ("Télétravail", job.remote),
    ):
        if value:
            lines.append(f"{label} : {value}")
    lines += ["", "Description :", description, ""]

    lines.append("# Analyse d'adéquation")
    if fit.matched:
        lines.append(
            "Points du CV réellement demandés par l'annonce — à exploiter : "
            + ", ".join(fit.matched)
        )
    if fit.gaps:
        lines.append(
            "Exigences non couvertes — ne pas prétendre les avoir, ne pas s'en "
            "excuser non plus : " + ", ".join(fit.gaps)
        )
    lines.append("")
    lines.append(f"Rédige la lettre en {langue}, langue de l'annonce.")

    if feedback:
        lines += ["", "# Reprise", feedback]

    return "\n".join(lines)


def clean(text: str) -> str:
    """Retire l'enrobage que le modèle ajoute malgré la consigne.

    Balises de code, ligne « Objet : … », guillemets encadrant tout le texte :
    trois habitudes tenaces qu'il est moins coûteux de couper ici que de
    combattre par une régénération.
    """
    cleaned = _FENCE.sub("", text).strip()
    lines = cleaned.splitlines()
    while lines and re.match(r"^\s*(objet|subject)\s*:", lines[0], re.IGNORECASE):
        lines = lines[1:]
    cleaned = "\n".join(lines).strip()
    if len(cleaned) > 1 and cleaned[0] == '"' and cleaned[-1] == '"':
        cleaned = cleaned[1:-1].strip()
    return cleaned


class LetterAgent:
    """Rédige une lettre et signale ses défauts, sans décider des reprises."""

    def __init__(self, router: Router, profile: Profile, config: LetterConfig) -> None:
        self.router = router
        self.profile = profile
        self.config = config
        self.system = _system_prompt(profile, config)

    def write(
        self, job: Job, fit: FitVerdict, *, feedback: str | None = None
    ) -> Letter:
        """Un appel LLM. Propage `LlmError` et `BudgetExceeded`.

        `feedback` sert aux reprises : formules détectées, longueur hors bornes,
        ou reproches de la revue. L'appelant décide s'il y en a une.
        """
        text = clean(
            self.router.complete(
                TASK,
                build_prompt(job, fit, feedback),
                system=self.system,
                max_tokens=MAX_TOKENS,
                job_id=job.id,
            )
        )
        letter = Letter(
            text=text,
            language=fit.language,
            banned_hits=self.profile.banned.find(text),
            regenerated=feedback is not None,
        )
        logger.info(
            "lettre %s — %d mots, %d formule(s) interdite(s)",
            job.title,
            letter.word_count,
            len(letter.banned_hits),
        )
        return letter

    def defects(self, letter: Letter) -> list[str]:
        """Défauts déterministes de la lettre : formules interdites, longueur.

        Vide ⇒ la lettre peut passer à la revue LLM. C'est le seul contrôle
        gratuit, il passe donc en premier.
        """
        problems: list[str] = []
        if letter.banned_hits:
            problems.append(
                "formules interdites employées : "
                + ", ".join(f"« {hit} »" for hit in letter.banned_hits)
            )
        words = letter.word_count
        if words < self.config.min_words:
            problems.append(
                f"trop courte : {words} mots pour un minimum de "
                f"{self.config.min_words}"
            )
        elif words > self.config.max_words:
            problems.append(
                f"trop longue : {words} mots pour un maximum de "
                f"{self.config.max_words}"
            )
        return problems
