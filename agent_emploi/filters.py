"""Pré-filtrage déterministe — aucun appel LLM.

C'est l'étage qui absorbe le volume : sur une passe de recherche, la grande
majorité des offres est écartée ici, gratuitement. Le LLM ne voit que ce qui a
survécu.

Le filtrage se fait en deux temps, calqués sur la recherche en deux temps des
sources :

* `screen_metadata()` ne regarde que les métadonnées (titre, contrat, date…).
  Il s'exécute **avant** l'enrichissement : une offre rejetée ici ne coûte même
  pas la requête de détail.
* `screen_content()` regarde la description complète, donc après enrichissement.
  Il porte le score lexical CV ↔ offre, dernier verrou avant le LLM.

Chaque rejet porte un motif court et stable (`excluded:commercial`,
`lexical:0.04<0.10`) : ce motif est persisté dans `seen.jsonl`, et c'est en le
relisant qu'on affine les seuils au fil des passes.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta

from agent_emploi.config import FiltersConfig, SearchConfig
from agent_emploi.models import Job, utcnow

#: Mots vides français et anglais, plus le vocabulaire d'annonce qui apparaît
#: partout (« poste », « équipe », « experience ») et ne discrimine donc rien.
STOPWORDS = frozenset(
    """
    a ai au aux avec ce ces dans de des du elle en et eux il je la le les leur
    lui ma mais me meme mes moi mon ne nos notre nous on ou par pas pour qu que
    qui sa se ses son sur ta te tes toi ton tu un une vos votre vous y est sont
    etre avoir plus tres bien tout tous toute toutes chez sans sous entre aussi
    and are as at be by for from has have in is it its of on or that the this
    to was were will with you your our we us
    poste postes emploi offre offres mission missions equipe equipes entreprise
    societe candidat candidate profil recherche recherchons rejoindre stage
    cdi cdd alternance experience experiences an ans annee annees jour jours
    """.split()
)

#: Jetons retenus : lettres, chiffres, et les symboles qui font partie d'un nom
#: de technologie (`c++`, `c#`, `node.js`). Deux caractères minimum, sauf pour
#: les sigles courts déjà significatifs (`ia`, `ml`, `c`).
_TOKEN = re.compile(r"[a-z0-9][a-z0-9+#._-]*")

#: Sigles d'une ou deux lettres qu'on garde malgré leur brièveté.
SHORT_KEEP = frozenset({"c", "r", "go", "ia", "ml", "ai", "nlp", "llm", "cv", "bi"})


def fold(text: str) -> str:
    """Minuscules sans accents : « Ingénieur IA » et « ingenieur ia » se valent."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def tokenize(text: str) -> list[str]:
    """Découpe un texte en jetons comparables, mots vides retirés."""
    tokens = []
    for match in _TOKEN.finditer(fold(text)):
        token = match.group(0).strip("._-")
        if not token or token in STOPWORDS:
            continue
        if len(token) < 2 and token not in SHORT_KEEP:
            continue
        if token.isdigit():
            continue
        tokens.append(token)
    return tokens


def _vector(tokens: list[str]) -> dict[str, float]:
    """Vecteur log-TF normalisé : `1 + log(n)` par jeton, puis norme 1.

    Le log amortit les répétitions — une description qui répète dix fois
    « python » ne doit pas peser dix fois plus qu'une qui le cite une fois.
    """
    counts = Counter(tokens)
    weights = {token: 1.0 + math.log(count) for token, count in counts.items()}
    norm = math.sqrt(sum(weight * weight for weight in weights.values()))
    if norm == 0:
        return {}
    return {token: weight / norm for token, weight in weights.items()}


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    """Produit scalaire de deux vecteurs normés, parcouru sur le plus petit."""
    if not left or not right:
        return 0.0
    smaller, larger = (left, right) if len(left) < len(right) else (right, left)
    return sum(weight * larger.get(token, 0.0) for token, weight in smaller.items())


class LexicalScorer:
    """Similarité cosinus entre le CV et une offre, sur vecteurs log-TF.

    Le CV est vectorisé une fois pour toutes à la construction : le score d'une
    offre coûte alors un parcours de sa description. Pas de dépendance externe,
    et surtout un score reproductible — le même couple (CV, offre) donne
    toujours la même valeur, ce qui rend les seuils réglables.

    Avec plusieurs variantes du CV, le score est celui de la plus proche :
    une offre doit ressembler à l'une d'elles, pas à leur moyenne.
    """

    def __init__(self, cv_text: str | list[str]) -> None:
        texts = [cv_text] if isinstance(cv_text, str) else list(cv_text)
        self.cv_vectors = [_vector(tokenize(text)) for text in texts]

    def scores(self, text: str) -> list[float]:
        """Score de l'offre contre chaque CV, dans l'ordre de construction."""
        other = _vector(tokenize(text))
        return [_cosine(cv_vector, other) for cv_vector in self.cv_vectors]

    def score(self, text: str) -> float:
        """Score dans [0, 1] : 0 si aucun vocabulaire commun."""
        return max(self.scores(text), default=0.0)

    def best(self, text: str) -> int:
        """Indice du CV le plus proche de l'offre (le premier à égalité)."""
        scores = self.scores(text)
        return scores.index(max(scores)) if scores else 0

    def score_job(self, job: Job) -> float:
        """Score d'une offre : titre et description confondus."""
        return self.score(f"{job.title}\n{job.description}")


@dataclass(frozen=True)
class Screening:
    """Résultat d'un étage de filtrage.

    `reason` est vide quand l'offre passe, et sinon porte un motif court et
    stable, destiné à être persisté puis agrégé.
    """

    passed: bool
    reason: str = ""
    lexical_score: float | None = None

    @classmethod
    def ok(cls, lexical_score: float | None = None) -> Screening:
        return cls(True, "", lexical_score)

    @classmethod
    def rejected(cls, reason: str, lexical_score: float | None = None) -> Screening:
        return cls(False, reason, lexical_score)


#: Longueur en deçà de laquelle un terme est traité comme un sigle : la
#: comparaison passe alors par les limites de mot. « ai » cherché en
#: sous-chaîne se trouve dans « financial », « ml » dans « html » — un filtre
#: qui laisse tout passer ne filtre rien.
ACRONYM_LEN = 4

#: Ce qui colle à un sigle sans le prolonger. L'apostrophe en fait partie :
#: sans elle, « ai » se trouverait dans « j'ai ».
_BOUNDARY = "a-z0-9'’"


def _matcher(term: str) -> re.Pattern[str] | None:
    """Motif à limites de mot pour un sigle, ou `None` pour un terme ordinaire."""
    if len(term) > ACRONYM_LEN or not term.isalnum():
        return None
    return re.compile(
        rf"(?<![{_BOUNDARY}]){re.escape(term)}(?![{_BOUNDARY}])"
    )


def contains_any(text: str, terms: list[str]) -> str | None:
    """Retourne le premier terme trouvé dans le texte, ou `None`.

    La comparaison est faite sur les formes repliées et par sous-chaîne : un
    terme de configuration comme « data scien » attrape « data science » comme
    « data scientist », ce qui évite d'énumérer les variantes.

    Les sigles font exception. Cherchés en sous-chaîne, ils attrapent n'importe
    quoi — « IA » dans « financial », « ML » dans « HTML » — et pris à la
    lettre ils manquent l'essentiel : sans limites de mot, on ne peut pas
    inscrire « AI » dans la liste, et toutes les annonces intitulées
    « AI Engineer » sont écartées comme hors sujet.
    """
    haystack = fold(text)
    for term in terms:
        needle = fold(term).strip()
        if not needle:
            continue
        pattern = _matcher(needle)
        if pattern.search(haystack) if pattern else needle in haystack:
            return term
    return None


#: Emplois du mot « apprentissage » qui ne parlent pas de contrat mais de
#: machine learning. Sans cette exception, « Ingénieur de recherche en
#: apprentissage automatique » — un CDI — passerait pour une alternance.
_ML_APPRENTISSAGE = re.compile(
    r"apprentissage\s+(?:automatique|profond|machine|statistique|artificiel|"
    r"(?:non\s+)?supervise|par\s+renforcement)"
)

#: Formats de contrat qu'un intitulé annonce sans ambiguïté. Le stage est
#: cherché en premier : sur « Stage ou alternance », c'est lui qui est retenu
#: quand la source ne tranche pas.
_TITLE_CONTRACTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Stage", ("stage", "stagiaire", "internship")),
    (
        "Alternance",
        ("alternance", "alternant", "apprentissage", "apprenti", "contrat pro"),
    ),
)


def contract_from_title(title: str, declared: str | None) -> str | None:
    """Contrat de l'offre, l'intitulé l'emportant sur le champ de la source.

    Les deux sources se trompent sur les stages et les alternances : « Stage :
    Ingénieur en informatique » est publiée en CDI chez France Travail comme
    « Alternance – Ingénieur IA » chez Welcome to the Jungle. Le champ de
    contrat sert à classer l'offre côté employeur ; l'intitulé, lui, est écrit
    pour le candidat, et c'est le plus fiable des deux.

    La valeur déclarée est conservée dès qu'elle s'accorde avec l'intitulé :
    elle est souvent plus riche (« Stage 6 mois »), et sur une annonce ouverte
    aux deux formats — « AI Engineer (Stagiaire, Alternant) » — c'est la source
    qui sait lequel elle a publié. On ne corrige que ce qui est contredit.
    """
    folded = _ML_APPRENTISSAGE.sub("", fold(title))
    announced = [
        label
        for label, needles in _TITLE_CONTRACTS
        if any(needle in folded for needle in needles)
    ]
    if not announced:
        return declared
    if declared and any(fold(label) in fold(declared) for label in announced):
        return declared
    return announced[0]


def _matches_option(value: str | None, allowed: list[str]) -> bool:
    """Vrai si la valeur correspond à l'une des options acceptées.

    Une valeur absente passe : les sources ne renseignent pas toutes les mêmes
    champs, et il vaut mieux payer un appel de détail que jeter une offre
    pertinente sur une métadonnée manquante.
    """
    if not allowed or value is None:
        return True
    folded = fold(value)
    return any(fold(option) in folded or folded in fold(option) for option in allowed)


def screen_metadata(
    job: Job,
    filters: FiltersConfig,
    search: SearchConfig,
    *,
    now: datetime | None = None,
) -> Screening:
    """Filtre une offre sur ses seules métadonnées, avant enrichissement.

    Ordre volontaire : les tests qui rejettent le plus et coûtent le moins
    d'abord, pour que le motif retenu soit le plus parlant.
    """
    excluded = contains_any(job.title, filters.excluded_any)
    if excluded is not None:
        return Screening.rejected(f"exclu:{fold(excluded)}")

    if not _matches_option(job.contract, search.contracts):
        return Screening.rejected(f"contrat:{fold(job.contract or '?')}")

    if not _matches_option(job.remote, search.remote):
        return Screening.rejected(f"teletravail:{fold(job.remote or '?')}")

    if job.posted_at is not None and search.max_age_days:
        age = (now or utcnow()) - job.posted_at
        if age > timedelta(days=search.max_age_days):
            return Screening.rejected(f"anciennete:{age.days}j>{search.max_age_days}j")

    # Le terme requis est cherché dans le titre tant que la description n'est
    # pas là. C'est le compromis qui économise le plus : sans lui, il faudrait
    # une requête de détail pour chaque offre remontée par la recherche.
    if filters.require_in_title and filters.required_any:
        if contains_any(job.title, filters.required_any) is None:
            return Screening.rejected("titre_hors_sujet")

    return Screening.ok()


def screen_content(
    job: Job, filters: FiltersConfig, scorer: LexicalScorer
) -> Screening:
    """Filtre une offre enrichie : terme requis dans le texte, puis score lexical."""
    if not job.is_enriched:
        return Screening.rejected("description_absente")

    text = f"{job.title}\n{job.description}"
    if filters.required_any and contains_any(text, filters.required_any) is None:
        return Screening.rejected("aucun_terme_requis")

    score = scorer.score(text)
    if score < filters.min_lexical_score:
        return Screening.rejected(
            f"lexical:{score:.3f}<{filters.min_lexical_score:.3f}", score
        )

    return Screening.ok(score)
