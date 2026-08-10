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


class LexicalScorer:
    """Similarité cosinus entre le CV et une offre, sur vecteurs log-TF.

    Le CV est vectorisé une fois pour toutes à la construction : le score d'une
    offre coûte alors un parcours de sa description. Pas de dépendance externe,
    et surtout un score reproductible — le même couple (CV, offre) donne
    toujours la même valeur, ce qui rend les seuils réglables.
    """

    def __init__(self, cv_text: str) -> None:
        self.cv_vector = _vector(tokenize(cv_text))

    def score(self, text: str) -> float:
        """Score dans [0, 1] : 0 si aucun vocabulaire commun."""
        if not self.cv_vector:
            return 0.0
        other = _vector(tokenize(text))
        if not other:
            return 0.0
        smaller, larger = (
            (other, self.cv_vector)
            if len(other) < len(self.cv_vector)
            else (self.cv_vector, other)
        )
        return sum(weight * larger.get(token, 0.0) for token, weight in smaller.items())

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


def contains_any(text: str, terms: list[str]) -> str | None:
    """Retourne le premier terme trouvé dans le texte, ou `None`.

    La comparaison est faite sur les formes repliées et par sous-chaîne : un
    terme de configuration comme « data scien » attrape « data science » comme
    « data scientist », ce qui évite d'énumérer les variantes.
    """
    haystack = fold(text)
    for term in terms:
        needle = fold(term).strip()
        if needle and needle in haystack:
            return term
    return None


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
