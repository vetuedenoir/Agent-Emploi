"""Modèles de données partagés par tous les modules.

Chaque offre traverse une machine à états (`JobState`) ; son état est persisté
après chaque transition pour que la boucle soit reprenable après interruption.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import StrEnum
from typing import Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class JobState(StrEnum):
    """États d'une offre. L'ordre suit le parcours nominal."""

    DISCOVERED = "discovered"
    PRESCREENED = "prescreened"
    FIT_OK = "fit_ok"
    DRAFTED = "drafted"
    REVIEWED = "reviewed"
    AWAITING_USER = "awaiting_user"
    APPROVED = "approved"
    PREFILLED = "prefilled"
    SUBMITTED = "submitted"
    HANDOFF = "handoff"
    #: Offre ajoutée à la main pour mémoire : ni notée ni rédigée, mais suivie
    #: jusqu'à l'envoi — ou reprise plus tard pour préparer une candidature.
    TRACKED = "tracked"
    REJECTED = "rejected"


#: Transitions autorisées. Toute offre peut être rejetée à n'importe quel stade.
ALLOWED_TRANSITIONS: dict[JobState, frozenset[JobState]] = {
    JobState.DISCOVERED: frozenset({JobState.PRESCREENED, JobState.REJECTED}),
    JobState.PRESCREENED: frozenset({JobState.FIT_OK, JobState.REJECTED}),
    JobState.FIT_OK: frozenset({JobState.DRAFTED, JobState.REJECTED}),
    JobState.DRAFTED: frozenset({JobState.REVIEWED, JobState.REJECTED}),
    # Une revue négative renvoie en rédaction pour une seule reprise.
    JobState.REVIEWED: frozenset(
        {JobState.AWAITING_USER, JobState.DRAFTED, JobState.REJECTED}
    ),
    # Une offre en attente ne peut que recevoir une décision de l'utilisateur :
    # aucun chemin ne mène au remplissage sans passer par `APPROVED`.
    JobState.AWAITING_USER: frozenset({JobState.APPROVED, JobState.REJECTED}),
    JobState.APPROVED: frozenset(
        {JobState.PREFILLED, JobState.HANDOFF, JobState.REJECTED}
    ),
    JobState.PREFILLED: frozenset(
        {JobState.SUBMITTED, JobState.HANDOFF, JobState.REJECTED}
    ),
    JobState.SUBMITTED: frozenset(),
    JobState.HANDOFF: frozenset({JobState.SUBMITTED, JobState.REJECTED}),
    # Une offre suivie se prépare (elle rejoint alors le parcours au
    # pré-filtrage, sans en subir les filtres), s'envoie à la main, ou s'oublie.
    JobState.TRACKED: frozenset(
        {JobState.PRESCREENED, JobState.SUBMITTED, JobState.REJECTED}
    ),
    JobState.REJECTED: frozenset(),
}


class InvalidTransition(ValueError):
    """Transition d'état interdite par `ALLOWED_TRANSITIONS`."""


def check_transition(current: JobState, target: JobState) -> None:
    """Lève `InvalidTransition` si le passage n'est pas autorisé."""
    if target not in ALLOWED_TRANSITIONS[current]:
        raise InvalidTransition(f"{current} -> {target} n'est pas autorisé")


#: Paramètres de pistage, retirés même quand la requête est conservée.
TRACKING_PARAMS = frozenset(
    {"ref", "refid", "trk", "trackingid", "src", "source", "from", "fbclid", "gclid"}
)


def canonical_url(url: str, *, keep_query: bool = False) -> str:
    """Normalise une URL pour le calcul d'identifiant.

    Retire le fragment, les paramètres de requête (souvent du tracking) et la
    barre oblique finale, pour que la même offre partagée via deux liens
    différents produise le même identifiant.

    `keep_query` sert aux URL collées à la main : chez Indeed
    (`viewjob?jk=…`), c'est la requête qui désigne l'offre, et la retirer
    donnerait le même identifiant à toutes. Les paramètres de pistage partent
    quand même, et les autres sont triés.
    """
    parts = urlsplit(url.strip())
    path = parts.path.rstrip("/") or "/"
    query = ""
    if keep_query:
        kept = sorted(
            (key, value)
            for key, value in parse_qsl(parts.query)
            if not key.lower().startswith("utm_") and key.lower() not in TRACKING_PARAMS
        )
        query = urlencode(kept)
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, query, ""))


def job_id(source: str, url: str, *, keep_query: bool = False) -> str:
    """Identifiant stable d'une offre : hash de la source et de l'URL canonique."""
    canonical = canonical_url(url, keep_query=keep_query)
    digest = hashlib.sha256(f"{source}\n{canonical}".encode()).hexdigest()
    return digest[:32]


class Job(BaseModel):
    """Une offre normalisée, quelle que soit sa source.

    Les offres sortent de `search()` sans `description` : celle-ci demande une
    requête par offre et n'est récupérée qu'après le pré-filtrage, via
    `enrich()`. `is_enriched` permet de savoir où l'on en est.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    source: str
    url: HttpUrl
    title: str
    company: str
    location: str | None = None
    contract: str | None = None
    remote: str | None = None
    salary: str | None = None
    description: str = ""
    posted_at: datetime | None = None
    #: Langue de l'annonce telle que déclarée par la source (indice, pas vérité).
    language: str | None = None
    #: Secteurs d'activité, utiles au pré-filtrage thématique.
    sectors: list[str] = Field(default_factory=list)
    #: URL réelle de candidature — souvent un ATS externe (Greenhouse, Lever…).
    apply_url: str | None = None
    #: Nom de l'ATS, qui détermine la stratégie de remplissage à l'étape 6.
    ats: str | None = None
    #: Identifiants internes à la source, nécessaires pour aller chercher le détail.
    source_ref: dict[str, str] = Field(default_factory=dict)
    #: Charge utile brute de la source, conservée pour déboguer sans re-requêter.
    raw: dict = Field(default_factory=dict)

    @property
    def is_enriched(self) -> bool:
        """Vrai si la description a été récupérée."""
        return bool(self.description)

    @classmethod
    def build(cls, *, source: str, url: str, **fields) -> Job:
        """Construit une offre en dérivant son `id` de la source et de l'URL."""
        return cls(id=job_id(source, url), source=source, url=url, **fields)


class FitVerdict(BaseModel):
    """Verdict d'adéquation entre une offre et le CV.

    `language` est produit ici plutôt que par un appel dédié : le modèle lit déjà
    l'annonce, et cette valeur suffit ensuite à choisir la version du CV.
    """

    score: int = Field(ge=0, le=100)
    verdict: Literal["apply", "maybe", "skip"]
    matched: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    reason: str
    language: Literal["fr", "en"]


class GateVerdict(BaseModel):
    """Verdict de la porte Jev, rendu avant le fit-check.

    Des nombres, pas de texte : Jev ne rédige pas. Les probabilités sont celles
    d'un « oui » à la question posée ; `contract_ok` est absent quand aucune
    liste de contrats n'est configurée, la question n'ayant alors pas été posée.
    """

    #: Position sur l'échelle d'adéquation, fractionnaire (0 = hors périmètre).
    adequation: float
    experience_blocking: float = Field(ge=0.0, le=1.0)
    contract_ok: float | None = Field(default=None, ge=0.0, le=1.0)
    passed: bool
    #: Motif court d'un rejet (`jev:experience(0.82)`), absent si l'offre passe.
    reason: str | None = None
    model: str = "jev-latest"
    at: datetime = Field(default_factory=utcnow)


class Letter(BaseModel):
    """Lettre de motivation générée."""

    text: str
    language: Literal["fr", "en"]
    #: Formules interdites détectées après génération (voir banned_phrases.txt).
    banned_hits: list[str] = Field(default_factory=list)
    regenerated: bool = False
    #: Vrai si l'utilisateur a corrigé `lettre.md` à la main avant d'approuver.
    #: C'est alors sa version qui est ici, et c'est elle qui partira à l'étape 6.
    edited: bool = False

    @property
    def word_count(self) -> int:
        return len(self.text.split())


class ReviewVerdict(BaseModel):
    """Revue de la lettre : cohérence factuelle avec le CV, ton, longueur."""

    approved: bool
    issues: list[str] = Field(default_factory=list)
    #: Affirmations de la lettre absentes du CV — motif de rejet le plus grave.
    unsupported_claims: list[str] = Field(default_factory=list)


class UserDecision(BaseModel):
    """La décision de l'utilisateur sur un dossier — le seul feu vert du système.

    Persistée à la fois dans `jobs.jsonl` (pour l'étape 6) et dans
    `decision.json` au sein du dossier (pour que le dossier reste lisible sans
    lancer le programme).
    """

    decision: Literal["approved", "rejected"]
    at: datetime = Field(default_factory=utcnow)
    #: Motif d'un rejet, ou remarque libre sur une approbation.
    note: str | None = None
    #: Réserves affichées au moment de la décision : une approbation « malgré
    #: tout » doit rester traçable.
    concerns: list[str] = Field(default_factory=list)
    letter_edited: bool = False


class ApplyOutcome(BaseModel):
    """Ce qu'a donné le remplissage assisté du formulaire, à l'étape 6.

    `submitted` reste faux tant que l'utilisateur n'a pas déclaré avoir envoyé
    lui-même : aucun chemin de code ne clique sur le bouton d'envoi, donc rien
    ne peut le passer à vrai sans une affirmation humaine.
    """

    status: Literal["prefilled", "handoff"]
    at: datetime = Field(default_factory=utcnow)
    apply_url: str
    ats: str | None = None
    #: Champs effectivement remplis, sous leur libellé affiché.
    filled: list[str] = Field(default_factory=list)
    #: Champs laissés à l'utilisateur parce qu'aucune donnée ne leur correspond
    #: (cases à cocher, listes déroulantes, questions libres).
    todo: list[str] = Field(default_factory=list)
    #: Motif d'un `handoff` : captcha, connexion requise, champ requis inconnu.
    blockers: list[str] = Field(default_factory=list)
    screenshot: str | None = None
    submitted: bool = False


class JobRecord(BaseModel):
    """Une offre enrichie et ce qu'on en sait, dans `data/jobs.jsonl`.

    `seen.jsonl` ne garde que l'état d'une offre — assez pour ne pas la
    retraiter, pas assez pour rédiger une lettre. Les offres qui franchissent le
    pré-filtrage sont donc conservées en entier ici, avec leur verdict : l'étape
    de rédaction les reprend sans réinterroger la source.
    """

    job: Job
    #: Verdict de la porte Jev. Conservé même quand l'offre passe : une passe
    #: reprise après un échec du fit-check ne repaie pas la porte.
    gate: GateVerdict | None = None
    fit: FitVerdict | None = None
    lexical_score: float | None = None
    #: Lettre rédigée à l'étape 4, conservée pour qu'une passe interrompue après
    #: la génération n'ait pas à repayer l'appel.
    letter: Letter | None = None
    review: ReviewVerdict | None = None
    #: Dossier `outbox/` produit pour l'utilisateur, une fois la revue passée.
    outbox: str | None = None
    #: Décision rendue à l'étape 5. Tant qu'elle est absente, rien ne part.
    decision: UserDecision | None = None
    #: Résultat du remplissage assisté, à l'étape 6.
    application: ApplyOutcome | None = None
    #: Dossier d'archive `applications/`, une fois la candidature classée. Sa
    #: présence signale que l'offre est sortie du flux courant.
    archive: str | None = None
    at: datetime = Field(default_factory=utcnow)


class SeenEntry(BaseModel):
    """Une ligne de `data/seen.jsonl` : la mémoire des offres déjà traitées."""

    id: str
    source: str
    url: str
    company: str
    title: str
    state: JobState
    first_seen: datetime
    last_state_change: datetime
    reason: str | None = None

    def dedup_key(self) -> tuple[str, str]:
        """Clé de dédoublonnage secondaire : entreprise + titre normalisés.

        Une même offre republiée change souvent d'URL — donc d'identifiant — mais
        garde son entreprise et son intitulé.
        """
        return (normalize(self.company), normalize(self.title))


def normalize(text: str) -> str:
    """Minuscules, espaces compactés : base des comparaisons approximatives."""
    return " ".join(text.lower().split())


class LlmUsage(BaseModel):
    """Une ligne de `data/llm_usage.jsonl` : la trace de coût d'un appel."""

    at: datetime = Field(default_factory=utcnow)
    task: str
    provider: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_est: float = 0.0
    job_id: str | None = None
    ok: bool = True
    error: str | None = None
