"""Chargement et validation de `config.yaml`.

La configuration est validée au démarrage : une erreur de frappe dans le YAML
échoue immédiatement avec un message précis, plutôt qu'au milieu d'une passe.
Les secrets ne transitent jamais par ce fichier — ils sont lus dans
l'environnement au moment de l'appel (voir `llm.providers`).
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

DEFAULT_CONFIG_PATH = Path("config.yaml")


class ProfileConfig(BaseModel):
    cv_markdown: Path
    cv_fr: Path
    cv_en: Path
    voice: Path
    banned_phrases: Path


class PathsConfig(BaseModel):
    data: Path = Path("data")
    outbox: Path = Path("outbox")
    applications: Path = Path("applications")

    @property
    def seen_file(self) -> Path:
        return self.data / "seen.jsonl"

    @property
    def usage_file(self) -> Path:
        return self.data / "llm_usage.jsonl"

    @property
    def jobs_file(self) -> Path:
        return self.data / "jobs.jsonl"

    def ensure(self) -> None:
        """Crée les répertoires de travail s'ils n'existent pas."""
        for directory in (self.data, self.outbox, self.applications):
            directory.mkdir(parents=True, exist_ok=True)


class SearchConfig(BaseModel):
    #: Sources interrogées, dans l'ordre. Doivent exister dans `sources.REGISTRY`.
    sources: list[str] = Field(default_factory=lambda: ["wttj"])
    queries: list[str]
    #: Pays, tels que nommés par la source (facette `offices.country` chez WTTJ).
    countries: list[str] = Field(default_factory=lambda: ["France"])
    contracts: list[str] = Field(default_factory=list)
    remote: list[str] = Field(default_factory=list)
    max_age_days: int = 30
    per_query_limit: int = 40


class FiltersConfig(BaseModel):
    required_any: list[str] = Field(default_factory=list)
    excluded_any: list[str] = Field(default_factory=list)
    #: Si vrai, un terme requis doit déjà figurer dans le titre — l'offre est
    #: alors écartée avant même la requête de détail. C'est le réglage qui pèse
    #: le plus sur le nombre de requêtes par passe.
    require_in_title: bool = True
    min_lexical_score: float = 0.0
    dedup_window_days: int = 60


class FitConfig(BaseModel):
    min_score: int = 65
    accept_verdicts: list[str] = Field(default_factory=lambda: ["apply"])


class GateConfig(BaseModel):
    """Porte Jev, entre le pré-filtrage déterministe et le fit-check LLM.

    Désactivée par défaut : elle demande une clé (`JEVMODEL_API_KEY`), et son
    absence ne doit rien changer au comportement d'une installation existante.
    """

    enabled: bool = False
    #: Adéquation minimale, sur l'échelle 0 (hors périmètre) – 3 (forte).
    min_score: float = 1.5
    #: Au-delà de cette probabilité, l'expérience exigée est jugée bloquante.
    max_blocking: float = 0.7
    #: En deçà de cette probabilité, le contrat est jugé incompatible.
    min_contract: float = 0.3


class LetterConfig(BaseModel):
    min_words: int = 150
    max_words: int = 200
    max_regenerations: int = 1
    #: Lettres rédigées par passe, à défaut de `--letters` / `--limit`.
    max_per_day: int = 5


class ModelRef(BaseModel):
    """Désignation d'un modèle chez un fournisseur."""

    provider: str
    model: str


class TaskRoute(ModelRef):
    """Route d'une tâche : un modèle principal et un repli optionnel."""

    tier: str = "free"
    fallback: ModelRef | None = None


class Pricing(BaseModel):
    """Tarifs en dollars par million de tokens."""

    input: float = 0.0
    output: float = 0.0


class LlmConfig(BaseModel):
    tasks: dict[str, TaskRoute]
    pricing: dict[str, Pricing] = Field(default_factory=dict)

    def route(self, task: str) -> TaskRoute:
        try:
            return self.tasks[task]
        except KeyError:
            known = ", ".join(sorted(self.tasks))
            raise KeyError(f"tâche LLM inconnue: {task!r} (connues: {known})") from None

    def price(self, model: str) -> Pricing:
        """Tarif d'un modèle ; gratuit par défaut si non déclaré."""
        return self.pricing.get(model, Pricing())


class BudgetConfig(BaseModel):
    daily_usd: float = 1.0
    daily_calls: int = 500


class Config(BaseModel):
    profile: ProfileConfig
    paths: PathsConfig = Field(default_factory=PathsConfig)
    search: SearchConfig
    filters: FiltersConfig = Field(default_factory=FiltersConfig)
    gate: GateConfig = Field(default_factory=GateConfig)
    fit: FitConfig = Field(default_factory=FitConfig)
    letter: LetterConfig = Field(default_factory=LetterConfig)
    llm: LlmConfig
    budget: BudgetConfig = Field(default_factory=BudgetConfig)


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> Config:
    """Charge et valide la configuration depuis un fichier YAML."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"configuration introuvable: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Config.model_validate(data)


def load_dotenv(path: Path | str = Path(".env")) -> list[str]:
    """Charge les clés d'un fichier `.env` dans l'environnement du processus.

    Les variables déjà présentes dans l'environnement l'emportent : un secret
    exporté à la main, ou injecté par un gestionnaire de secrets, ne doit pas
    être écrasé par un fichier oublié dans le dépôt. Retourne les noms chargés,
    jamais les valeurs.

    Un analyseur de quinze lignes suffit ici : le format est un `CLE=valeur` par
    ligne, ce qui ne justifie pas une dépendance supplémentaire.
    """
    path = Path(path)
    if not path.exists():
        return []
    loaded: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.removeprefix("export ").strip()
        value = value.strip().strip("'\"")
        if not name or not value or name in os.environ:
            continue
        os.environ[name] = value
        loaded.append(name)
    return loaded


def api_key(provider: str) -> str:
    """Clé d'API d'un fournisseur, lue dans l'environnement.

    Lève `RuntimeError` avec le nom exact de la variable manquante — c'est
    l'erreur de configuration la plus fréquente au premier lancement.
    """
    env_var = f"{provider.upper()}_API_KEY"
    key = os.environ.get(env_var)
    if not key:
        raise RuntimeError(
            f"clé d'API absente pour le fournisseur {provider!r}: "
            f"définir {env_var} (voir .env.example)"
        )
    return key
