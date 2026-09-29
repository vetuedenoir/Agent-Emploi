"""Interface commune aux sources d'offres.

La recherche se fait en deux temps, volontairement :

1. `search()` interroge l'index de la source et renvoie les métadonnées d'un
   grand nombre d'offres — rapide, une requête pour des dizaines de résultats.
2. `enrich()` ne va chercher le texte complet que des offres ayant survécu au
   pré-filtrage — une requête par offre.

Séparer les deux évite de télécharger des milliers de descriptions dont la
grande majorité serait jetée par les filtres déterministes.
"""

from __future__ import annotations

import os
from getpass import getpass
from typing import Callable, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from agent_emploi.config import Location
from agent_emploi.models import Job


class SourceError(RuntimeError):
    """Échec d'interrogation d'une source (réseau, format inattendu, blocage)."""


class SearchQuery(BaseModel):
    """Critères d'une passe de recherche, indépendants de la source."""

    text: str
    countries: list[str] = Field(default_factory=list)
    #: Zone ciblée. Absente : tout le pays.
    location: Location | None = None
    contracts: list[str] = Field(default_factory=list)
    remote: list[str] = Field(default_factory=list)
    limit: int = 40
    max_age_days: int | None = None


@runtime_checkable
class JobSource(Protocol):
    name: str

    #: Variables d'environnement sans lesquelles la source ne peut rien faire
    #: (identifiants d'API). Vide pour une source ouverte comme WTTJ. Elles sont
    #: vérifiées **avant** la première requête, pour que l'absence de clé soit
    #: signalée une fois, clairement, plutôt qu'à chaque requête de chaque passe.
    required_env: tuple[str, ...]

    def search(self, query: SearchQuery) -> list[Job]:
        """Renvoie les offres correspondant aux critères, sans description."""
        ...

    def enrich(self, job: Job) -> Job:
        """Complète une offre avec sa description et son URL de candidature."""
        ...


def missing_env(source_cls: type) -> list[str]:
    """Variables d'environnement requises par une source, et non définies.

    Sert au diagnostic (`doctor`) comme à la passe de recherche : une source mal
    configurée est annoncée et ignorée, elle ne fait pas échouer la passe.
    """
    return [
        name
        for name in getattr(source_cls, "required_env", ())
        if not os.environ.get(name)
    ]


#: Une variable dont le nom porte l'un de ces mots est saisie en masqué. La
#: règle est volontairement grossière : mieux vaut masquer un identifiant public
#: par excès que laisser une clé secrète s'afficher dans le terminal, puis dans
#: l'historique de la fenêtre et dans une capture d'écran.
_SECRET_HINTS = ("SECRET", "PASSWORD", "KEY", "TOKEN")


def is_secret_env(name: str) -> bool:
    """Vrai si la variable doit être saisie sans être affichée."""
    return any(hint in name.upper() for hint in _SECRET_HINTS)


def prompt_missing_env(
    source_cls: type,
    *,
    ask: Callable[[str], str] | None = None,
    ask_secret: Callable[[str], str] | None = None,
) -> list[str]:
    """Demande à l'utilisateur les identifiants absents d'une source.

    Ne pas obliger à écrire un secret dans un fichier. Les valeurs saisies sont
    posées dans l'environnement du **processus courant** — elles vivent le temps
    de la commande, ne sont jamais écrites sur disque, et ne sont pas réutilisées
    à la session suivante.

    Retourne les variables effectivement renseignées ; une saisie vide laisse la
    variable absente, et la source sera ignorée comme si l'on n'avait rien
    demandé — abandonner à l'invite doit rester possible.
    """
    # Résolues à l'appel, et non en valeurs par défaut : `input` et `getpass`
    # doivent pouvoir être remplacés — par un test, ou par une interface qui ne
    # serait pas un terminal.
    ask = ask or input
    ask_secret = ask_secret or getpass

    filled: list[str] = []
    for name in missing_env(source_cls):
        prompt = f"{name} : "
        value = (ask_secret(prompt) if is_secret_env(name) else ask(prompt)).strip()
        if not value:
            continue
        os.environ[name] = value
        filled.append(name)
    return filled
