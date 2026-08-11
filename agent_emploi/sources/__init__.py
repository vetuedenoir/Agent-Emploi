"""Sources d'offres d'emploi.

Chaque source implémente `JobSource` : le reste du système ne sait pas d'où
viennent les offres, seulement qu'elles sont normalisées en `Job`.

Deux sources, de natures opposées : WTTJ repose sur des services publics non
contractuels — rapide à interroger, susceptible de fermer du jour au lendemain ;
France Travail est une API officielle, documentée et stable, mais demande des
identifiants. La seconde est le filet de sécurité de la première.
"""

from agent_emploi.sources.base import (
    JobSource,
    SearchQuery,
    SourceError,
    missing_env,
    prompt_missing_env,
)
from agent_emploi.sources.france_travail import FranceTravailSource
from agent_emploi.sources.wttj import WttjSource

REGISTRY: dict[str, type] = {
    "wttj": WttjSource,
    "france_travail": FranceTravailSource,
}

__all__ = [
    "REGISTRY",
    "FranceTravailSource",
    "JobSource",
    "SearchQuery",
    "SourceError",
    "WttjSource",
    "missing_env",
    "prompt_missing_env",
]
