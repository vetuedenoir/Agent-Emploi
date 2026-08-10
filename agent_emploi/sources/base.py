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

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from agent_emploi.models import Job


class SourceError(RuntimeError):
    """Échec d'interrogation d'une source (réseau, format inattendu, blocage)."""


class SearchQuery(BaseModel):
    """Critères d'une passe de recherche, indépendants de la source."""

    text: str
    countries: list[str] = Field(default_factory=list)
    contracts: list[str] = Field(default_factory=list)
    remote: list[str] = Field(default_factory=list)
    limit: int = 40
    max_age_days: int | None = None


@runtime_checkable
class JobSource(Protocol):
    name: str

    def search(self, query: SearchQuery) -> list[Job]:
        """Renvoie les offres correspondant aux critères, sans description."""
        ...

    def enrich(self, job: Job) -> Job:
        """Complète une offre avec sa description et son URL de candidature."""
        ...
