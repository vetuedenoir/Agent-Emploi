"""Sources d'offres d'emploi.

Chaque source implémente `JobSource` : le reste du système ne sait pas d'où
viennent les offres, seulement qu'elles sont normalisées en `Job`.
"""

from agent_emploi.sources.base import JobSource, SearchQuery, SourceError
from agent_emploi.sources.wttj import WttjSource

REGISTRY: dict[str, type] = {"wttj": WttjSource}

__all__ = ["REGISTRY", "JobSource", "SearchQuery", "SourceError", "WttjSource"]
