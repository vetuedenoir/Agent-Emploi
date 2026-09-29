"""Passe de recherche : interroge les sources et enregistre les offres nouvelles.

C'est la seule étape qui parle aux sources. Elle ne fait aucun jugement sur la
pertinence des offres — ce travail revient au pré-filtrage (étape 3). Son rôle
est de découvrir, dédoublonner et mémoriser.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from agent_emploi.config import Config
from agent_emploi.models import Job, JobState
from agent_emploi.sources import REGISTRY, SearchQuery, SourceError, missing_env
from agent_emploi.store.seen import SeenStore

logger = logging.getLogger(__name__)


@dataclass
class SearchReport:
    """Bilan d'une passe, pour l'affichage et le suivi."""

    found: int = 0
    new: list[Job] = field(default_factory=list)
    #: Offres déjà mémorisées mais jamais filtrées (état `DISCOVERED`). Une passe
    #: interrompue avant le filtrage les laisse en plan ; les remonter ici permet
    #: à l'étape suivante de les reprendre plutôt que de les oublier.
    pending: list[Job] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def duplicates(self) -> int:
        return self.found - len(self.new)

    @property
    def to_screen(self) -> list[Job]:
        """Offres à soumettre au filtrage : les nouvelles et les oubliées."""
        return self.new + self.pending


def run_search(
    config: Config, store: SeenStore, *, limit: int | None = None, record: bool = True
) -> SearchReport:
    """Interroge chaque source pour chaque requête et retient les offres inédites.

    `record=False` permet une passe à blanc : on voit ce qui serait retenu sans
    marquer les offres comme vues, ce qui rend l'essai rejouable.
    """
    report = SearchReport()
    per_query = limit or config.search.per_query_limit
    collected: list[Job] = []

    for source_name in config.search.sources:
        source_cls = REGISTRY.get(source_name)
        if source_cls is None:
            report.errors.append(f"source inconnue: {source_name}")
            continue

        # Une source sans identifiants est annoncée une fois et ignorée : sans
        # ce test elle échouerait à chaque requête, noyant le rapport sous le
        # même message répété. Les autres sources, elles, tournent normalement.
        absent = missing_env(source_cls)
        if absent:
            report.errors.append(
                f"{source_name}: ignorée, variables absentes: {', '.join(absent)}"
            )
            continue

        source = source_cls()
        try:
            for text in config.search.queries:
                query = SearchQuery(
                    text=text,
                    countries=config.search.countries,
                    location=config.search.location,
                    contracts=config.search.contracts,
                    remote=config.search.remote,
                    limit=per_query,
                    max_age_days=config.search.max_age_days,
                )
                try:
                    jobs = source.search(query)
                except SourceError as exc:
                    report.errors.append(f"{source_name} / {text!r}: {exc}")
                    logger.warning("recherche échouée (%s, %r): %s", source_name, text, exc)
                    continue

                logger.info("%s / %r: %d offres", source_name, text, len(jobs))
                collected.extend(jobs)
        finally:
            close = getattr(source, "close", None)
            if close is not None:
                close()

    report.found = len(collected)
    # Avant d'enregistrer quoi que ce soit : les offres connues restées au stade
    # de la découverte n'ont jamais été filtrées, elles sont donc à reprendre.
    seen_ids: set[str] = set()
    for job in collected:
        if job.id in seen_ids:
            continue
        seen_ids.add(job.id)
        entry = store.get(job.id)
        if entry is not None and entry.state is JobState.DISCOVERED:
            report.pending.append(job)

    # `filter_new` dédoublonne à la fois contre l'historique et à l'intérieur du
    # lot : plusieurs requêtes ramènent forcément les mêmes offres.
    report.new = store.filter_new(collected)

    if record:
        for job in report.new:
            store.record(job)

    return report
