"""Persistance : mémoire des offres vues et archivage des candidatures."""

from agent_emploi.store.jobs import JobStore
from agent_emploi.store.seen import SeenStore

__all__ = ["JobStore", "SeenStore"]
