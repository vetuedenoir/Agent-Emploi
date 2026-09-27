"""Les stores vus par le serveur web, tenus à jour sans redémarrage.

La CLI peut écrire pendant que le serveur tourne : une passe `run` lancée dans
un terminal ajoute des lignes à `seen.jsonl` et `jobs.jsonl`. Les stores sont
donc rechargés dès que la taille ou la date de modification d'un journal
change. Un rechargement complet coûte quelques dizaines de millisecondes, et le
format en ajout seul garantit qu'il ne lit jamais une écriture à moitié faite
autrement qu'en ignorant sa dernière ligne.
"""

from __future__ import annotations

import threading
from pathlib import Path

from agent_emploi.config import Config
from agent_emploi.store.jobs import JobStore
from agent_emploi.store.seen import SeenStore


def _signature(path: Path) -> tuple[int, int] | None:
    """Ce qui change quand un journal est écrit : date et taille."""
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


class Stores:
    """Accès aux stores, rechargés quand leur fichier a bougé."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._lock = threading.Lock()
        #: Sérialise les écritures : les routes tournent dans un pool de
        #: threads, et deux clics rapprochés ne doivent pas valider deux fois la
        #: même transition.
        self.write_lock = threading.Lock()
        self._seen: SeenStore | None = None
        self._seen_sig: tuple[int, int] | None = None
        self._jobs: JobStore | None = None
        self._jobs_sig: tuple[int, int] | None = None

    def seen(self) -> SeenStore:
        path = self.config.paths.seen_file
        with self._lock:
            signature = _signature(path)
            if self._seen is None or signature != self._seen_sig:
                self._seen = SeenStore(
                    path, dedup_window_days=self.config.filters.dedup_window_days
                )
                self._seen_sig = signature
            return self._seen

    def jobs(self) -> JobStore:
        path = self.config.paths.jobs_file
        with self._lock:
            signature = _signature(path)
            if self._jobs is None or signature != self._jobs_sig:
                self._jobs = JobStore(path)
                self._jobs_sig = signature
            return self._jobs
