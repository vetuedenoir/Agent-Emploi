"""Passes lancées depuis l'interface, exécutées en arrière-plan.

Rien n'est réimplémenté : chaque passe est la commande CLI elle-même
(`cmd_search`, `cmd_screen`…), appelée dans un thread. Ce qu'elle imprime, et
ce que ses modules journalisent, est capturé ligne à ligne pour la page de
suivi : on lit exactement le rapport qu'on lirait dans un terminal.

Une passe à la fois. Deux passes simultanées tiendraient chacune leur propre
copie en mémoire de `seen.jsonl` et pourraient traiter la même offre deux fois.
Pour la même raison, les actions de l'interface sont refusées pendant une passe
(voir `routes.actions`).

Les identifiants de source ne sont jamais demandés ici : personne ne lit
l'invite d'un thread. Ils viennent de l'environnement ou de `.env`, sinon la
source est ignorée, comme dans une tâche planifiée.
"""

from __future__ import annotations

import io
import json
import logging
import sys
import threading
import traceback
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable
from uuid import uuid4

from agent_emploi.config import Config
from agent_emploi.models import utcnow

#: Lignes gardées dans le journal des passes, par passe : assez pour le bilan.
TAIL_LINES = 60

#: Passes gardées en mémoire pour la page et le tableau de bord.
HISTORY = 20

#: Journaux trop bavards pour la page : une ligne par requête HTTP.
QUIET_LOGGERS = ("httpx", "httpcore")


@dataclass(frozen=True)
class PassSpec:
    label: str
    help: str
    #: Options de formulaire proposées, parmi `limit` et `letters`.
    options: tuple[str, ...]
    run: Callable[..., int]


def _commands() -> dict[str, PassSpec]:
    """Les passes proposées, chacune adossée à sa commande CLI."""
    from agent_emploi import cli

    return {
        "search": PassSpec(
            "Recherche",
            "interroge les sources et mémorise les offres nouvelles",
            ("limit",),
            lambda config, dry_run, limit, letters: cli.cmd_search(
                config, limit=limit, dry_run=dry_run, ask=False
            ),
        ),
        "screen": PassSpec(
            "Filtrage",
            "recherche, filtres, porte Jev et fit-check",
            ("limit",),
            lambda config, dry_run, limit, letters: cli.cmd_screen(
                config, limit=limit, dry_run=dry_run, ask=False
            ),
        ),
        "draft": PassSpec(
            "Rédaction",
            "lettres, relecture et dossiers outbox/ — la seule étape payante",
            ("limit",),
            lambda config, dry_run, limit, letters: cli.cmd_draft(
                config, limit=limit, dry_run=dry_run
            ),
        ),
        "run": PassSpec(
            "Chaîne complète",
            "recherche, filtrage, rédaction et archivage, jusqu'à votre validation",
            ("limit", "letters"),
            lambda config, dry_run, limit, letters: cli.cmd_run(
                config, limit=limit, draft_limit=letters, dry_run=dry_run, ask=False
            ),
        ),
        "archive": PassSpec(
            "Archivage",
            "classe les candidatures envoyées dans applications/",
            (),
            lambda config, dry_run, limit, letters: cli.cmd_archive(
                config, dry_run=dry_run, include_rejected=False, keep=False
            ),
        ),
    }


class PassBusy(RuntimeError):
    """Une passe tourne déjà."""


@dataclass
class PassRun:
    id: str
    name: str
    label: str
    dry_run: bool = False
    limit: int | None = None
    letters: int | None = None
    started: datetime = field(default_factory=utcnow)
    finished: datetime | None = None
    #: `running`, puis `ok` (code 0), `failed` (code non nul) ou `crashed`.
    status: str = "running"
    exit_code: int | None = None
    lines: list[str] = field(default_factory=list)
    #: Page à rouvrir une fois la passe finie (la fiche d'une offre préparée),
    #: sous la forme (adresse, libellé).
    link: tuple[str, str] | None = None

    @property
    def running(self) -> bool:
        return self.status == "running"

    @property
    def options(self) -> str:
        parts = []
        if self.dry_run:
            parts.append("à blanc")
        if self.limit is not None:
            parts.append(f"limite {self.limit}")
        if self.letters is not None:
            parts.append(f"{self.letters} lettre(s) max")
        return ", ".join(parts)

    def record(self) -> dict:
        """La ligne de `passes.jsonl` : le bilan, pas tout le journal."""
        return {
            "id": self.id,
            "name": self.name,
            "label": self.label,
            "dry_run": self.dry_run,
            "limit": self.limit,
            "letters": self.letters,
            "started": self.started.isoformat(),
            "finished": self.finished.isoformat() if self.finished else None,
            "status": self.status,
            "exit_code": self.exit_code,
            "lines": self.lines[-TAIL_LINES:],
            "link": list(self.link) if self.link else None,
        }

    @classmethod
    def from_record(cls, data: dict) -> PassRun:
        run = cls(
            id=data["id"],
            name=data["name"],
            label=data.get("label", data["name"]),
            dry_run=data.get("dry_run", False),
            limit=data.get("limit"),
            letters=data.get("letters"),
            started=datetime.fromisoformat(data["started"]),
            status=data.get("status", "ok"),
            exit_code=data.get("exit_code"),
            lines=data.get("lines", []),
            link=tuple(data["link"]) if data.get("link") else None,
        )
        if data.get("finished"):
            run.finished = datetime.fromisoformat(data["finished"])
        return run


class _Capture(io.TextIOBase):
    """Remplace `sys.stdout`/`sys.stderr` pendant une passe.

    Seul le thread de la passe est capturé : les autres threads du serveur
    continuent d'écrire là où ils écrivaient. Les lignes incomplètes attendent
    leur fin (`print` écrit le texte, puis le saut de ligne).
    """

    def __init__(self, original, thread_id: int, sink: Callable[[str], None]) -> None:
        self.original = original
        self.thread_id = thread_id
        self.sink = sink
        self._pending = ""

    def write(self, text: str) -> int:
        if threading.get_ident() != self.thread_id:
            return self.original.write(text)
        self._pending += text
        *complete, self._pending = self._pending.split("\n")
        for line in complete:
            self.sink(line)
        return len(text)

    def flush(self) -> None:
        if threading.get_ident() != self.thread_id:
            self.original.flush()

    def drain(self) -> None:
        if self._pending:
            self.sink(self._pending)
            self._pending = ""


class _LogToRun(logging.Handler):
    """Les journaux du thread de la passe, en lignes de la page."""

    def __init__(self, thread_id: int, sink: Callable[[str], None]) -> None:
        super().__init__(logging.INFO)
        self.thread_id = thread_id
        self.sink = sink
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))

    def filter(self, record: logging.LogRecord) -> bool:
        return record.thread == self.thread_id and not record.name.startswith(QUIET_LOGGERS)

    def emit(self, record: logging.LogRecord) -> None:
        self.sink(self.format(record))


class PassRunner:
    """Lance les passes, une à la fois, et garde leur trace."""

    def __init__(
        self,
        config: Config,
        *,
        commands: dict[str, PassSpec] | None = None,
        write_lock: threading.RLock | None = None,
    ) -> None:
        self.config = config
        self.commands = commands if commands is not None else _commands()
        #: Le verrou d'écriture des stores, pris pour démarrer : une action
        #: commencée finit avant que la passe ne parte, et une action qui le
        #: prend ensuite voit la passe en cours.
        self.write_lock = write_lock or threading.RLock()
        self.path = config.paths.data / "passes.jsonl"
        self._lock = threading.Lock()
        self._current: PassRun | None = None
        self._thread: threading.Thread | None = None
        self._history: deque[PassRun] = deque(self._load(), maxlen=HISTORY)

    # ------------------------------------------------------------------ lecture

    def _load(self) -> list[PassRun]:
        if not self.path.exists():
            return []
        runs: list[PassRun] = []
        for line in self.path.read_text(encoding="utf-8").splitlines()[-HISTORY:]:
            try:
                runs.append(PassRun.from_record(json.loads(line)))
            except (ValueError, KeyError):
                continue
        return runs

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._current is not None

    @property
    def current(self) -> PassRun | None:
        with self._lock:
            return self._current

    def history(self) -> list[PassRun]:
        """Les passes terminées puis l'éventuelle passe en cours, les récentes d'abord."""
        with self._lock:
            runs = list(self._history)
            if self._current is not None:
                runs.append(self._current)
        return list(reversed(runs))

    def get(self, run_id: str) -> PassRun | None:
        return next((run for run in self.history() if run.id == run_id), None)

    # ------------------------------------------------------------------ écriture

    def start(
        self,
        name: str,
        *,
        dry_run: bool = False,
        limit: int | None = None,
        letters: int | None = None,
    ) -> PassRun:
        """Démarre une passe. Lève `PassBusy` si une autre tourne, `KeyError` si inconnue."""
        spec = self.commands[name]
        run = PassRun(
            id=uuid4().hex[:12],
            name=name,
            label=spec.label,
            dry_run=dry_run,
            limit=limit if "limit" in spec.options else None,
            letters=letters if "letters" in spec.options else None,
        )
        return self._launch(run, spec.run)

    def start_task(
        self,
        name: str,
        label: str,
        task: Callable[[], int],
        *,
        link: tuple[str, str] | None = None,
    ) -> PassRun:
        """Démarre une passe ponctuelle, hors des commandes proposées.

        Même régime que les autres : une à la fois, sortie capturée, bilan
        gardé. C'est ce qui sert au fit-check d'une offre ajoutée à la main.
        """
        run = PassRun(id=uuid4().hex[:12], name=name, label=label, link=link)
        return self._launch(run, lambda config, dry_run, limit, letters: task())

    def _launch(self, run: PassRun, command: Callable[..., int]) -> PassRun:
        with self.write_lock, self._lock:
            if self._current is not None:
                raise PassBusy(f"une passe est déjà en cours : {self._current.label}")
            self._current = run

        self._thread = threading.Thread(
            target=self._execute, args=(command, run), name=f"passe-{run.name}", daemon=True
        )
        self._thread.start()
        return run

    def join(self, timeout: float | None = None) -> None:
        """Attend la fin de la passe en cours (utile aux tests)."""
        if self._thread is not None:
            self._thread.join(timeout)

    def _execute(self, command: Callable[..., int], run: PassRun) -> None:
        thread_id = threading.get_ident()
        stdout = _Capture(sys.stdout, thread_id, run.lines.append)
        stderr = _Capture(sys.stderr, thread_id, run.lines.append)
        handler = _LogToRun(thread_id, run.lines.append)
        root = logging.getLogger()
        previous_level = root.level
        # Les modules journalisent en INFO ce qu'ils font offre par offre ; le
        # niveau par défaut (WARNING) le ferait disparaître de la page.
        if root.getEffectiveLevel() > logging.INFO:
            root.setLevel(logging.INFO)
        root.addHandler(handler)
        sys.stdout, sys.stderr = stdout, stderr
        try:
            code = command(self.config, run.dry_run, run.limit, run.letters)
            run.exit_code = code
            run.status = "ok" if code == 0 else "failed"
        except BaseException:  # noqa: BLE001 — une passe ne doit pas tuer le serveur
            run.lines.extend(traceback.format_exc().rstrip().splitlines())
            run.status = "crashed"
        finally:
            stdout.drain()
            stderr.drain()
            # Remis seulement si personne ne les a remplacés entre-temps.
            if sys.stdout is stdout:
                sys.stdout = stdout.original
            if sys.stderr is stderr:
                sys.stderr = stderr.original
            root.removeHandler(handler)
            root.setLevel(previous_level)
            run.finished = utcnow()
            self._persist(run)
            with self._lock:
                self._history.append(run)
                self._current = None

    def _persist(self, run: PassRun) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(run.record(), ensure_ascii=False) + "\n")
        except OSError:
            logging.getLogger(__name__).warning("passes.jsonl non écrit", exc_info=True)


def parse_int(value: str | None) -> int | None:
    """Entier positif d'un champ de formulaire, ou `None` s'il est vide ou invalide."""
    try:
        number = int((value or "").strip())
    except ValueError:
        return None
    return number if number > 0 else None
