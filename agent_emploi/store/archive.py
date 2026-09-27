"""Archivage des candidatures closes — `applications/<date>_<entreprise>_<poste>/`.

Deux destinations, et elles ne servent pas au même public :

- **pour le programme**, `seen.jsonl` garde l'état final et son motif ; c'est ce
  qui empêche de retraiter une offre et ce qui, relu, sert à régler les seuils.
  Rien à faire ici : les passes précédentes l'ont déjà écrit.
- **pour vous**, le dossier complet est classé dans `applications/` avec une
  fiche de suivi : ce qui a été envoyé, quand, à qui, et où en est la réponse.

Une candidature envoyée n'a plus rien à faire dans `outbox/`, qui est la pile
des dossiers en cours. L'archivage la déplace donc, pièces comprises — la lettre
telle qu'elle est partie, le CV joint, l'annonce, le verdict, la décision.

`README.md` est la seule pièce que le programme écrit une fois et ne réécrit
jamais : c'est votre carnet de suivi, et une relance notée à la main ne doit pas
disparaître à la passe suivante.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from agent_emploi.config import Config
from agent_emploi.models import JobRecord, JobState, SeenEntry, utcnow
from agent_emploi.outbox import slugify
from agent_emploi.store.jobs import JobStore
from agent_emploi.store.seen import SeenStore

logger = logging.getLogger(__name__)

#: Fiche de suivi, écrite une seule fois puis laissée à l'utilisateur.
FOLLOWUP_FILE = "README.md"

#: Délai indicatif avant relance, en jours. Une date écrite noir sur blanc se
#: relance ; un « pensez à relancer » ne se relance pas.
FOLLOWUP_DAYS = 14

#: États qui closent une candidature du point de vue du programme, avec leur
#: libellé dans la fiche. `APPROVED` n'y est pas : tant que l'utilisateur n'a
#: pas déclaré l'envoi, le dossier est encore à faire.
ARCHIVABLE: dict[JobState, str] = {
    JobState.SUBMITTED: "envoyée",
    JobState.REJECTED: "rejetée avant envoi",
}


@dataclass
class ArchivedApplication:
    """Un dossier classé, et où il se trouve désormais."""

    record: JobRecord
    directory: Path
    #: Vrai si le dossier `outbox/` d'origine a été retiré après la copie.
    moved: bool = False

    @property
    def label(self) -> str:
        return f"{self.record.job.company} — {self.record.job.title}"


@dataclass
class ArchiveReport:
    """Bilan d'une passe d'archivage."""

    candidates: int = 0
    archived: list[ArchivedApplication] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def archive_path(root: Path, record: JobRecord, at: datetime | None = None) -> Path:
    """Chemin d'archive d'une offre : `<date>_<entreprise>_<poste>`.

    Même forme que le dossier `outbox/` dont il prend la suite, mais daté du
    jour où la candidature s'est close, pas du jour où elle a été préparée :
    c'est cette date-là qu'on cherche en rouvrant l'archive.
    """
    day = (at or closed_at(record)).strftime("%Y-%m-%d")
    return Path(root) / (
        f"{day}_{slugify(record.job.company)}_{slugify(record.job.title)}"
    )


def closed_at(record: JobRecord) -> datetime:
    """Date de clôture : l'envoi déclaré, sinon la décision, sinon maintenant."""
    if record.submitted_at is not None:
        return record.submitted_at
    if record.decision is not None:
        return record.decision.at
    return utcnow()


def select_candidates(
    job_store: JobStore, seen: SeenStore, *, include_rejected: bool = False
) -> list[tuple[JobRecord, SeenEntry]]:
    """Dossiers à classer : ceux dont l'histoire est finie et qui en ont un.

    Une offre rejetée avant même qu'un dossier soit préparé n'a rien à archiver
    — elle vit déjà dans `seen.jsonl` avec son motif, ce qui suffit. Un dossier
    déjà classé et toujours présent à sa place n'est pas reclassé, et un dossier
    que l'utilisateur a supprimé à la main n'est pas ressuscité.
    """
    states = (
        set(ARCHIVABLE)
        if include_rejected
        else {state for state in ARCHIVABLE if state is not JobState.REJECTED}
    )
    ready: list[tuple[JobRecord, SeenEntry]] = []
    for record in job_store.records():
        if record.outbox is None or not Path(record.outbox).exists():
            continue
        if record.archive is not None and Path(record.archive).exists():
            continue
        entry = seen.get(record.job.id)
        if entry is None or entry.state not in states:
            continue
        ready.append((record, entry))
    ready.sort(key=lambda pair: closed_at(pair[0]))
    return ready


#: Pièces d'un dossier et ce qu'elles contiennent, pour la fiche de suivi.
_PIECES: dict[str, str] = {
    "lettre.md": "la lettre telle qu'elle est partie",
    "cv.pdf": "le CV joint",
    "offre.md": "l'annonce complète",
    "candidature.md": "la fiche de reprise à la main",
    "formulaire.png": "la capture du formulaire rempli",
    "fit.json": "le verdict d'adéquation",
    "review.json": "la relecture de la lettre",
    "decision.json": "votre décision",
    "preview.html": "le dossier sur une page",
}


def followup_markdown(
    record: JobRecord, entry: SeenEntry, files: list[str] | None = None
) -> str:
    """La fiche de suivi : ce qui est parti, et ce qu'il reste à surveiller.

    `files` est la liste réelle des pièces classées : une fiche qui annonce un
    `decision.json` absent envoie l'utilisateur chercher un fichier qui n'existe
    pas. Sans elle, la section des pièces est simplement omise.
    """
    job = record.job
    closed = closed_at(record)
    state_label = ARCHIVABLE.get(entry.state, entry.state.value)

    lines = [
        f"# {job.company} — {job.title}",
        "",
        f"- Candidature **{state_label}** le {closed.strftime('%d/%m/%Y')}",
        f"- Annonce : {job.url}",
    ]
    if job.apply_url:
        lines.append(f"- Formulaire : {job.apply_url}")
    if job.ats:
        lines.append(f"- ATS : {job.ats}")
    if job.location:
        lines.append(f"- Lieu : {job.location}")
    if job.contract:
        lines.append(f"- Contrat : {job.contract}")
    if record.fit is not None:
        lines.append(f"- Adéquation : {record.fit.score}/100 — {record.fit.reason}")
    if record.letter is not None:
        edited = " (corrigée à la main)" if record.letter.edited else ""
        lines.append(
            f"- Lettre : {record.letter.word_count} mots, "
            f"{record.letter.language}{edited}"
        )
    if record.decision is not None and record.decision.note:
        lines.append(f"- Votre note : {record.decision.note}")
    if entry.reason:
        lines.append(f"- Dernier motif enregistré : `{entry.reason}`")

    pieces = [name for name in _PIECES if name in set(files or [])]
    if pieces:
        lines += ["", "## Pièces", ""]
        lines += [f"- `{name}` — {_PIECES[name]}" for name in pieces]

    lines += ["", "## Suivi", ""]
    if entry.state is JobState.SUBMITTED:
        relance = (closed + timedelta(days=FOLLOWUP_DAYS)).strftime("%d/%m/%Y")
        lines += [
            f"- [ ] Relancer à partir du {relance} (J+{FOLLOWUP_DAYS})",
            "- [ ] Réponse reçue le … :",
            "- [ ] Entretien le … :",
        ]
    else:
        lines.append("Candidature close avant envoi — rien à surveiller.")

    lines += [
        "",
        "### Journal",
        "",
        f"- {closed.strftime('%d/%m/%Y')} — candidature {state_label}.",
        "",
        "> Ce fichier est à vous : le programme l'écrit une fois et ne le "
        "réécrit jamais.",
        "",
    ]
    return "\n".join(lines)


def archive_record(
    root: Path,
    record: JobRecord,
    entry: SeenEntry,
    *,
    move: bool = True,
) -> ArchivedApplication:
    """Classe un dossier dans `applications/` et retourne où il a atterri.

    L'ordre compte : la copie est intégrale et vérifiée avant que l'original ne
    soit retiré. Une interruption laisse au pire le dossier aux deux endroits,
    jamais à aucun.
    """
    source = Path(record.outbox or "")
    if not source.exists():
        raise FileNotFoundError(f"dossier introuvable: {source}")

    destination = _free_path(archive_path(root, record))
    destination.mkdir(parents=True, exist_ok=True)

    for item in sorted(source.iterdir()):
        if item.is_dir():
            shutil.copytree(item, destination / item.name, dirs_exist_ok=True)
        else:
            shutil.copyfile(item, destination / item.name)

    followup = destination / FOLLOWUP_FILE
    if not followup.exists():
        files = sorted(item.name for item in destination.iterdir())
        followup.write_text(
            followup_markdown(record, entry, files), encoding="utf-8"
        )

    moved = False
    if move and destination.resolve() != source.resolve():
        shutil.rmtree(source)
        moved = True

    return ArchivedApplication(record=record, directory=destination, moved=moved)


def _free_path(path: Path) -> Path:
    """Chemin libre, suffixé si besoin.

    Deux candidatures de la même entreprise au même intitulé closes le même jour
    restent deux candidatures : on ne les fond pas dans un seul dossier.
    """
    if not path.exists():
        return path
    for index in range(2, 100):
        candidate = path.with_name(f"{path.name}-{index}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"trop de dossiers homonymes: {path}")


def run_archive(
    config: Config,
    *,
    seen: SeenStore,
    job_store: JobStore,
    record: bool = True,
    include_rejected: bool = False,
    move: bool = True,
) -> ArchiveReport:
    """Classe les candidatures closes. Ne touche à aucun état.

    L'archivage est un rangement, pas une transition : `seen.jsonl` a déjà
    l'état final, et un dossier classé ne change pas de statut du fait d'avoir
    changé de répertoire.

    `record=False` fait une passe à blanc : rien n'est copié, rien n'est
    déplacé, rien n'est écrit — seule la liste de ce qui serait classé est
    rendue.
    """
    report = ArchiveReport()
    candidates = select_candidates(
        job_store, seen, include_rejected=include_rejected
    )
    report.candidates = len(candidates)

    for job_record, entry in candidates:
        if not record:
            report.archived.append(
                ArchivedApplication(
                    record=job_record,
                    directory=archive_path(config.paths.applications, job_record),
                )
            )
            continue

        try:
            archived = archive_and_save(
                config, job_record, entry, job_store=job_store, move=move
            )
        except OSError as exc:
            report.errors.append(f"{job_record.job.title}: archivage impossible ({exc})")
            logger.warning("archivage échoué (%s): %s", job_record.job.title, exc)
            continue
        report.archived.append(archived)

    return report


def archive_and_save(
    config: Config,
    record: JobRecord,
    entry: SeenEntry,
    *,
    job_store: JobStore,
    move: bool = True,
) -> ArchivedApplication:
    """Classe un dossier et enregistre où il est désormais.

    Commun à la passe d'archivage et à l'interface web, qui archive une offre à
    la fois. Lève `OSError` si le dossier ne peut pas être classé.
    """
    archived = archive_record(config.paths.applications, record, entry, move=move)
    # Le dossier a bougé : `outbox` doit suivre, sans quoi les commandes qui le
    # lisent (`sent`, `status`) désigneraient un chemin disparu.
    saved = job_store.save(
        record.job,
        archive=str(archived.directory),
        outbox=str(archived.directory) if archived.moved else None,
    )
    return ArchivedApplication(
        record=saved, directory=archived.directory, moved=archived.moved
    )
