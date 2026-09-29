"""Validation utilisateur : le point d'arrêt du système.

Toute la chaîne précédente aboutit à des dossiers `outbox/` en `awaiting_user`.
Rien n'ira plus loin sans une décision prise ici, à la main. C'est la seule
étape où le programme attend quelqu'un.

    python -m agent_emploi review           # passe interactive sur les dossiers
    python -m agent_emploi review --list    # liste sans rien décider
    python -m agent_emploi review <ref> --approve

Quatre issues, empruntées au plan : approuver, éditer, rejeter, remettre à plus
tard. « Plus tard » est le défaut implicite — quitter la passe ne décide de
rien, les dossiers non traités restent en attente.

Deux points de conception valent d'être explicités :

**`lettre.md` fait foi.** L'utilisateur corrige ce fichier, pas la base : à
l'approbation, la lettre est relue depuis le dossier et réécrite dans
`jobs.jsonl`. C'est sa version qui est archivée, et les formules
interdites sont revérifiées sur elle — une correction à la main peut en
réintroduire.

**Les réserves n'empêchent pas d'approuver.** Elles sont affichées avant la
question et conservées dans la décision : si l'utilisateur approuve une lettre
signalée, cela doit se voir après coup.
"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from agent_emploi.config import Config
from agent_emploi.models import (
    InvalidTransition,
    JobRecord,
    JobState,
    UserDecision,
    utcnow,
)
from agent_emploi.outbox import read_letter, refresh_preview, write_decision
from agent_emploi.profile import BannedPhrases
from agent_emploi.store.jobs import JobStore
from agent_emploi.store.seen import SeenStore

logger = logging.getLogger(__name__)

#: Longueur minimale d'une référence de dossier, pour que « a » ne désigne pas
#: la moitié de la file d'attente.
MIN_REF = 3

#: Réponse annoncée pour renoncer à un rejet en cours.
CANCEL = "q"

#: Réponses qui annulent le rejet. `q` y figure parce que c'est la touche pour
#: quitter la passe : la taper à l'invite du motif veut dire « je voulais
#: sortir », jamais « rejette avec le motif q ». Un motif n'a pas à s'écrire en
#: un seul caractère, la perte est nulle.
CANCEL_ANSWERS = frozenset({CANCEL, "annuler", "cancel", "non"})


@dataclass
class Pending:
    """Un dossier en attente de décision."""

    record: JobRecord
    directory: Path
    #: Réserves calculées à l'affichage : formules interdites, longueur, revue.
    concerns: list[str] = field(default_factory=list)

    @property
    def job_id(self) -> str:
        return self.record.job.id

    @property
    def label(self) -> str:
        return f"{self.record.job.company} — {self.record.job.title}"


@dataclass
class ReviewReport:
    """Bilan d'une passe de validation."""

    approved: list[Pending] = field(default_factory=list)
    rejected: list[Pending] = field(default_factory=list)
    #: Dossiers laissés en attente : sortie anticipée ou « plus tard » explicite.
    postponed: list[Pending] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def concerns(record: JobRecord, directory: Path, config: Config) -> list[str]:
    """Réserves à afficher avant de demander une décision.

    Recalculées plutôt que persistées : la lettre a pu être corrigée depuis la
    rédaction, et une réserve qui ne tient plus ne doit pas rester affichée.
    """
    issues: list[str] = []
    letter = record.letter
    if letter is None:
        return ["aucune lettre dans ce dossier"]

    issues += [f"formule interdite : {hit}" for hit in letter.banned_hits]
    words = letter.word_count
    if not config.letter.min_words <= words <= config.letter.max_words:
        issues.append(
            f"longueur : {words} mots (attendu {config.letter.min_words}"
            f"–{config.letter.max_words})"
        )

    review = record.review
    if review is not None:
        issues += [
            f"invention probable : {claim}" for claim in review.unsupported_claims
        ]
        if not review.approved:
            issues += [f"relecture : {issue}" for issue in review.issues] or [
                "relecture : lettre refusée sans motif détaillé"
            ]

    if not directory.exists():
        issues.append(f"dossier introuvable : {directory}")
    elif not any(directory.glob("cv.*")):
        issues.append("aucun CV joint dans le dossier")

    return issues


def pending(job_store: JobStore, seen: SeenStore, config: Config) -> list[Pending]:
    """Dossiers en attente de décision, les mieux notés d'abord.

    La mémoire fait foi sur l'état : `jobs.jsonl` garde aussi les dossiers déjà
    tranchés, qui ne doivent pas repasser devant l'utilisateur.
    """
    items: list[Pending] = []
    for record in job_store.records():
        if record.outbox is None:
            continue
        entry = seen.get(record.job.id)
        if entry is None or entry.state is not JobState.AWAITING_USER:
            continue
        directory = Path(record.outbox)
        items.append(
            Pending(
                record=record,
                directory=directory,
                concerns=concerns(record, directory, config),
            )
        )
    items.sort(
        key=lambda item: item.record.fit.score if item.record.fit else 0, reverse=True
    )
    return items


def resolve(items: list[Pending], ref: str) -> Pending:
    """Retrouve un dossier depuis un fragment d'identifiant ou de nom de dossier.

    Lève `LookupError` sur une référence inconnue ou ambiguë : décider du mauvais
    dossier serait pire que de redemander.
    """
    ref = ref.strip()
    if len(ref) < MIN_REF:
        raise LookupError(f"référence trop courte: {ref!r} ({MIN_REF} caractères mini)")

    needle = ref.lower()
    matches = [
        item
        for item in items
        if item.job_id.startswith(needle) or needle in item.directory.name.lower()
    ]
    if not matches:
        raise LookupError(f"aucun dossier en attente ne correspond à {ref!r}")
    if len(matches) > 1:
        names = ", ".join(item.directory.name for item in matches[:5])
        raise LookupError(f"référence ambiguë {ref!r} — {len(matches)} dossiers: {names}")
    return matches[0]


# --------------------------------------------------------------------- décisions


def collect_letter(item: Pending, banned: BannedPhrases) -> JobRecord:
    """Relit `lettre.md` et retourne le dossier à jour.

    La lettre relue porte `edited=True` : c'est ce drapeau, et non une
    comparaison ponctuelle, qui dit ensuite s'il faut la réécrire dans
    `jobs.jsonl` — une correction faite plus tôt dans la même passe ne doit pas
    se perdre au moment d'approuver.

    Le contrôle des formules interdites est refait sur le texte relu : c'est
    gratuit, et l'utilisateur qui réécrit une phrase peut en réintroduire une.
    """
    record = item.record
    letter = record.letter
    if letter is None:
        return record

    text = read_letter(item.directory)
    if text is None:
        logger.warning(
            "lettre.md absente de %s — version enregistrée conservée", item.directory
        )
        return record
    if text == letter.text.strip():
        return record

    edited = letter.model_copy(
        update={"text": text, "banned_hits": banned.find(text), "edited": True}
    )
    return record.model_copy(update={"letter": edited})


def reload(item: Pending, banned: BannedPhrases, config: Config) -> Pending:
    """Relit le dossier depuis le disque : lettre à jour, réserves recalculées.

    Les réserves ne sont pas conservées d'un affichage à l'autre : une longueur
    corrigée ou une formule retirée doit cesser d'être signalée, et une formule
    réintroduite à la main doit apparaître.
    """
    record = collect_letter(item, banned)
    return Pending(
        record=record,
        directory=item.directory,
        concerns=concerns(record, item.directory, config),
    )


def approve(
    item: Pending,
    *,
    seen: SeenStore,
    job_store: JobStore,
    banned: BannedPhrases,
    config: Config,
    note: str | None = None,
) -> Pending:
    """Approuve un dossier : lettre relue, décision persistée, état `approved`.

    L'ordre importe. La lettre et la décision sont écrites avant la transition
    d'état : une interruption entre les deux laisse le dossier en attente — donc
    à revalider — plutôt qu'approuvé avec une lettre périmée.
    """
    record = collect_letter(item, banned)
    was_edited = record.letter is not None and record.letter.edited
    remaining = concerns(record, item.directory, config)

    decision = UserDecision(
        decision="approved",
        at=utcnow(),
        note=note,
        concerns=remaining,
        letter_edited=was_edited,
    )
    saved = job_store.save(
        record.job,
        letter=record.letter if was_edited else None,
        decision=decision,
    )
    if item.directory.exists():
        write_decision(item.directory, decision)
        # Le dossier part tel quel à l'archivage : sa preview doit montrer la
        # lettre approuvée, pas celle qu'avait produite le modèle.
        refresh_preview(item.directory, record)

    seen.transition(
        item.job_id,
        JobState.APPROVED,
        "utilisateur:approuvé" + (" (lettre corrigée)" if was_edited else ""),
    )
    return Pending(record=saved, directory=item.directory, concerns=remaining)


def reject(
    item: Pending,
    *,
    seen: SeenStore,
    job_store: JobStore,
    note: str | None = None,
) -> Pending:
    """Rejette un dossier. Le motif est libre mais vivement utile.

    Il est conservé dans `seen.jsonl` : c'est en relisant ces motifs qu'on règle
    les filtres, exactement comme pour les rejets automatiques.
    """
    decision = UserDecision(
        decision="rejected", at=utcnow(), note=note, concerns=item.concerns
    )
    saved = job_store.save(item.record.job, decision=decision)
    if item.directory.exists():
        write_decision(item.directory, decision)

    reason = f"utilisateur:{note}" if note else "utilisateur:rejeté"
    seen.transition(item.job_id, JobState.REJECTED, reason[:200])
    return Pending(record=saved, directory=item.directory, concerns=item.concerns)


# ------------------------------------------------------------------- interaction


#: Éditeurs cherchés dans le `PATH` quand ni `$VISUAL` ni `$EDITOR` n'est
#: défini. L'ordre va du plus abordable au plus exigeant : personne ne doit
#: découvrir `vi` sans l'avoir demandé, mais rester coincé sans éditeur est
#: pire — la lettre s'édite alors dans un autre terminal, et le menu tourne à
#: vide.
FALLBACK_EDITORS = ("nano", "micro", "nvim", "vim", "vi")


def find_editor() -> str | None:
    """Commande d'édition à lancer : le choix de l'utilisateur, ou un défaut."""
    chosen = os.environ.get("VISUAL") or os.environ.get("EDITOR")
    if chosen:
        return chosen
    return next(
        (name for name in FALLBACK_EDITORS if shutil.which(name)),
        None,
    )


def open_editor(path: Path) -> str | None:
    """Ouvre un éditeur sur un fichier. Retourne un message d'erreur, ou `None`.

    `$VISUAL`/`$EDITOR` d'abord — c'est le choix de l'utilisateur. À défaut, le
    premier éditeur courant trouvé dans le `PATH` : sans cela, l'entrée « e »
    du menu ne fait rien du tout sur une machine où ces variables ne sont pas
    exportées, ce qui est le cas par défaut sous zsh.
    """
    editor = find_editor()
    if not editor:
        return (
            f"aucun éditeur trouvé (ni $VISUAL, ni $EDITOR, ni {'/'.join(FALLBACK_EDITORS)})"
            f" — éditez {path} à la main"
        )
    try:
        subprocess.run([*shlex.split(editor), str(path)], check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"éditeur {editor!r} en échec ({exc}) — éditez {path} à la main"
    return None


@dataclass
class Console:
    """Les entrées-sorties de la passe, isolées pour rester testables.

    La boucle de validation ne touche ni à `input`, ni à `print`, ni au
    navigateur : tout passe par ici, ce qui permet de la dérouler entièrement
    dans un test avec des réponses écrites d'avance.
    """

    write: Callable[[str], None] = print
    ask: Callable[[str], str] = input
    edit: Callable[[Path], str | None] = open_editor
    open_url: Callable[[str], None] = webbrowser.open

    def blank(self) -> None:
        self.write("")


#: Actions du menu : touche -> (libellé, aide). L'ordre est celui de l'affichage.
ACTIONS: dict[str, str] = {
    "a": "approuver",
    "e": "éditer la lettre",
    "o": "ouvrir preview.html",
    "r": "rejeter",
    "p": "plus tard",
    "q": "quitter",
}


def show(item: Pending, console: Console, *, position: str = "") -> None:
    """Affiche un dossier : de quoi décider sans ouvrir autre chose."""
    record = item.record
    job, letter, fit = record.job, record.letter, record.fit

    console.blank()
    console.write("─" * 72)
    header = f"{item.label}"
    if position:
        header = f"[{position}] {header}"
    console.write(header)
    if fit is not None:
        variant = f" · CV « {fit.cv} »" if fit.cv else ""
        console.write(f"  adéquation {fit.score}/100{variant} — {fit.reason}")
    console.write(f"  annonce : {job.url}")
    if job.apply_url:
        ats = f" ({job.ats})" if job.ats else ""
        console.write(f"  candidature : {job.apply_url}{ats}")
    console.write(f"  dossier : {item.directory}")

    if letter is not None:
        console.blank()
        console.write(f"Lettre — {letter.word_count} mots ({letter.language})"
                      + (" · corrigée à la main" if letter.edited else ""))
        console.write("")
        for line in letter.text.strip().splitlines():
            console.write(f"  {line}")

    if item.concerns:
        console.blank()
        console.write("Réserves :")
        for concern in item.concerns:
            console.write(f"  ⚠  {concern}")
    console.blank()


def ask_action(console: Console) -> str:
    """Demande une action jusqu'à en obtenir une valide. `q` en fin d'entrée."""
    menu = "  ".join(f"[{key}] {label}" for key, label in ACTIONS.items())
    while True:
        try:
            answer = console.ask(f"{menu}\n> ").strip().lower()
        except EOFError:
            # Entrée fermée (redirection, Ctrl-D) : on quitte sans rien décider.
            return "q"
        if answer in ACTIONS:
            return answer
        if answer:
            console.write(f"réponse inconnue : {answer!r}")


def run_review(
    config: Config,
    *,
    seen: SeenStore,
    job_store: JobStore,
    banned: BannedPhrases,
    console: Console | None = None,
    only: str | None = None,
) -> ReviewReport:
    """Déroule la passe de validation, dossier par dossier.

    `only` restreint la passe à un dossier désigné par un fragment
    d'identifiant ou de nom. Un dossier laissé de côté reste en attente : c'est
    l'issue par défaut, y compris en cas de sortie anticipée.
    """
    console = console or Console()
    report = ReviewReport()

    queue = pending(job_store, seen, config)
    if only is not None:
        try:
            queue = [resolve(queue, only)]
        except LookupError as exc:
            report.errors.append(str(exc))
            return report

    total = len(queue)
    index = 0
    while index < total:
        item = queue[index]
        show(item, console, position=f"{index + 1}/{total}")
        action = ask_action(console)

        if action == "q":
            report.postponed.extend(queue[index:])
            break

        if action == "p":
            report.postponed.append(item)
            index += 1
            continue

        if action == "o":
            # La preview est écrite une seule fois, à la rédaction. Elle est
            # donc réécrite depuis `lettre.md` avant d'être ouverte : c'est ce
            # fichier qui fait foi, y compris quand il a été corrigé hors de
            # cette boucle — dans un autre terminal, ou lors d'une passe
            # précédente. Sans cela, on relit la version du modèle en croyant
            # relire la sienne.
            queue[index] = item = reload(item, banned, config)
            refresh_preview(item.directory, item.record)
            preview = item.directory / "preview.html"
            if preview.exists():
                # `outbox` est un chemin relatif dans la configuration, et un
                # chemin relatif n'a pas d'URI : sans `resolve()`, ouvrir la
                # preview lève au lieu d'ouvrir quoi que ce soit.
                # `webbrowser.open` renvoie faux quand il n'a trouvé aucun
                # navigateur à lancer : sans ce test, l'échec est silencieux.
                if console.open_url(preview.resolve().as_uri()) is False:
                    console.write(f"aucun navigateur lancé — ouvrez {preview}")
            else:
                console.write(f"preview.html introuvable dans {item.directory}")
            continue

        if action == "e":
            error = console.edit(item.directory / "lettre.md")
            if error:
                console.write(error)
                continue
            # Le dossier est rechargé : la lettre corrigée est relue, et les
            # réserves recalculées sur elle avant de redemander une décision.
            queue[index] = item = reload(item, banned, config)
            # La preview est réécrite dans la foulée : elle date de la
            # rédaction, et laissée telle quelle elle afficherait encore la
            # version du modèle à qui vient de la corriger.
            refresh_preview(item.directory, item.record)
            continue

        note: str | None = None
        if action == "r":
            # Le rejet est irréversible : `ALLOWED_TRANSITIONS[REJECTED]` est
            # vide, aucun chemin n'en sort. L'invite doit donc offrir une
            # sortie, et la nommer. Sans cela, un `q` tapé ici pour quitter la
            # passe est enregistré comme motif de rejet — la décision, elle,
            # était déjà prise en appuyant sur `r`.
            note = console.ask(f"motif (facultatif, « {CANCEL} » pour annuler) > ").strip()
            if note.lower() in CANCEL_ANSWERS:
                console.write("rejet annulé — le dossier reste en attente")
                continue
            note = note or None

        try:
            if action == "a":
                decided = approve(
                    item,
                    seen=seen,
                    job_store=job_store,
                    banned=banned,
                    config=config,
                    note=note,
                )
                report.approved.append(decided)
                console.write(f"✓ approuvé — {item.label}")
            else:
                decided = reject(item, seen=seen, job_store=job_store, note=note)
                report.rejected.append(decided)
                console.write(f"✗ rejeté — {item.label}")
        except (InvalidTransition, KeyError, OSError) as exc:
            # Un dossier qui résiste ne doit pas arrêter la file : il reste en
            # attente et l'erreur remonte dans le bilan.
            report.errors.append(f"{item.label}: {exc}")
            report.postponed.append(item)

        index += 1

    return report
