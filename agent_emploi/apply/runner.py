"""Passe de candidature assistée : des dossiers approuvés aux formulaires prêts.

Pour chaque offre en `approved`, les mieux notées d'abord :

1. ouverture de l'URL de candidature dans le navigateur ;
2. lecture des champs du formulaire ;
3. appariement déterministe champ ↔ information connue (`fields.py`) ;
4. remplissage, téléversement du CV, collage de la lettre ;
5. capture d'écran, fiche `candidature.md`, état `prefilled` ;
6. **arrêt.** L'onglet reste ouvert, l'utilisateur vérifie et envoie lui-même.

Ce que le code ne fait jamais : cliquer sur « envoyer ». Il n'existe aucune
fonction pour cela dans `browser.py`, donc aucun réglage, aucun argument et
aucun bogue ne peut y conduire.

La lettre et le CV sont pris **dans le dossier `outbox/`**, pas dans la base :
c'est la version que l'utilisateur a relue et approuvée qui doit partir, y
compris s'il a corrigé `lettre.md` à la main juste avant.

La passe est reprenable. Une offre dont le formulaire a été rempli passe en
`prefilled` ; une offre qui a rendu la main passe en `handoff`. Dans les deux
cas l'issue est écrite dans `jobs.jsonl` et dans le dossier, et l'utilisateur
déclare ensuite lui-même ce qu'il a envoyé — c'est la seule façon honnête de
connaître l'état réel d'une candidature qu'on n'a pas envoyée soi-même.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from agent_emploi.apply.browser import Page
from agent_emploi.apply.fields import FormField, Plan, Slot, build_plan
from agent_emploi.apply.handoff import SCREENSHOT_FILE, write_handoff
from agent_emploi.apply.identity import Identity
from agent_emploi.config import Config
from agent_emploi.models import ApplyOutcome, JobRecord, JobState, utcnow
from agent_emploi.outbox import read_letter
from agent_emploi.store.jobs import JobStore
from agent_emploi.store.seen import SeenStore

logger = logging.getLogger(__name__)

#: Un formulaire de candidature qui expose plus de champs que cela n'en est
#: plus un : page de recherche, tableau de bord, mur de connexion.
MAX_FIELDS = 40

#: Recours d'appariement : (champs, emplacements disponibles, id de l'offre)
#: -> {sélecteur: emplacement}. Injecté plutôt qu'importé, pour que la passe
#: se déroule à l'identique sans LLM (voir `mapping.guess_slots`).
Guess = Callable[[list[FormField], list[Slot], str], dict[str, Slot]]


@dataclass
class PreparedApplication:
    """Une candidature préparée dans le navigateur, prête à être vérifiée."""

    record: JobRecord
    directory: Path
    outcome: ApplyOutcome

    @property
    def label(self) -> str:
        return f"{self.record.job.company} — {self.record.job.title}"

    @property
    def ok(self) -> bool:
        return self.outcome.status == "prefilled"


@dataclass
class ApplyReport:
    """Bilan d'une passe de candidature."""

    candidates: int = 0
    prepared: list[PreparedApplication] = field(default_factory=list)
    handoffs: list[PreparedApplication] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    #: Motif d'interruption anticipée (navigateur indisponible, plafond).
    stopped: str | None = None

    @property
    def touched(self) -> list[PreparedApplication]:
        return self.prepared + self.handoffs


def select_candidates(
    job_store: JobStore, seen: SeenStore, *, limit: int | None = None
) -> list[JobRecord]:
    """Offres approuvées par l'utilisateur, les mieux notées d'abord.

    La mémoire fait foi : seul l'état `approved` ouvre cette passe, et il ne
    s'obtient que par une décision prise dans `review`.
    """
    ready = [
        record
        for record in job_store.records()
        if record.outbox is not None
        and (entry := seen.get(record.job.id)) is not None
        and entry.state is JobState.APPROVED
    ]
    ready.sort(key=lambda record: record.fit.score if record.fit else 0, reverse=True)
    return ready[:limit] if limit is not None else ready


def bundle_values(
    record: JobRecord, directory: Path, identity: Identity
) -> dict[Slot, str]:
    """Ce que l'on sait poser dans un formulaire, pour cette offre.

    Le CV et la lettre viennent du dossier validé, pas de la base : c'est la
    version relue par l'utilisateur qui doit partir. La lettre enregistrée ne
    sert que de repli, si `lettre.md` a disparu du dossier.
    """
    values: dict[Slot, str] = dict(identity.values())

    text = read_letter(directory)
    if not text and record.letter is not None:
        text = record.letter.text.strip()
    if text:
        values[Slot.COVER_LETTER] = text

    for candidate in sorted(directory.glob("cv.*")):
        values[Slot.CV] = str(candidate.resolve())
        break

    return values


def fill_form(page: Page, plan: Plan) -> tuple[list[str], list[str]]:
    """Exécute le plan de remplissage. Retourne (champs remplis, échecs).

    Un champ qui résiste n'interrompt pas les autres : le formulaire à moitié
    rempli reste utile à l'utilisateur, et le champ manquant apparaît dans la
    fiche comme dans le navigateur.
    """
    filled: list[str] = []
    failures: list[str] = []

    for assignment in plan.assignments:
        label = assignment.field.display
        try:
            if assignment.is_file:
                path = Path(assignment.value)
                if not path.exists():
                    failures.append(f"{label} : fichier introuvable ({path})")
                    continue
                page.attach(assignment.field, path)
            else:
                page.fill(assignment.field, assignment.value)
        except Exception as exc:  # noqa: BLE001 — un champ récalcitrant, pas une panne
            logger.warning("champ %r non rempli: %s", label, exc)
            failures.append(f"{label} : non rempli ({type(exc).__name__})")
            continue
        filled.append(label)

    return filled, failures


def prepare(
    record: JobRecord,
    page: Page,
    values: dict[Slot, str],
    directory: Path,
    *,
    guess: Guess | None = None,
) -> tuple[ApplyOutcome, Plan | None]:
    """Remplit le formulaire d'une offre et rend son issue.

    L'ordre est celui de la prudence : les blocages sont cherchés avant le
    remplissage — inutile de poser un nom et un téléphone sur une page qui
    exige d'abord une connexion.

    `guess` n'est sollicité que si les motifs ont laissé un champ obligatoire
    sans réponse : tant que le déterministe suffit, aucun appel LLM n'a lieu.
    """
    job = record.job
    apply_url = job.apply_url or str(job.url)

    def outcome(status, **kwargs) -> ApplyOutcome:
        return ApplyOutcome(
            status=status, apply_url=apply_url, ats=job.ats, at=utcnow(), **kwargs
        )

    blockers = page.blockers()
    if blockers:
        return outcome("handoff", blockers=blockers), None

    fields = page.fields()
    if not fields:
        return outcome("handoff", blockers=["aucun champ de formulaire sur la page"]), None
    if len(fields) > MAX_FIELDS:
        return (
            outcome(
                "handoff",
                blockers=[
                    f"{len(fields)} champs sur la page — ce n'est probablement "
                    "pas un formulaire de candidature"
                ],
            ),
            None,
        )

    plan = build_plan(fields, values)
    if not plan.ok and guess is not None:
        # Les motifs ont buté sur un libellé inattendu. Un appel du tier
        # gratuit peut encore rattacher ces champs — sans jamais fournir de
        # valeur, seulement en désignant l'information attendue.
        assigned = {item.slot for item in plan.assignments}
        available = [
            slot for slot in values if slot is not Slot.CV and slot not in assigned
        ]
        overrides = guess(fields, available, job.id)
        if overrides:
            plan = build_plan(fields, values, overrides)

    if not plan.ok:
        # Un champ obligatoire qu'on ne sait pas remplir : on ne remplit rien
        # du tout. Un formulaire à moitié rempli qu'on ne peut pas terminer est
        # plus déroutant qu'une page vierge accompagnée de la fiche.
        return outcome("handoff", blockers=plan.blocking, todo=plan.todo), plan

    filled, failures = fill_form(page, plan)
    screenshot = page.screenshot(directory / SCREENSHOT_FILE)

    return (
        outcome(
            "prefilled",
            filled=filled,
            todo=plan.todo + failures,
            screenshot=str(screenshot) if screenshot else None,
        ),
        plan,
    )


def run_apply(
    config: Config,
    *,
    browser,
    seen: SeenStore,
    job_store: JobStore,
    identity: Identity,
    limit: int | None = None,
    record: bool = True,
    only: str | None = None,
    guess: Guess | None = None,
) -> ApplyReport:
    """Prépare les candidatures approuvées, sans jamais en envoyer aucune.

    `record=False` fait une passe à blanc : le navigateur ouvre bien les
    formulaires et les remplit — c'est le seul moyen de voir ce que ça donne —
    mais ni la mémoire, ni `jobs.jsonl`, ni les dossiers ne sont touchés.

    `guess=None` désactive le recours LLM : la passe se déroule alors
    entièrement sur les motifs déterministes.
    """
    report = ApplyReport()
    candidates = select_candidates(job_store, seen, limit=limit)
    if only is not None:
        needle = only.strip().lower()
        candidates = [
            item
            for item in candidates
            if item.job.id.startswith(needle)
            or needle in Path(item.outbox or "").name.lower()
        ]
        if not candidates:
            report.errors.append(f"aucun dossier approuvé ne correspond à {only!r}")
            return report
    report.candidates = len(candidates)

    for job_record in candidates:
        job = job_record.job
        directory = Path(job_record.outbox or "")
        apply_url = job.apply_url or str(job.url)

        if not directory.exists():
            report.errors.append(f"{job.title}: dossier introuvable ({directory})")
            continue

        values = bundle_values(job_record, directory, identity)
        if Slot.CV not in values:
            logger.warning("aucun CV dans %s — le champ restera à joindre", directory)

        try:
            page = browser.open(apply_url)
        except Exception as exc:  # noqa: BLE001 — page injoignable, offre suivante
            report.errors.append(f"{job.title}: page inaccessible ({exc})")
            outcome = ApplyOutcome(
                status="handoff",
                apply_url=apply_url,
                ats=job.ats,
                blockers=[f"page inaccessible : {exc}"],
            )
            plan = None
        else:
            outcome, plan = prepare(job_record, page, values, directory, guess=guess)

        item = PreparedApplication(
            record=job_record, directory=directory, outcome=outcome
        )

        try:
            write_handoff(directory, job_record, outcome, plan=plan, values=values)
        except OSError as exc:
            report.errors.append(f"{job.title}: fiche non écrite ({exc})")

        if record:
            saved = job_store.save(job, application=outcome)
            item = PreparedApplication(
                record=saved, directory=directory, outcome=outcome
            )
            target = (
                JobState.PREFILLED if outcome.status == "prefilled" else JobState.HANDOFF
            )
            reason = (
                f"formulaire:{len(outcome.filled)} champs remplis"
                if outcome.status == "prefilled"
                else f"handoff:{outcome.blockers[0] if outcome.blockers else 'inconnu'}"
            )
            seen.advance(job.id, target, reason[:200])

        (report.prepared if item.ok else report.handoffs).append(item)

    return report


def mark_submitted(
    job_id: str,
    *,
    seen: SeenStore,
    job_store: JobStore,
    note: str | None = None,
) -> JobRecord | None:
    """Enregistre qu'une candidature a bien été envoyée — par l'utilisateur.

    C'est la seule façon dont une offre atteint `submitted` : le programme
    n'envoie rien, il ne peut donc que prendre acte d'une déclaration humaine.
    """
    record = job_store.get(job_id)
    if record is None or record.application is None:
        return None

    outcome = record.application.model_copy(update={"submitted": True, "at": utcnow()})
    saved = job_store.save(record.job, application=outcome)
    seen.advance(job_id, JobState.SUBMITTED, f"utilisateur:envoyé{f' — {note}' if note else ''}"[:200])
    return saved
