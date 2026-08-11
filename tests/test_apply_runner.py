"""La passe de candidature, déroulée sans navigateur.

Le protocole `Page` de `apply/browser.py` tient en cinq méthodes : une page
factice suffit à vérifier tout ce qui compte — le remplissage, le chemin
`handoff`, les états, et surtout qu'aucun envoi n'a lieu.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_emploi.apply.fields import FormField, Slot
from agent_emploi.apply.identity import Identity
from agent_emploi.apply.runner import (
    bundle_values,
    mark_submitted,
    run_apply,
    select_candidates,
)
from agent_emploi.models import (
    FitVerdict,
    Job,
    JobState,
    Letter,
    UserDecision,
)
from agent_emploi.store.jobs import JobStore
from agent_emploi.store.seen import SeenStore


class FakePage:
    """Une page de formulaire scriptée. Elle n'expose aucun moyen d'envoyer."""

    def __init__(self, fields, blockers=None, url="https://ats.exemple/apply"):
        self.url = url
        self._fields = fields
        self._blockers = blockers or []
        self.filled: dict[str, str] = {}
        self.attached: dict[str, Path] = {}
        self.screenshots: list[Path] = []

    def fields(self):
        return list(self._fields)

    def fill(self, field, value):
        self.filled[field.selector] = value

    def attach(self, field, path):
        self.attached[field.selector] = path

    def screenshot(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"PNG")
        self.screenshots.append(path)
        return path

    def blockers(self):
        return list(self._blockers)


class FakeBrowser:
    def __init__(self, page):
        self.page = page
        self.opened: list[str] = []

    def open(self, url):
        self.opened.append(url)
        return self.page


@pytest.fixture
def identity() -> Identity:
    return Identity(
        first_name="Killian",
        last_name="Scordel",
        email="killian@exemple.fr",
        phone="+33612345678",
    )


def make_job() -> Job:
    return Job.build(
        source="wttj",
        url="https://www.welcometothejungle.com/fr/companies/acme/jobs/ml",
        title="Ingénieur IA",
        company="Acme",
        apply_url="https://ats.exemple/apply",
        ats="greenhouse",
        description="LLM, RAG.",
    )


@pytest.fixture
def approved(tmp_path, config):
    """Une offre approuvée, avec son dossier `outbox/` complet."""
    job = make_job()
    directory = tmp_path / "outbox" / "2026-08-11_acme_ingenieur-ia"
    directory.mkdir(parents=True)
    (directory / "lettre.md").write_text(
        "<!-- en-tête -->\n\nMadame, Monsieur, voici ma lettre relue.\n",
        encoding="utf-8",
    )
    (directory / "cv.pdf").write_bytes(b"%PDF")

    seen = SeenStore(config.paths.seen_file)
    seen.record(job)
    for state in (
        JobState.PRESCREENED,
        JobState.FIT_OK,
        JobState.DRAFTED,
        JobState.REVIEWED,
        JobState.AWAITING_USER,
        JobState.APPROVED,
    ):
        seen.transition(job.id, state)

    job_store = JobStore(config.paths.jobs_file)
    job_store.save(
        job,
        fit=FitVerdict(score=80, verdict="apply", reason="ok", language="fr"),
        letter=Letter(text="Madame, Monsieur, voici ma lettre.", language="fr"),
        outbox=str(directory),
        decision=UserDecision(decision="approved"),
    )
    return job, directory, seen, job_store


def text_field(selector, label, **kwargs):
    return FormField(selector=selector, kind="text", label=label, **kwargs)


def test_seules_les_offres_approuvees_sont_candidates(approved, config):
    job, _, seen, job_store = approved
    assert [record.job.id for record in select_candidates(job_store, seen)] == [job.id]

    seen.transition(job.id, JobState.PREFILLED)
    assert select_candidates(job_store, seen) == []


def test_la_lettre_vient_du_dossier_pas_de_la_base(approved, identity):
    """`lettre.md` fait foi : c'est la version relue par l'utilisateur."""
    _, directory, _, job_store = approved
    record = job_store.records()[0]

    values = bundle_values(record, directory, identity)

    assert values[Slot.COVER_LETTER] == "Madame, Monsieur, voici ma lettre relue."
    assert values[Slot.CV] == str((directory / "cv.pdf").resolve())


def test_formulaire_rempli_puis_arret(approved, config, identity):
    job, directory, seen, job_store = approved
    page = FakePage(
        [
            text_field("s1", "Prénom", required=True),
            text_field("s2", "Nom", required=True),
            text_field("s3", "Email", required=True),
            FormField(selector="s4", kind="file", label="CV", required=True),
            FormField(selector="s5", kind="textarea", label="Lettre de motivation"),
        ]
    )
    browser = FakeBrowser(page)

    report = run_apply(
        config, browser=browser, seen=seen, job_store=job_store, identity=identity
    )

    assert browser.opened == ["https://ats.exemple/apply"]
    assert len(report.prepared) == 1
    assert report.handoffs == []
    assert page.filled["s1"] == "Killian"
    assert page.filled["s3"] == "killian@exemple.fr"
    assert page.filled["s5"] == "Madame, Monsieur, voici ma lettre relue."
    assert page.attached["s4"] == (directory / "cv.pdf").resolve()

    outcome = report.prepared[0].outcome
    assert outcome.status == "prefilled"
    assert outcome.submitted is False
    assert seen.get(job.id).state is JobState.PREFILLED
    assert (directory / "formulaire.png").exists()
    assert (directory / "candidature.md").exists()


def test_champ_obligatoire_inconnu_rend_la_main_sans_rien_remplir(
    approved, config, identity
):
    """Un formulaire à moitié rempli qu'on ne peut pas finir est pire que rien."""
    job, directory, seen, job_store = approved
    page = FakePage(
        [
            text_field("s1", "Prénom", required=True),
            text_field("s2", "Combien de brouettes ?", required=True),
        ]
    )

    report = run_apply(
        config,
        browser=FakeBrowser(page),
        seen=seen,
        job_store=job_store,
        identity=identity,
    )

    assert page.filled == {}
    assert len(report.handoffs) == 1
    assert report.handoffs[0].outcome.status == "handoff"
    assert seen.get(job.id).state is JobState.HANDOFF
    fiche = (directory / "candidature.md").read_text(encoding="utf-8")
    assert "Combien de brouettes" in fiche
    assert "voici ma lettre relue" in fiche  # de quoi candidater à la main


def test_captcha_rend_la_main_avant_de_rien_taper(approved, config, identity):
    job, _, seen, job_store = approved
    page = FakePage([text_field("s1", "Prénom")], blockers=["captcha détecté"])

    report = run_apply(
        config,
        browser=FakeBrowser(page),
        seen=seen,
        job_store=job_store,
        identity=identity,
    )

    assert page.filled == {}
    assert report.handoffs[0].outcome.blockers == ["captcha détecté"]
    assert seen.get(job.id).state is JobState.HANDOFF


def test_page_inaccessible_ne_stoppe_pas_la_passe(approved, config, identity):
    job, _, seen, job_store = approved

    class Broken:
        def open(self, url):
            raise RuntimeError("timeout")

    report = run_apply(
        config, browser=Broken(), seen=seen, job_store=job_store, identity=identity
    )

    assert report.errors and "timeout" in report.errors[0]
    assert seen.get(job.id).state is JobState.HANDOFF


def test_passe_a_blanc_nenregistre_rien(approved, config, identity):
    job, _, seen, job_store = approved
    page = FakePage([text_field("s1", "Prénom")])

    report = run_apply(
        config,
        browser=FakeBrowser(page),
        seen=seen,
        job_store=job_store,
        identity=identity,
        record=False,
    )

    assert page.filled["s1"] == "Killian"
    assert len(report.prepared) == 1
    assert seen.get(job.id).state is JobState.APPROVED
    assert job_store.get(job.id).application is None


def test_envoi_declare_par_lutilisateur(approved, config, identity):
    """`submitted` ne s'obtient que par une déclaration humaine."""
    job, _, seen, job_store = approved
    run_apply(
        config,
        browser=FakeBrowser(FakePage([text_field("s1", "Prénom")])),
        seen=seen,
        job_store=job_store,
        identity=identity,
    )
    assert job_store.get(job.id).application.submitted is False

    updated = mark_submitted(job.id, seen=seen, job_store=job_store, note="via ATS")

    assert updated.application.submitted is True
    assert seen.get(job.id).state is JobState.SUBMITTED


def test_le_navigateur_nexpose_aucun_moyen_denvoyer():
    """Garantie structurelle : rien à appeler, donc rien à cliquer."""
    from agent_emploi.apply import browser, runner

    for module in (browser, runner):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "def submit" not in source
        assert ".click(" not in source
