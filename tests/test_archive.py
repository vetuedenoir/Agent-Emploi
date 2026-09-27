"""L'archivage : ce qui est classé, ce qui ne l'est pas, ce qui n'est jamais écrasé.

Deux garanties comptent ici. Une candidature ne quitte `outbox/` qu'après avoir
été copiée entière dans `applications/`. Et la fiche de suivi, où l'utilisateur
note ses relances, n'est écrite qu'une fois.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from agent_emploi.models import (
    FitVerdict,
    Job,
    JobState,
    Letter,
    UserDecision,
)
from agent_emploi.store.archive import (
    FOLLOWUP_FILE,
    archive_path,
    followup_markdown,
    run_archive,
    select_candidates,
)
from agent_emploi.store.jobs import JobStore
from agent_emploi.store.seen import SeenStore

SENT_AT = datetime(2026, 8, 11, 9, 30, tzinfo=timezone.utc)


def make_job(slug: str = "ml", title: str = "Ingénieur IA") -> Job:
    return Job.build(
        source="wttj",
        url=f"https://www.welcometothejungle.com/fr/companies/acme/jobs/{slug}",
        title=title,
        company="Acme",
        apply_url="https://ats.exemple/apply",
        ats="greenhouse",
        description="LLM, RAG.",
    )


def make_bundle(tmp_path, name: str = "2026-08-09_acme_ingenieur-ia"):
    """Un dossier `outbox/` complet, tel que `draft` puis `apply` le laissent."""
    directory = tmp_path / "outbox" / name
    directory.mkdir(parents=True)
    (directory / "lettre.md").write_text("Ma lettre relue.\n", encoding="utf-8")
    (directory / "cv.pdf").write_bytes(b"%PDF")
    (directory / "offre.md").write_text("# Ingénieur IA\n", encoding="utf-8")
    (directory / "preview.html").write_text("<html></html>", encoding="utf-8")
    return directory


#: Le parcours nominal d'une offre, du filtrage à l'envoi déclaré.
NOMINAL = [
    JobState.PRESCREENED,
    JobState.FIT_OK,
    JobState.DRAFTED,
    JobState.REVIEWED,
    JobState.AWAITING_USER,
    JobState.APPROVED,
    JobState.SUBMITTED,
]

_REASONS = {
    JobState.SUBMITTED: "utilisateur:envoyé",
    JobState.REJECTED: "utilisateur:rejeté — pas assez proche",
}


def prepare(tmp_path, config, *, state: JobState, job: Job | None = None,
            directory=None, submitted: bool = True):
    """Une offre menée jusqu'à `state`, avec son dossier et sa mémoire."""
    job = job or make_job()
    directory = directory or make_bundle(tmp_path)

    seen = SeenStore(config.paths.seen_file)
    seen.record(job)
    if state is JobState.REJECTED:
        # Un rejet ne se produit qu'une fois le dossier soumis à l'utilisateur.
        steps = NOMINAL[: NOMINAL.index(JobState.AWAITING_USER) + 1] + [state]
    else:
        steps = NOMINAL[: NOMINAL.index(state) + 1]
    for step in steps:
        seen.transition(job.id, step, _REASONS.get(step))

    job_store = JobStore(config.paths.jobs_file)
    job_store.save(
        job,
        fit=FitVerdict(score=78, verdict="apply", reason="profil aligné", language="fr"),
        letter=Letter(text="Ma lettre relue.", language="fr"),
        outbox=str(directory),
        decision=UserDecision(decision="approved", at=SENT_AT),
        submitted_at=SENT_AT if submitted else None,
    )
    return job, directory, seen, job_store


def test_seules_les_candidatures_envoyees_sont_classees(tmp_path, config):
    """Un dossier encore en attente de décision reste dans `outbox/`."""
    job, _, seen, job_store = prepare(
        tmp_path, config, state=JobState.AWAITING_USER, submitted=False
    )
    assert select_candidates(job_store, seen) == []

    seen.advance(job.id, JobState.APPROVED)
    assert select_candidates(job_store, seen) == []

    seen.advance(job.id, JobState.SUBMITTED, "utilisateur:envoyé")
    assert [record.job.id for record, _ in select_candidates(job_store, seen)] == [job.id]


def test_le_dossier_est_deplace_avec_toutes_ses_pieces(tmp_path, config):
    job, directory, seen, job_store = prepare(
        tmp_path, config, state=JobState.SUBMITTED
    )

    report = run_archive(config, seen=seen, job_store=job_store)

    assert len(report.archived) == 1
    archived = report.archived[0]
    assert archived.moved
    assert not directory.exists()
    for name in ("lettre.md", "cv.pdf", "offre.md", "preview.html", FOLLOWUP_FILE):
        assert (archived.directory / name).exists(), name
    assert (archived.directory / "lettre.md").read_text(encoding="utf-8") == (
        "Ma lettre relue.\n"
    )
    # Le chemin est enregistré, et `outbox` suit le déplacement : plus aucune
    # commande ne pointe vers un dossier disparu.
    saved = job_store.get(job.id)
    assert saved.archive == str(archived.directory)
    assert saved.outbox == str(archived.directory)


def test_le_dossier_est_date_du_jour_de_l_envoi(tmp_path, config):
    """C'est la date de clôture qu'on cherche en rouvrant l'archive, pas celle
    de préparation du dossier."""
    _, _, seen, job_store = prepare(tmp_path, config, state=JobState.SUBMITTED)

    report = run_archive(config, seen=seen, job_store=job_store)

    assert report.archived[0].directory.name.startswith("2026-08-11_acme_")


def test_la_fiche_de_suivi_n_est_jamais_reecrite(tmp_path, config):
    """L'utilisateur y note ses relances : le programme ne repasse pas dessus."""
    _, _, seen, job_store = prepare(tmp_path, config, state=JobState.SUBMITTED)
    report = run_archive(config, seen=seen, job_store=job_store)
    followup = report.archived[0].directory / FOLLOWUP_FILE

    followup.write_text("# Mes notes\n\n- relancé le 20/08\n", encoding="utf-8")

    # Une relance de la passe ne doit ni reclasser le dossier ni toucher la fiche.
    again = run_archive(config, seen=seen, job_store=job_store)
    assert again.archived == []
    assert "relancé le 20/08" in followup.read_text(encoding="utf-8")


def test_la_fiche_porte_la_date_de_relance(tmp_path, config):
    _, _, seen, job_store = prepare(tmp_path, config, state=JobState.SUBMITTED)
    record = job_store.records()[0]

    text = followup_markdown(
        record, seen.get(record.job.id), ["cv.pdf", "lettre.md", "offre.md"]
    )

    assert "Acme — Ingénieur IA" in text
    # Les pièces annoncées sont celles réellement classées, pas une liste fixe.
    assert "`lettre.md`" in text
    assert "decision.json" not in text
    assert "envoyée** le 11/08/2026" in text
    assert (SENT_AT + timedelta(days=14)).strftime("%d/%m/%Y") in text
    assert "78/100" in text
    assert "https://ats.exemple/apply" in text


def test_passe_a_blanc_ne_deplace_rien(tmp_path, config):
    _, directory, seen, job_store = prepare(tmp_path, config, state=JobState.SUBMITTED)

    report = run_archive(config, seen=seen, job_store=job_store, record=False)

    assert len(report.archived) == 1
    assert directory.exists()
    assert not config.paths.applications.exists() or not any(
        config.paths.applications.iterdir()
    )
    assert job_store.records()[0].archive is None


def test_keep_laisse_le_dossier_dans_outbox(tmp_path, config):
    _, directory, seen, job_store = prepare(tmp_path, config, state=JobState.SUBMITTED)

    report = run_archive(config, seen=seen, job_store=job_store, move=False)

    assert directory.exists()
    assert not report.archived[0].moved
    # `outbox` n'a pas bougé : les deux copies coexistent volontairement.
    assert job_store.get(report.archived[0].record.job.id).outbox == str(directory)


def test_les_rejetes_ne_sont_classes_que_sur_demande(tmp_path, config):
    _, _, seen, job_store = prepare(tmp_path, config, state=JobState.REJECTED)

    assert run_archive(config, seen=seen, job_store=job_store).archived == []

    report = run_archive(
        config, seen=seen, job_store=job_store, include_rejected=True
    )
    assert len(report.archived) == 1
    text = (report.archived[0].directory / FOLLOWUP_FILE).read_text(encoding="utf-8")
    assert "rejetée avant envoi" in text
    assert "Relancer" not in text


def test_deux_candidatures_homonymes_restent_distinctes(tmp_path, config):
    """Même entreprise, même intitulé, même jour : deux dossiers, pas un."""
    first = make_job(slug="ml-1")
    second = make_job(slug="ml-2")
    prepare(
        tmp_path,
        config,
        state=JobState.SUBMITTED,
        job=first,
        directory=make_bundle(tmp_path, "2026-08-09_acme_a"),
    )
    prepare(
        tmp_path,
        config,
        state=JobState.SUBMITTED,
        job=second,
        directory=make_bundle(tmp_path, "2026-08-09_acme_b"),
    )
    seen = SeenStore(config.paths.seen_file)
    job_store = JobStore(config.paths.jobs_file)

    report = run_archive(config, seen=seen, job_store=job_store)

    directories = {item.directory for item in report.archived}
    assert len(directories) == 2


def test_un_dossier_supprime_a_la_main_n_est_pas_ressuscite(tmp_path, config):
    _, directory, seen, job_store = prepare(tmp_path, config, state=JobState.SUBMITTED)
    import shutil

    shutil.rmtree(directory)

    report = run_archive(config, seen=seen, job_store=job_store)

    assert report.archived == []
    assert report.errors == []


def test_archive_path_utilise_la_meme_forme_que_outbox(tmp_path, config):
    _, _, _, job_store = prepare(tmp_path, config, state=JobState.SUBMITTED)
    record = job_store.records()[0]

    path = archive_path(config.paths.applications, record)

    assert path.parent == config.paths.applications
    assert path.name == "2026-08-11_acme_ingenieur-ia"


@pytest.mark.parametrize("state", [JobState.HANDOFF, JobState.PREFILLED])
def test_une_candidature_non_envoyee_reste_a_faire(tmp_path, config, state):
    """`handoff` et `prefilled`, hérités du remplissage automatique, ne sont
    pas des fins : la main est encore à vous."""
    job, _, seen, job_store = prepare(
        tmp_path, config, state=JobState.APPROVED, submitted=False
    )
    seen.advance(job.id, state)

    assert select_candidates(job_store, seen) == []
