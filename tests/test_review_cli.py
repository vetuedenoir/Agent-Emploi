import json

import pytest

from agent_emploi.models import (
    FitVerdict,
    Job,
    JobState,
    Letter,
    ReviewVerdict,
)
from agent_emploi.outbox import read_letter, write_bundle
from agent_emploi.profile import BannedPhrases
from agent_emploi.review_cli import (
    Console,
    approve,
    concerns,
    pending,
    reject,
    resolve,
    run_review,
)
from agent_emploi.store.jobs import JobStore
from agent_emploi.store.seen import SeenStore

LETTRE = " ".join(["mot"] * 170)


def make_job(n: int = 1, **fields) -> Job:
    base = {
        "title": f"Ingénieur IA {n}",
        "company": f"Acme {n}",
        "description": "Agents LLM en Python.",
        "apply_url": f"https://ats.example.com/{n}",
    }
    return Job.build(
        source="fake", url=f"https://example.com/jobs/{n}", **{**base, **fields}
    )


def fit(**fields) -> FitVerdict:
    base = {
        "score": 80,
        "verdict": "apply",
        "matched": ["Python"],
        "gaps": [],
        "reason": "ok",
        "language": "fr",
    }
    return FitVerdict.model_validate({**base, **fields})


class ScriptedConsole(Console):
    """Console de test : des réponses écrites d'avance, la sortie capturée."""

    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.lines: list[str] = []
        self.edited: list = []
        self.opened: list[str] = []
        super().__init__(
            write=self.lines.append,
            ask=self._ask,
            edit=self._edit,
            open_url=self.opened.append,
        )
        #: Texte injecté dans `lettre.md` quand l'action « éditer » est jouée.
        self.editor_writes: str | None = None

    def _ask(self, prompt: str) -> str:
        if not self.answers:
            raise AssertionError(f"réponse non prévue par le scénario: {prompt!r}")
        return self.answers.pop(0)

    def _edit(self, path):
        self.edited.append(path)
        if self.editor_writes is not None:
            path.write_text(self.editor_writes, encoding="utf-8")
        return None

    @property
    def output(self) -> str:
        return "\n".join(self.lines)


@pytest.fixture
def wired(config, tmp_path, profile):
    """Un dossier écrit dans outbox/, l'offre en attente de validation."""
    config.paths.ensure()
    seen = SeenStore(tmp_path / "seen.jsonl")
    job_store = JobStore(tmp_path / "jobs.jsonl")
    return config, seen, job_store, profile


def enqueue(wired, job: Job, *, letter_text: str = LETTRE, review=None, **fields):
    """Amène une offre jusqu'en `awaiting_user`, dossier écrit sur disque."""
    config, seen, job_store, profile = wired
    seen.record(job)
    for target in (
        JobState.PRESCREENED,
        JobState.FIT_OK,
        JobState.DRAFTED,
        JobState.REVIEWED,
        JobState.AWAITING_USER,
    ):
        seen.advance(job.id, target)

    record = job_store.save(
        job,
        fit=fields.pop("fit", fit()),
        letter=Letter(text=letter_text, language="fr", **fields),
        review=review or ReviewVerdict(approved=True),
    )
    directory = write_bundle(config.paths.outbox, record, profile.cv_fr)
    job_store.save(job, outbox=str(directory))
    return job, directory


class TestPending:
    def test_lists_bundles_awaiting_a_decision_best_scored_first(self, wired):
        _, seen, job_store, _ = wired
        enqueue(wired, make_job(1), fit=fit(score=70))
        enqueue(wired, make_job(2), fit=fit(score=90))
        queue = pending(job_store, seen, wired[0])
        assert [item.record.fit.score for item in queue] == [90, 70]

    def test_ignores_offers_already_decided(self, wired):
        config, seen, job_store, _ = wired
        job, _ = enqueue(wired, make_job(1))
        seen.advance(job.id, JobState.APPROVED)
        assert pending(job_store, seen, config) == []

    def test_ignores_offers_without_a_bundle(self, wired):
        config, seen, job_store, _ = wired
        job = make_job(1)
        seen.record(job)
        job_store.save(job, fit=fit())
        assert pending(job_store, seen, config) == []


class TestConcerns:
    def test_clean_bundle_has_none(self, wired):
        config, seen, job_store, _ = wired
        enqueue(wired, make_job(1))
        [item] = pending(job_store, seen, config)
        assert item.concerns == []

    def test_reports_banned_phrases_length_and_review(self, wired, tmp_path):
        config, seen, job_store, _ = wired
        enqueue(
            wired,
            make_job(1),
            letter_text="trop courte",
            banned_hits=["passionné par"],
            review=ReviewVerdict(
                approved=False,
                issues=["lettre interchangeable"],
                unsupported_claims=["cinq ans de Kubernetes"],
            ),
        )
        [item] = pending(job_store, seen, config)
        joined = " | ".join(item.concerns)
        assert "passionné par" in joined
        assert "longueur" in joined
        assert "cinq ans de Kubernetes" in joined
        assert "interchangeable" in joined

    def test_reports_a_missing_cv(self, wired, config):
        _, seen, job_store, _ = wired
        _, directory = enqueue(wired, make_job(1))
        (directory / "cv.pdf").unlink()
        record = job_store.get(pending(job_store, seen, config)[0].job_id)
        assert any("CV" in c for c in concerns(record, directory, config))


class TestResolve:
    def test_matches_an_id_prefix_and_a_directory_fragment(self, wired):
        config, seen, job_store, _ = wired
        job, directory = enqueue(wired, make_job(1))
        queue = pending(job_store, seen, config)
        assert resolve(queue, job.id[:8]).job_id == job.id
        assert resolve(queue, "acme-1").job_id == job.id

    def test_refuses_an_ambiguous_or_unknown_reference(self, wired):
        config, seen, job_store, _ = wired
        enqueue(wired, make_job(1))
        enqueue(wired, make_job(2))
        queue = pending(job_store, seen, config)
        with pytest.raises(LookupError, match="ambiguë"):
            resolve(queue, "acme")
        with pytest.raises(LookupError, match="aucun dossier"):
            resolve(queue, "zzzzz")


class TestApprove:
    def test_records_the_decision_everywhere(self, wired):
        config, seen, job_store, _ = wired
        job, directory = enqueue(wired, make_job(1))
        [item] = pending(job_store, seen, config)

        approve(
            item,
            seen=seen,
            job_store=job_store,
            banned=BannedPhrases([]),
            config=config,
            note="à relancer sous 10 jours",
        )

        assert seen.get(job.id).state is JobState.APPROVED
        decision = job_store.get(job.id).decision
        assert decision.decision == "approved"
        assert decision.note == "à relancer sous 10 jours"
        written = json.loads((directory / "decision.json").read_text(encoding="utf-8"))
        assert written["decision"] == "approved"

    def test_takes_the_hand_corrected_letter(self, wired):
        config, seen, job_store, _ = wired
        job, directory = enqueue(wired, make_job(1))
        corrected = "Ma propre version, écrite à la main."
        (directory / "lettre.md").write_text(corrected, encoding="utf-8")

        [item] = pending(job_store, seen, config)
        approve(
            item,
            seen=seen,
            job_store=job_store,
            banned=BannedPhrases([]),
            config=config,
        )

        letter = job_store.get(job.id).letter
        assert letter.text == corrected
        assert letter.edited
        assert job_store.get(job.id).decision.letter_edited

    def test_rechecks_banned_phrases_on_the_corrected_letter(self, wired):
        config, seen, job_store, _ = wired
        job, directory = enqueue(wired, make_job(1))
        (directory / "lettre.md").write_text(
            "Fort de mon expérience, je postule.", encoding="utf-8"
        )

        [item] = pending(job_store, seen, config)
        decided = approve(
            item,
            seen=seen,
            job_store=job_store,
            banned=BannedPhrases(["fort de mon expérience"]),
            config=config,
        )

        assert job_store.get(job.id).letter.banned_hits == ["fort de mon expérience"]
        # La réserve est conservée dans la décision : approuver malgré tout doit
        # rester visible après coup.
        assert any("fort de mon expérience" in c for c in decided.concerns)
        assert job_store.get(job.id).decision.concerns

    def test_ignores_the_letter_header_comment(self, wired):
        config, seen, job_store, _ = wired
        job, directory = enqueue(wired, make_job(1))
        # Fichier inchangé : l'en-tête HTML ne doit pas passer pour une correction.
        [item] = pending(job_store, seen, config)
        approve(
            item,
            seen=seen,
            job_store=job_store,
            banned=BannedPhrases([]),
            config=config,
        )
        assert not job_store.get(job.id).letter.edited
        assert read_letter(directory) == LETTRE


class TestReject:
    def test_records_the_reason_in_memory(self, wired):
        config, seen, job_store, _ = wired
        job, directory = enqueue(wired, make_job(1))
        [item] = pending(job_store, seen, config)

        reject(item, seen=seen, job_store=job_store, note="salaire non publié")

        entry = seen.get(job.id)
        assert entry.state is JobState.REJECTED
        assert "salaire non publié" in entry.reason
        assert job_store.get(job.id).decision.decision == "rejected"
        assert (directory / "decision.json").exists()


class TestRunReview:
    def run(self, wired, answers, **kwargs):
        config, seen, job_store, _ = wired
        console = ScriptedConsole(answers)
        report = run_review(
            config,
            seen=seen,
            job_store=job_store,
            banned=BannedPhrases([]),
            console=console,
            **kwargs,
        )
        return report, console

    def test_approve_then_reject(self, wired):
        job1, _ = enqueue(wired, make_job(1), fit=fit(score=90))
        job2, _ = enqueue(wired, make_job(2), fit=fit(score=70))
        report, _ = self.run(wired, ["a", "r", "hors sujet"])

        assert [item.job_id for item in report.approved] == [job1.id]
        assert [item.job_id for item in report.rejected] == [job2.id]
        _, seen, job_store, _ = wired
        assert seen.get(job1.id).state is JobState.APPROVED
        assert seen.get(job2.id).state is JobState.REJECTED

    def test_later_leaves_the_offer_waiting(self, wired):
        job, _ = enqueue(wired, make_job(1))
        report, _ = self.run(wired, ["p"])
        assert [item.job_id for item in report.postponed] == [job.id]
        assert wired[1].get(job.id).state is JobState.AWAITING_USER

    def test_quitting_decides_nothing_for_the_rest(self, wired):
        enqueue(wired, make_job(1), fit=fit(score=90))
        job2, _ = enqueue(wired, make_job(2), fit=fit(score=70))
        report, _ = self.run(wired, ["q"])
        assert len(report.postponed) == 2
        assert wired[1].get(job2.id).state is JobState.AWAITING_USER

    def test_closed_input_is_a_quit_not_a_decision(self, wired):
        job, _ = enqueue(wired, make_job(1))

        def closed(prompt: str) -> str:
            raise EOFError

        config, seen, job_store, _ = wired
        report = run_review(
            config,
            seen=seen,
            job_store=job_store,
            banned=BannedPhrases([]),
            console=Console(write=lambda _: None, ask=closed),
        )
        assert report.postponed and not report.approved
        assert seen.get(job.id).state is JobState.AWAITING_USER

    def test_editing_reloads_the_letter_before_deciding(self, wired):
        job, directory = enqueue(wired, make_job(1))
        config, seen, job_store, _ = wired
        console = ScriptedConsole(["e", "a"])
        console.editor_writes = "Version corrigée dans l'éditeur."
        run_review(
            config,
            seen=seen,
            job_store=job_store,
            banned=BannedPhrases([]),
            console=console,
        )
        assert console.edited == [directory / "lettre.md"]
        assert job_store.get(job.id).letter.text == "Version corrigée dans l'éditeur."

    def test_unknown_answer_asks_again(self, wired):
        enqueue(wired, make_job(1))
        report, console = self.run(wired, ["x", "p"])
        assert len(report.postponed) == 1
        assert "réponse inconnue" in console.output

    def test_open_shows_the_preview_without_deciding(self, wired):
        job, directory = enqueue(wired, make_job(1))
        report, console = self.run(wired, ["o", "p"])
        assert console.opened == [(directory / "preview.html").as_uri()]
        assert wired[1].get(job.id).state is JobState.AWAITING_USER

    def test_only_restricts_the_pass_to_one_bundle(self, wired):
        job1, _ = enqueue(wired, make_job(1), fit=fit(score=90))
        job2, _ = enqueue(wired, make_job(2), fit=fit(score=70))
        report, _ = self.run(wired, ["a"], only=job2.id[:8])
        assert [item.job_id for item in report.approved] == [job2.id]
        assert wired[1].get(job1.id).state is JobState.AWAITING_USER

    def test_unknown_reference_is_an_error_not_a_full_pass(self, wired):
        job, _ = enqueue(wired, make_job(1))
        report, _ = self.run(wired, [], only="inconnu")
        assert report.errors and not report.approved
        assert wired[1].get(job.id).state is JobState.AWAITING_USER

    def test_shows_the_letter_and_the_concerns(self, wired):
        enqueue(
            wired,
            make_job(1),
            letter_text="trop courte",
            banned_hits=["passionné par"],
        )
        _, console = self.run(wired, ["p"])
        assert "trop courte" in console.output
        assert "passionné par" in console.output
        assert "Réserves" in console.output
