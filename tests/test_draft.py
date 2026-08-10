import pytest

from agent_emploi.draft import run_draft, select_candidates
from agent_emploi.llm.budget import BudgetExceeded
from agent_emploi.llm.router import LlmError
from agent_emploi.models import FitVerdict, Job, JobState, Letter, ReviewVerdict
from agent_emploi.store.jobs import JobStore
from agent_emploi.store.seen import SeenStore

LETTRE = " ".join(["mot"] * 170)


def make_job(n: int = 1, **fields) -> Job:
    base = {
        "title": f"Ingénieur IA {n}",
        "company": f"Acme {n}",
        "contract": "stage",
        "description": "Agents LLM en Python.",
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


class FakeLetterAgent:
    """Rédacteur scripté : un texte (ou une exception) par appel."""

    def __init__(self, texts: list, config=None) -> None:
        self.texts = list(texts)
        self.calls: list[str | None] = []
        self.config = config

    def write(self, job: Job, verdict: FitVerdict, *, feedback: str | None = None):
        self.calls.append(feedback)
        result = self.texts.pop(0) if self.texts else LETTRE
        if isinstance(result, Exception):
            raise result
        return Letter(
            text=result,
            language=verdict.language,
            banned_hits=["passionné par"] if "passionné par" in result else [],
            regenerated=feedback is not None,
        )

    def defects(self, letter: Letter) -> list[str]:
        problems = []
        if letter.banned_hits:
            problems.append("formules interdites : " + ", ".join(letter.banned_hits))
        if letter.word_count < 150:
            problems.append(f"trop courte : {letter.word_count} mots")
        return problems


class FakeReviewAgent:
    """Relecteur scripté : un verdict (ou une exception) par appel."""

    def __init__(self, verdicts: list) -> None:
        self.verdicts = list(verdicts)
        self.seen: list[str] = []

    def review(self, job: Job, letter: Letter) -> ReviewVerdict:
        self.seen.append(job.id)
        result = self.verdicts.pop(0) if self.verdicts else ReviewVerdict(approved=True)
        if isinstance(result, Exception):
            raise result
        return result

    @staticmethod
    def feedback(verdict: ReviewVerdict) -> str:
        return "reprends : " + " ; ".join(verdict.issues)


@pytest.fixture
def wired(config, tmp_path, profile):
    """Mémoires vierges, une offre en `fit_ok`, prête à être rédigée."""
    config.profile.cv_fr = profile.cv_fr
    seen = SeenStore(tmp_path / "seen.jsonl")
    job_store = JobStore(tmp_path / "jobs.jsonl")
    return config, seen, job_store, profile


def enqueue(wired, job: Job, *, state: JobState = JobState.FIT_OK, **record_fields):
    """Amène une offre jusqu'à l'état voulu dans les deux mémoires."""
    config, seen, job_store, _ = wired
    seen.record(job)
    for target in (JobState.PRESCREENED, JobState.FIT_OK, JobState.DRAFTED, JobState.REVIEWED):
        seen.advance(job.id, target)
        if seen.get(job.id).state is state:
            break
    job_store.save(job, fit=record_fields.pop("fit", fit()), **record_fields)
    return job


def draft(wired, *, letters=None, reviews=None, **kwargs):
    config, seen, job_store, profile = wired
    letter_agent = FakeLetterAgent(letters or [])
    review_agent = FakeReviewAgent(reviews or [])
    report = run_draft(
        config,
        seen=seen,
        job_store=job_store,
        profile=profile,
        letter_agent=letter_agent,
        review_agent=review_agent,
        **kwargs,
    )
    return report, letter_agent, review_agent


class TestSelection:
    def test_takes_fit_ok_offers_best_scored_first(self, wired):
        _, seen, job_store, _ = wired
        enqueue(wired, make_job(1), fit=fit(score=70))
        enqueue(wired, make_job(2), fit=fit(score=90))
        assert [r.fit.score for r in select_candidates(job_store, seen)] == [90, 70]

    def test_ignores_offers_not_yet_judged(self, wired):
        _, seen, job_store, _ = wired
        enqueue(wired, make_job(1), state=JobState.PRESCREENED)
        assert select_candidates(job_store, seen) == []

    def test_ignores_offers_already_handed_to_the_user(self, wired):
        _, seen, job_store, _ = wired
        job = enqueue(wired, make_job(1))
        seen.advance(job.id, JobState.DRAFTED)
        seen.advance(job.id, JobState.REVIEWED)
        seen.advance(job.id, JobState.AWAITING_USER)
        assert select_candidates(job_store, seen) == []

    def test_resumes_an_interrupted_pass(self, wired):
        _, seen, job_store, _ = wired
        enqueue(wired, make_job(1), state=JobState.DRAFTED)
        assert len(select_candidates(job_store, seen)) == 1

    def test_honours_the_limit(self, wired):
        _, seen, job_store, _ = wired
        for n in (1, 2, 3):
            enqueue(wired, make_job(n))
        assert len(select_candidates(job_store, seen, limit=2)) == 2


class TestPipeline:
    def test_prepares_a_bundle_and_hands_over(self, wired):
        job = enqueue(wired, make_job(1))
        report, letters, reviews = draft(wired, letters=[LETTRE])

        assert report.generated == 1
        assert report.reviewed == 1
        [prepared] = report.prepared
        assert prepared.ok
        assert (prepared.directory / "lettre.md").exists()
        assert (prepared.directory / "cv.pdf").exists()

        _, seen, job_store, _ = wired
        assert seen.get(job.id).state is JobState.AWAITING_USER
        assert job_store.get(job.id).outbox == str(prepared.directory)

    def test_persists_the_letter_and_the_review(self, wired):
        job = enqueue(wired, make_job(1))
        draft(wired, letters=[LETTRE])
        record = JobStore(wired[2].path).get(job.id)
        assert record.letter.text == LETTRE
        assert record.review.approved

    def test_stops_before_any_submission(self, wired):
        enqueue(wired, make_job(1))
        report, _, _ = draft(wired, letters=[LETTRE])
        # Le dossier vit dans outbox/, jamais dans applications/ : rien n'est
        # parti, et l'archivage est l'affaire de l'étape 7.
        assert report.prepared[0].directory.is_relative_to(wired[0].paths.outbox)

    def test_nothing_to_do_is_not_an_error(self, wired):
        report, letters, _ = draft(wired)
        assert report.candidates == 0
        assert letters.calls == []


class TestRegeneration:
    def test_a_banned_phrase_triggers_one_rewrite(self, wired):
        enqueue(wired, make_job(1))
        report, letters, reviews = draft(
            wired, letters=["passionné par " + LETTRE, LETTRE]
        )

        assert report.regenerated == 1
        assert "formules interdites" in letters.calls[1]
        assert report.prepared[0].ok
        # La relecture payante n'a lieu qu'une fois la lettre propre.
        assert len(reviews.seen) == 1

    def test_a_refused_review_triggers_one_rewrite(self, wired):
        enqueue(wired, make_job(1))
        refused = ReviewVerdict(approved=False, issues=["lettre interchangeable"])
        report, letters, reviews = draft(
            wired,
            letters=[LETTRE, LETTRE],
            reviews=[refused, ReviewVerdict(approved=True)],
        )

        assert report.regenerated == 1
        assert "lettre interchangeable" in letters.calls[1]
        assert report.prepared[0].ok

    def test_defects_share_the_same_budget_as_the_review(self, wired):
        # Une reprise consommée par une formule interdite n'en laisse aucune
        # pour la revue : la lettre est livrée avec ses réserves.
        enqueue(wired, make_job(1))
        refused = ReviewVerdict(approved=False, unsupported_claims=["cinq ans"])
        report, letters, _ = draft(
            wired, letters=["passionné par " + LETTRE, LETTRE], reviews=[refused]
        )

        assert len(letters.calls) == 2
        [prepared] = report.prepared
        assert not prepared.ok
        assert "invention probable : cinq ans" in prepared.warnings

    def test_an_uncorrected_defect_is_escalated_not_dropped(self, wired):
        enqueue(wired, make_job(1))
        report, _, reviews = draft(
            wired, letters=["passionné par " + LETTRE, "passionné par " + LETTRE]
        )

        [prepared] = report.prepared
        assert not prepared.ok
        assert "passionné par" in prepared.warnings[0]
        # Inutile de payer une relecture d'un texte déjà disqualifié.
        assert reviews.seen == []
        assert report.flagged == [prepared]

    def test_the_budget_of_rewrites_is_configurable(self, wired):
        wired[0].letter.max_regenerations = 0
        enqueue(wired, make_job(1))
        report, letters, _ = draft(wired, letters=["passionné par " + LETTRE])
        assert len(letters.calls) == 1
        assert not report.prepared[0].ok


class TestResume:
    def test_an_already_written_letter_is_not_paid_twice(self, wired):
        enqueue(
            wired,
            make_job(1),
            state=JobState.DRAFTED,
            letter=Letter(text=LETTRE, language="fr"),
        )
        report, letters, reviews = draft(wired)

        assert letters.calls == []
        assert report.generated == 0
        assert report.resumed == 1
        assert len(reviews.seen) == 1
        assert report.prepared[0].ok

    def test_an_already_reviewed_letter_goes_straight_to_the_bundle(self, wired):
        enqueue(
            wired,
            make_job(1),
            state=JobState.REVIEWED,
            letter=Letter(text=LETTRE, language="fr"),
            review=ReviewVerdict(approved=True),
        )
        report, letters, reviews = draft(wired)

        assert letters.calls == []
        assert reviews.seen == []
        assert report.prepared[0].directory.exists()


class TestCv:
    def test_the_offer_language_picks_the_cv(self, wired):
        config, _, _, profile = wired
        profile.cv_en.write_bytes(b"%PDF-en")
        enqueue(wired, make_job(1), fit=fit(language="en"))
        report, _, _ = draft(wired, letters=[LETTRE])
        assert (report.prepared[0].directory / "cv.pdf").read_bytes() == b"%PDF-en"

    def test_a_missing_cv_is_a_warning_not_a_blocker(self, wired):
        enqueue(wired, make_job(1), fit=fit(language="en"))
        report, _, _ = draft(wired, letters=[LETTRE])
        [prepared] = report.prepared
        assert prepared.directory.exists()
        assert "aucun CV à joindre" in prepared.warnings[0]


class TestFailures:
    def test_budget_ceiling_stops_the_pass_cleanly(self, wired):
        job1 = enqueue(wired, make_job(1), fit=fit(score=90))
        enqueue(wired, make_job(2), fit=fit(score=80))
        report, letters, _ = draft(
            wired, letters=[LETTRE, BudgetExceeded("plafond atteint")]
        )

        assert len(report.prepared) == 1
        assert "plafond" in report.stopped
        _, seen, _, _ = wired
        assert seen.get(job1.id).state is JobState.AWAITING_USER

    def test_one_llm_failure_leaves_the_offer_for_the_next_pass(self, wired):
        job1 = enqueue(wired, make_job(1), fit=fit(score=90))
        enqueue(wired, make_job(2), fit=fit(score=80))
        report, _, _ = draft(wired, letters=[LlmError("503"), LETTRE])

        assert len(report.prepared) == 1
        assert report.errors and "503" in report.errors[0]
        _, seen, _, _ = wired
        assert seen.get(job1.id).state is JobState.FIT_OK

    def test_repeated_failures_stop_the_pass(self, wired):
        for n in (1, 2, 3):
            enqueue(wired, make_job(n))
        report, letters, _ = draft(
            wired, letters=[LlmError("503"), LlmError("503"), LETTRE]
        )

        assert report.prepared == []
        assert "échecs LLM consécutifs" in report.stopped
        assert len(letters.calls) == 2


class TestDryRun:
    def test_writes_nothing(self, wired):
        job = enqueue(wired, make_job(1))
        report, letters, _ = draft(wired, letters=[LETTRE], record=False)

        assert len(letters.calls) == 1
        assert len(report.prepared) == 1
        assert not report.prepared[0].directory.exists()

        _, seen, job_store, _ = wired
        assert seen.get(job.id).state is JobState.FIT_OK
        assert JobStore(job_store.path).get(job.id).letter is None
