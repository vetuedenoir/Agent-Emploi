import pytest

from agent_emploi.models import (
    ALLOWED_TRANSITIONS,
    InvalidTransition,
    Job,
    JobRecord,
    JobState,
    canonical_url,
    check_transition,
    job_id,
)


def make_job(**overrides) -> Job:
    fields = {
        "source": "wttj",
        "url": "https://example.com/jobs/ml-engineer",
        "title": "Machine Learning Engineer",
        "company": "Acme",
    }
    fields.update(overrides)
    source = fields.pop("source")
    url = fields.pop("url")
    return Job.build(source=source, url=url, **fields)


class TestCanonicalUrl:
    def test_strips_query_fragment_and_trailing_slash(self):
        assert (
            canonical_url("https://Example.com/jobs/42/?utm_source=x#top")
            == "https://example.com/jobs/42"
        )

    def test_same_offer_different_tracking_gives_same_id(self):
        a = job_id("wttj", "https://x.com/j/1?utm_campaign=a")
        b = job_id("wttj", "https://x.com/j/1?ref=newsletter")
        assert a == b

    def test_different_sources_give_different_ids(self):
        assert job_id("wttj", "https://x.com/j/1") != job_id(
            "france_travail", "https://x.com/j/1"
        )


class TestTransitions:
    def test_nominal_path_is_allowed(self):
        path = [
            JobState.DISCOVERED,
            JobState.PRESCREENED,
            JobState.FIT_OK,
            JobState.DRAFTED,
            JobState.REVIEWED,
            JobState.AWAITING_USER,
            JobState.APPROVED,
            JobState.SUBMITTED,
        ]
        for current, target in zip(path, path[1:]):
            check_transition(current, target)

    def test_rejection_allowed_from_any_active_state(self):
        for state in (
            JobState.DISCOVERED,
            JobState.PRESCREENED,
            JobState.FIT_OK,
            JobState.DRAFTED,
            JobState.REVIEWED,
            JobState.AWAITING_USER,
            JobState.APPROVED,
            JobState.PREFILLED,
        ):
            check_transition(state, JobState.REJECTED)

    def test_review_can_send_back_to_drafting(self):
        check_transition(JobState.REVIEWED, JobState.DRAFTED)

    def test_cannot_submit_without_user_approval(self):
        for state in (JobState.DRAFTED, JobState.AWAITING_USER):
            with pytest.raises(InvalidTransition):
                check_transition(state, JobState.SUBMITTED)

    def test_legacy_states_only_lead_to_the_end(self):
        """`prefilled` et `handoff`, hérités du remplissage automatique."""
        for state in (JobState.PREFILLED, JobState.HANDOFF):
            check_transition(state, JobState.SUBMITTED)
            check_transition(state, JobState.REJECTED)
        with pytest.raises(InvalidTransition):
            check_transition(JobState.APPROVED, JobState.PREFILLED)

    def test_sent_application_can_lead_to_an_interview(self):
        check_transition(JobState.SUBMITTED, JobState.INTERVIEW)
        for state in (JobState.APPROVED, JobState.TRACKED, JobState.AWAITING_USER):
            with pytest.raises(InvalidTransition):
                check_transition(state, JobState.INTERVIEW)

    def test_terminal_states_have_no_exit(self):
        with pytest.raises(InvalidTransition):
            check_transition(JobState.INTERVIEW, JobState.SUBMITTED)
        with pytest.raises(InvalidTransition):
            check_transition(JobState.SUBMITTED, JobState.DISCOVERED)
        with pytest.raises(InvalidTransition):
            check_transition(JobState.REJECTED, JobState.DISCOVERED)

    def test_tracked_offer_can_be_prepared_sent_or_dropped(self):
        for target in (JobState.PRESCREENED, JobState.SUBMITTED, JobState.REJECTED):
            check_transition(JobState.TRACKED, target)

    def test_tracked_offer_cannot_skip_to_a_letter(self):
        # Préparer passe par le pré-filtrage, donc par le fit-check.
        for target in (JobState.FIT_OK, JobState.DRAFTED, JobState.APPROVED):
            with pytest.raises(InvalidTransition):
                check_transition(JobState.TRACKED, target)

    def test_every_state_has_its_transitions(self):
        assert set(ALLOWED_TRANSITIONS) == set(JobState)


class TestManualQuery:
    def test_keep_query_drops_only_tracking(self):
        assert (
            canonical_url("https://x.com/viewjob?utm_medium=a&jk=9&ref=b", keep_query=True)
            == "https://x.com/viewjob?jk=9"
        )

    def test_default_ids_are_unchanged(self):
        assert canonical_url("https://x.com/viewjob?jk=9") == "https://x.com/viewjob"


class TestLegacyApplication:
    """Les lignes de `jobs.jsonl` écrites du temps du remplissage automatique."""

    def test_submitted_date_is_recovered(self):
        job = Job.build(source="wttj", url="https://example.com/o/1", title="T", company="C")
        record = JobRecord.model_validate(
            {
                "job": job.model_dump(mode="json"),
                "application": {
                    "status": "handoff",
                    "at": "2026-08-11T09:30:00Z",
                    "apply_url": "https://ats/x",
                    "submitted": True,
                },
            }
        )
        assert record.submitted_at.isoformat() == "2026-08-11T09:30:00+00:00"

    def test_unsent_application_is_dropped(self):
        job = Job.build(source="wttj", url="https://example.com/o/1", title="T", company="C")
        record = JobRecord.model_validate(
            {
                "job": job.model_dump(mode="json"),
                "application": {"status": "handoff", "apply_url": "x", "submitted": False},
            }
        )
        assert record.submitted_at is None


def test_records_without_cv_variant_still_load():
    """Les verdicts et lettres écrits avant les variantes de CV se relisent."""
    record = JobRecord.model_validate(
        {
            "job": make_job().model_dump(mode="json"),
            "fit": {"score": 70, "verdict": "apply", "reason": "ok", "language": "fr"},
            "letter": {"text": "Madame, Monsieur.", "language": "fr"},
        }
    )
    assert record.fit.cv is None
    assert record.letter.cv is None
