import pytest

from agent_emploi.models import (
    ALLOWED_TRANSITIONS,
    InvalidTransition,
    Job,
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
            JobState.PREFILLED,
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

    def test_cannot_skip_user_validation(self):
        with pytest.raises(InvalidTransition):
            check_transition(JobState.DRAFTED, JobState.PREFILLED)

    def test_cannot_prefill_without_user_approval(self):
        # L'étape 6 ne doit avoir aucun chemin qui contourne la validation.
        with pytest.raises(InvalidTransition):
            check_transition(JobState.AWAITING_USER, JobState.PREFILLED)

    def test_cannot_submit_without_prefill_or_handoff(self):
        with pytest.raises(InvalidTransition):
            check_transition(JobState.AWAITING_USER, JobState.SUBMITTED)

    def test_terminal_states_have_no_exit(self):
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
