import json
from dataclasses import replace

import pytest

from agent_emploi.agents.review import MAX_DESCRIPTION_CHARS, ReviewAgent, build_prompt
from agent_emploi.llm.router import Router
from agent_emploi.models import Job, Letter, ReviewVerdict
from agent_emploi.profile import CvText
from tests.conftest import ScriptedProvider

APPROVED = {"approved": True, "issues": [], "unsupported_claims": []}
REFUSED = {
    "approved": False,
    "issues": ["lettre interchangeable"],
    "unsupported_claims": ["cinq ans d'expérience en Kubernetes"],
}


def make_job(**fields) -> Job:
    base = {
        "title": "Ingénieur LLM",
        "company": "Acme",
        "description": "Agents LLM en Python.",
    }
    return Job.build(source="fake", url="https://example.com/jobs/1", **{**base, **fields})


def make_letter(text: str = "Madame, Monsieur, j'ai écrit un MLP.") -> Letter:
    return Letter(text=text, language="fr")


@pytest.fixture
def agent(config, profile, monkeypatch) -> ReviewAgent:
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    router = Router(config)
    router._providers["gemini"] = ScriptedProvider([json.dumps(APPROVED)])
    return ReviewAgent(router, profile)


class TestBuildPrompt:
    def test_carries_cv_offer_and_letter(self, profile):
        prompt = build_prompt(make_job(), make_letter(), profile.cv_text)
        assert "Perceptron multicouche" in prompt
        assert "Acme" in prompt
        assert "j'ai écrit un MLP" in prompt

    def test_cv_comes_first_for_prompt_caching(self, profile):
        prompt = build_prompt(make_job(), make_letter(), profile.cv_text)
        assert prompt.index(profile.cv_text[:20]) < prompt.index("# Offre")

    def test_truncates_a_long_description(self, profile):
        prompt = build_prompt(make_job(description="mot " * 3000), make_letter(), "cv")
        assert "description tronquée" in prompt
        assert len(prompt) < MAX_DESCRIPTION_CHARS + 2000


class TestReview:
    def test_returns_a_validated_verdict(self, agent):
        assert agent.review(make_job(), make_letter()).approved

    def test_uses_the_free_route(self, agent):
        agent.review(make_job(), make_letter())
        call = agent.router._providers["gemini"].calls[0]
        assert call["model"] == "gemini-test"
        assert "relecteur exigeant" in call["system"]

    def test_reports_unsupported_claims(self, config, profile, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        router = Router(config)
        router._providers["gemini"] = ScriptedProvider([json.dumps(REFUSED)])
        verdict = ReviewAgent(router, profile).review(make_job(), make_letter())
        assert not verdict.approved
        assert verdict.unsupported_claims == ["cinq ans d'expérience en Kubernetes"]


class TestFeedback:
    def test_quotes_every_reproach_for_the_rewrite(self):
        feedback = ReviewAgent.feedback(ReviewVerdict.model_validate(REFUSED))
        assert "cinq ans d'expérience en Kubernetes" in feedback
        assert "lettre interchangeable" in feedback
        assert "Réécris" in feedback

    def test_stays_usable_when_the_review_gave_no_detail(self):
        feedback = ReviewAgent.feedback(ReviewVerdict(approved=False))
        assert "refusée" in feedback


def test_review_reads_the_cv_the_letter_was_written_from(config, profile, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    profile = replace(
        profile,
        cvs=(CvText("dev", "Dev", "Backend Django."), CvText("ml", "ML", "Keras, vision.")),
    )
    router = Router(config)
    provider = ScriptedProvider([json.dumps(APPROVED)])
    router._providers["gemini"] = provider
    letter = Letter(text="Madame, Monsieur.", language="fr", cv="ml")
    ReviewAgent(router, profile).review(make_job(), letter)
    assert "Keras, vision." in provider.calls[0]["prompt"]
