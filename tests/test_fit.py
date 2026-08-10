import json

import pytest

from agent_emploi.agents.fit import MAX_DESCRIPTION_CHARS, FitAgent, build_prompt
from agent_emploi.config import FitConfig
from agent_emploi.llm.providers.base import Completion
from agent_emploi.llm.router import Router
from agent_emploi.models import FitVerdict, Job

CV = "Python, Tensorflow, LLM, RAG. École 42."

VERDICT = {
    "score": 78,
    "verdict": "apply",
    "matched": ["Python", "LLM"],
    "gaps": ["Kubernetes"],
    "reason": "Bon recouvrement technique, expérience industrielle absente.",
    "language": "fr",
}


class ScriptedProvider:
    """Fournisseur qui renvoie des réponses préparées, dans l'ordre."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def complete(self, **kwargs) -> Completion:
        self.calls.append(kwargs)
        return Completion(text=self._responses.pop(0), tokens_in=800, tokens_out=120)


def make_job(**fields) -> Job:
    base = {
        "title": "Ingénieur LLM",
        "company": "Acme",
        "location": "Paris",
        "contract": "stage",
        "description": "Développement d'agents LLM en Python.",
    }
    return Job.build(source="fake", url="https://example.com/jobs/1", **{**base, **fields})


@pytest.fixture
def agent(config, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    router = Router(config)
    router._providers["groq"] = ScriptedProvider([json.dumps(VERDICT)])
    return FitAgent(router, CV, FitConfig(min_score=65, accept_verdicts=["apply"]))


class TestBuildPrompt:
    def test_contains_cv_offer_and_barème(self):
        prompt = build_prompt(make_job(), CV)
        assert CV in prompt
        assert "Ingénieur LLM" in prompt
        assert "Acme" in prompt
        assert "Barème de notation" in prompt

    def test_cv_comes_first_for_prompt_caching(self):
        prompt = build_prompt(make_job(), CV)
        assert prompt.index(CV) < prompt.index("# Offre")

    def test_omits_empty_metadata(self):
        prompt = build_prompt(make_job(salary=None, remote=None), CV)
        assert "Salaire" not in prompt
        assert "Télétravail" not in prompt

    def test_truncates_long_description(self):
        job = make_job(description="mot " * 5000)
        prompt = build_prompt(job, CV)
        assert "description tronquée" in prompt
        assert len(prompt) < MAX_DESCRIPTION_CHARS + 3000


class TestEvaluate:
    def test_returns_validated_verdict(self, agent):
        verdict = agent.evaluate(make_job())
        assert verdict.score == 78
        assert verdict.verdict == "apply"
        assert verdict.language == "fr"

    def test_logs_usage_against_the_job(self, agent, config):
        job = make_job()
        agent.evaluate(job)
        lines = config.paths.usage_file.read_text(encoding="utf-8").splitlines()
        usage = json.loads(lines[-1])
        assert usage["task"] == "fit_check"
        assert usage["job_id"] == job.id

    def test_uses_the_free_tier_route(self, agent):
        agent.evaluate(make_job())
        provider = agent.router._providers["groq"]
        assert provider.calls[0]["model"] == "llama-test"
        assert "recruteur technique" in provider.calls[0]["system"]


class TestDecision:
    @pytest.fixture
    def agent(self):
        return FitAgent(
            router=None, cv_text=CV, config=FitConfig(min_score=65, accept_verdicts=["apply"])
        )

    def verdict(self, **fields) -> FitVerdict:
        return FitVerdict.model_validate({**VERDICT, **fields})

    def test_accepts_above_threshold(self, agent):
        assert agent.accepts(self.verdict(score=65))

    def test_refuses_below_threshold(self, agent):
        low = self.verdict(score=64)
        assert not agent.accepts(low)
        assert agent.rejection_reason(low) == "fit:score 64<65"

    def test_refuses_unaccepted_verdict_despite_high_score(self, agent):
        maybe = self.verdict(verdict="maybe", score=90)
        assert not agent.accepts(maybe)
        assert agent.rejection_reason(maybe) == "fit:maybe(90)"

    def test_accept_verdicts_are_configurable(self):
        agent = FitAgent(None, CV, FitConfig(min_score=50, accept_verdicts=["apply", "maybe"]))
        assert agent.accepts(FitVerdict.model_validate({**VERDICT, "verdict": "maybe"}))
