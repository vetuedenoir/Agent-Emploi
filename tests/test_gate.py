"""Porte Jev : état envoyé, seuils, journalisation du coût."""

from __future__ import annotations

import json

import pytest

from agent_emploi.agents.gate import (
    API_KEY_ENV,
    GateAgent,
    build_state,
    decide,
    questions,
)
from agent_emploi.config import GateConfig, Pricing
from agent_emploi.llm.budget import BudgetExceeded, BudgetTracker
from agent_emploi.llm.jev import (
    MAX_STATE_CHARS,
    JevError,
    JevResponse,
    serialized_length,
)
from agent_emploi.models import Job


def make_job(description: str = "Stage ML en Python.", **fields) -> Job:
    base = {"title": "Stage ML", "company": "Acme", "contract": "stage"}
    return Job.build(
        source="fake",
        url="https://example.com/jobs/1",
        description=description,
        **{**base, **fields},
    )


class FakeClient:
    model = "jev-latest"

    def __init__(self, *results) -> None:
        self.results = list(results)
        self.calls: list[dict] = []

    def ask(self, state, asked, *, idempotency_key):
        self.calls.append({"state": state, "questions": asked, "key": idempotency_key})
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def answers(adequation=2.0, experience=0.1, contract=0.9) -> JevResponse:
    data = {
        "adequation": {"type": "score", "score": adequation},
        "experience_bloquante": {"type": "noul", "noul": experience},
    }
    if contract is not None:
        data["contrat_compatible"] = {"type": "noul", "noul": contract}
    return JevResponse(answers=data, tokens_in=2000)


@pytest.fixture
def budget(tmp_path):
    from agent_emploi.config import BudgetConfig

    return BudgetTracker(tmp_path / "usage.jsonl", BudgetConfig(daily_calls=10))


def agent(client, budget, contracts=("stage", "alternance")) -> GateAgent:
    return GateAgent(
        client,
        budget,
        "CV : Python, Tensorflow, LLM.",
        GateConfig(enabled=True),
        contracts=list(contracts),
        pricing=Pricing(input=0.042),
    )


class TestState:
    def test_long_description_is_cut_to_the_api_limit(self):
        # Retours à la ligne et guillemets s'échappent : c'est l'état sérialisé
        # qui doit tenir, pas seulement la description.
        state = build_state(make_job("ligne \"citée\"\n" * 2000), "CV " * 500)
        assert serialized_length(state) <= MAX_STATE_CHARS
        assert state["offre"]["description"]

    def test_emoji_and_spacing_count_toward_the_limit(self):
        # Refusé en réalité à 8 000 en JSON compact : les émojis comptent
        # double en UTF-16, et l'API peut compter les espaces du JSON.
        state = build_state(make_job("🚀 Stage IA, agents: RAG. " * 600), "CV " * 500)
        compact = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
        assert len(compact.encode("utf-16-le")) // 2 <= MAX_STATE_CHARS
        assert len(json.dumps(state, ensure_ascii=False)) <= MAX_STATE_CHARS

    def test_short_offer_is_sent_whole(self):
        job = make_job(location="Paris", remote=None)
        state = build_state(job, "CV")
        assert state["offre"] == {
            "titre": "Stage ML",
            "entreprise": "Acme",
            "contrat": "stage",
            "lieu": "Paris",
            "description": "Stage ML en Python.",
        }

    def test_contract_question_follows_the_configured_contracts(self):
        asked = questions(["stage", "alternance"])
        assert "stage, alternance" in asked["contrat_compatible"]["instructions"]
        assert "contrat_compatible" not in questions([])

    def test_instructions_fit_the_api_limit(self):
        for question in questions(["stage", "alternance", "CDI"]).values():
            assert len(question["instructions"]) <= 1800


class TestDecide:
    config = GateConfig(min_score=1.5, max_blocking=0.7, min_contract=0.3)

    def test_passes_when_every_threshold_holds(self):
        verdict = decide(self.config, 2.0, 0.2, 0.8)
        assert verdict.passed and verdict.reason is None

    def test_contract_is_the_first_reason(self):
        verdict = decide(self.config, 0.5, 0.9, 0.1)
        assert verdict.reason == "jev:contrat(0.10)"

    def test_blocking_experience(self):
        assert decide(self.config, 2.5, 0.82, 0.9).reason == "jev:experience(0.82)"

    def test_low_adequation(self):
        assert decide(self.config, 1.2, 0.1, 0.9).reason == "jev:adequation 1.2<1.5"

    def test_no_contract_question_no_contract_rejection(self):
        assert decide(self.config, 2.0, 0.1, None).passed


class TestAgent:
    def test_evaluates_and_logs_the_cost(self, budget):
        client = FakeClient(answers(experience=0.9))
        verdict = agent(client, budget).evaluate(make_job())

        assert verdict.reason == "jev:experience(0.90)"
        assert budget.calls == 1
        assert budget.spent_usd == pytest.approx(2000 * 0.042 / 1_000_000)
        assert client.calls[0]["key"].startswith("gate-")

    def test_same_offer_same_idempotency_key(self, budget):
        client = FakeClient(answers(), answers())
        gate = agent(client, budget)
        gate.evaluate(make_job())
        gate.evaluate(make_job())
        assert client.calls[0]["key"] == client.calls[1]["key"]

    def test_failure_is_logged_then_raised(self, budget):
        client = FakeClient(JevError("422"))
        with pytest.raises(JevError):
            agent(client, budget).evaluate(make_job())
        assert budget.calls == 1

    def test_incomplete_answer_is_an_error(self, budget):
        client = FakeClient(JevResponse(answers={"adequation": {"score": 2}}))
        with pytest.raises(JevError, match="incomplète"):
            agent(client, budget).evaluate(make_job())

    def test_budget_is_checked_before_calling(self, tmp_path):
        from agent_emploi.config import BudgetConfig

        spent = BudgetTracker(tmp_path / "u.jsonl", BudgetConfig(daily_calls=0))
        client = FakeClient()
        with pytest.raises(BudgetExceeded):
            agent(client, spent).evaluate(make_job())
        assert client.calls == []


class TestFromConfig:
    def test_disabled_gate_is_none(self, config, budget, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV, "sk-x")
        assert GateAgent.from_config(config, "CV", budget) is None

    def test_missing_key_is_none(self, config, budget, monkeypatch):
        monkeypatch.delenv(API_KEY_ENV, raising=False)
        config.gate.enabled = True
        assert GateAgent.from_config(config, "CV", budget) is None

    def test_force_ignores_enabled(self, config, budget, monkeypatch):
        monkeypatch.setenv(API_KEY_ENV, "sk-x")
        config.search.contracts = ["stage"]
        gate = GateAgent.from_config(config, "CV", budget, force=True)
        assert gate is not None and gate.contracts == ["stage"]
