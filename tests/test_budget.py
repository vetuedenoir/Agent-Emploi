from datetime import timedelta

import pytest

from agent_emploi.config import BudgetConfig, Pricing
from agent_emploi.llm.budget import BudgetExceeded, BudgetTracker, estimate_cost
from agent_emploi.models import LlmUsage, utcnow


def test_estimate_cost_uses_per_million_rates():
    pricing = Pricing(input=5.0, output=25.0)
    # 1000 tokens d'entrée + 1000 de sortie = 0,005 $ + 0,025 $
    assert estimate_cost(pricing, 1000, 1000) == pytest.approx(0.03)


def test_free_tier_costs_nothing():
    assert estimate_cost(Pricing(), 100_000, 100_000) == 0.0


class TestLimits:
    def test_under_limits_passes(self, tmp_path):
        tracker = BudgetTracker(
            tmp_path / "usage.jsonl", BudgetConfig(daily_usd=1.0, daily_calls=10)
        )
        tracker.check()

    def test_spend_limit_stops_the_loop(self, tmp_path):
        tracker = BudgetTracker(
            tmp_path / "usage.jsonl", BudgetConfig(daily_usd=0.05, daily_calls=100)
        )
        tracker.record(LlmUsage(task="letter", provider="anthropic", model="m", cost_est=0.06))
        with pytest.raises(BudgetExceeded, match="dépense"):
            tracker.check()

    def test_call_limit_stops_the_loop(self, tmp_path):
        tracker = BudgetTracker(
            tmp_path / "usage.jsonl", BudgetConfig(daily_usd=100.0, daily_calls=2)
        )
        for _ in range(2):
            tracker.record(LlmUsage(task="fit_check", provider="groq", model="m"))
        with pytest.raises(BudgetExceeded, match="appels"):
            tracker.check()


class TestPersistence:
    def test_reload_restores_today_spend(self, tmp_path):
        path = tmp_path / "usage.jsonl"
        config = BudgetConfig(daily_usd=1.0, daily_calls=100)
        BudgetTracker(path, config).record(
            LlmUsage(task="letter", provider="anthropic", model="m", cost_est=0.4)
        )

        # Une seconde exécution le même jour partage le même plafond.
        reloaded = BudgetTracker(path, config)
        assert reloaded.spent_usd == pytest.approx(0.4)
        assert reloaded.calls == 1

    def test_yesterday_usage_is_ignored(self, tmp_path):
        path = tmp_path / "usage.jsonl"
        stale = LlmUsage(
            task="letter",
            provider="anthropic",
            model="m",
            cost_est=99.0,
            at=utcnow() - timedelta(days=1),
        )
        path.write_text(stale.model_dump_json() + "\n", encoding="utf-8")

        tracker = BudgetTracker(path, BudgetConfig(daily_usd=1.0))
        assert tracker.spent_usd == 0.0
        tracker.check()
