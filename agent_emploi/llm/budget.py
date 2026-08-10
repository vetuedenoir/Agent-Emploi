"""Suivi de consommation LLM et plafond de dépense quotidien.

Chaque appel est journalisé dans `llm_usage.jsonl`. Avant chaque appel, le
routeur vérifie que les plafonds du jour ne sont pas atteints ; sinon la boucle
s'arrête proprement plutôt que de dériver en coût.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

from agent_emploi.config import BudgetConfig, Pricing
from agent_emploi.models import LlmUsage, utcnow

logger = logging.getLogger(__name__)


class BudgetExceeded(RuntimeError):
    """Plafond quotidien atteint. La boucle doit s'arrêter, pas réessayer."""


def estimate_cost(pricing: Pricing, tokens_in: int, tokens_out: int) -> float:
    """Coût estimé d'un appel, en dollars (tarifs par million de tokens)."""
    return (tokens_in * pricing.input + tokens_out * pricing.output) / 1_000_000


class BudgetTracker:
    """Journalise les appels et fait respecter les plafonds du jour."""

    def __init__(self, path: Path, config: BudgetConfig) -> None:
        self.path = Path(path)
        self.config = config
        self._today = utcnow().date()
        self._spent_usd = 0.0
        self._calls = 0
        self._load_today()

    def _load_today(self) -> None:
        """Recharge la consommation du jour depuis le journal.

        Nécessaire pour que plusieurs exécutions successives dans la même journée
        partagent le même plafond.
        """
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    usage = LlmUsage.model_validate_json(line)
                except (ValueError, json.JSONDecodeError):
                    continue
                if usage.at.date() != self._today:
                    continue
                self._spent_usd += usage.cost_est
                self._calls += 1

    def _roll_over_if_needed(self) -> None:
        """Remet les compteurs à zéro si le jour a changé pendant l'exécution."""
        today = utcnow().date()
        if today != self._today:
            self._today = today
            self._spent_usd = 0.0
            self._calls = 0

    @property
    def spent_usd(self) -> float:
        self._roll_over_if_needed()
        return self._spent_usd

    @property
    def calls(self) -> int:
        self._roll_over_if_needed()
        return self._calls

    @property
    def day(self) -> date:
        self._roll_over_if_needed()
        return self._today

    def check(self) -> None:
        """Lève `BudgetExceeded` si un plafond du jour est atteint."""
        self._roll_over_if_needed()
        if self._spent_usd >= self.config.daily_usd:
            raise BudgetExceeded(
                f"plafond de dépense atteint: "
                f"{self._spent_usd:.4f} $ / {self.config.daily_usd:.2f} $ le {self._today}"
            )
        if self._calls >= self.config.daily_calls:
            raise BudgetExceeded(
                f"plafond d'appels atteint: "
                f"{self._calls} / {self.config.daily_calls} le {self._today}"
            )

    def record(self, usage: LlmUsage) -> None:
        """Journalise un appel et met à jour les compteurs du jour."""
        self._roll_over_if_needed()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(usage.model_dump_json() + "\n")
        self._spent_usd += usage.cost_est
        self._calls += 1

    def summary(self) -> str:
        return (
            f"{self.day}: {self.calls} appels, "
            f"{self.spent_usd:.4f} $ estimés "
            f"(plafonds: {self.config.daily_calls} appels, {self.config.daily_usd:.2f} $)"
        )
