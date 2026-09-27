"""Client Jev : transport, reprises, et classement des échecs.

Deux familles d'échec qu'il ne faut pas confondre : un compte hors d'usage
(clé refusée, crédits épuisés) désactive la porte pour toute la passe, alors
qu'une requête refusée ne concerne que son offre.
"""

from __future__ import annotations

import httpx
import pytest

from agent_emploi.llm import jev
from agent_emploi.llm.jev import JevClient, JevError, JevUnavailable

ANSWERS = {
    "answers": {
        "adequation": {"type": "score", "score": 2.2, "confidence": 0.8},
        "experience_bloquante": {"type": "noul", "noul": 0.1},
    },
    "usage": {"input_tokens": 1800, "output_tokens": 12},
}


def response(status: int, payload=None, headers: dict | None = None) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        json=payload if payload is not None else {"error": {"type": "x"}},
        headers=headers or {},
        request=httpx.Request("POST", jev.ENDPOINT),
    )


@pytest.fixture
def scripted(monkeypatch):
    """Remplace `httpx.post` par une suite de réponses (ou d'exceptions)."""
    calls: list[dict] = []

    def install(*results):
        queue = list(results)

        def post(url, **kwargs):
            calls.append({"url": url, **kwargs})
            result = queue.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

        monkeypatch.setattr(jev.httpx, "post", post)
        return calls

    return install


def client() -> tuple[JevClient, list[float]]:
    waits: list[float] = []
    return JevClient("sk-test", sleep=waits.append), waits


def ask(c: JevClient, state="état"):
    return c.ask(state, {"q": {"type": "noul", "instructions": "?"}}, idempotency_key="k-1")


class TestSuccess:
    def test_parses_answers_and_billed_tokens(self, scripted):
        calls = scripted(response(200, ANSWERS))
        c, _ = client()
        result = ask(c)

        assert result.answers["adequation"]["score"] == 2.2
        assert result.tokens_in == 1800
        sent = calls[0]
        assert sent["url"] == jev.ENDPOINT
        assert sent["headers"]["Authorization"] == "Bearer sk-test"
        assert sent["headers"]["Idempotency-Key"] == "k-1"
        assert sent["json"]["model"] == "jev-latest"

    def test_oversized_state_is_refused_before_any_request(self, scripted):
        calls = scripted()
        c, _ = client()
        with pytest.raises(JevError, match="8000"):
            ask(c, state="x" * 9000)
        assert calls == []


class TestFailures:
    @pytest.mark.parametrize("status", [401, 402])
    def test_account_errors_mean_unavailable(self, scripted, status):
        scripted(response(status))
        c, waits = client()
        with pytest.raises(JevUnavailable):
            ask(c)
        assert waits == []

    def test_invalid_request_concerns_this_request_only(self, scripted):
        scripted(response(422))
        c, _ = client()
        with pytest.raises(JevError):
            ask(c)

    def test_rate_limit_is_retried_with_announced_delay(self, scripted):
        calls = scripted(
            response(429, headers={"retry-after": "7"}), response(200, ANSWERS)
        )
        c, waits = client()
        assert ask(c).tokens_in == 1800
        assert waits == [7.0]
        # Même clé d'idempotence d'un essai à l'autre : pas de double facturation.
        assert {call["headers"]["Idempotency-Key"] for call in calls} == {"k-1"}

    def test_upstream_and_network_errors_back_off(self, scripted):
        scripted(
            response(502),
            httpx.ConnectError("dns"),
            response(200, ANSWERS),
        )
        c, waits = client()
        ask(c)
        assert waits == [2.0, 4.0]

    def test_gives_up_after_the_retry_budget(self, scripted):
        scripted(*[response(502)] * (jev.MAX_RETRIES + 1))
        c, waits = client()
        with pytest.raises(JevError, match="reprises"):
            ask(c)
        assert len(waits) == jev.MAX_RETRIES

    def test_malformed_body_is_an_error(self, scripted):
        scripted(response(200, {"surprise": True}))
        c, _ = client()
        with pytest.raises(JevError, match="inattendue"):
            ask(c)


@pytest.mark.live
def test_live_contract():
    """Le format de réponse documenté est bien celui que l'API renvoie."""
    import os

    key = os.environ.get("JEVMODEL_API_KEY")
    if not key:
        pytest.skip("JEVMODEL_API_KEY absente")
    result = JevClient(key).ask(
        {"offre": "Stage data scientist, 6 mois, ouvert aux étudiants de master."},
        {
            "niveau": {
                "type": "score",
                "instructions": "Quel niveau d'expérience l'offre exige-t-elle ?",
                "criteria": ["aucun", "junior", "confirmé", "senior"],
            },
            "stage": {"type": "noul", "instructions": "Est-ce un stage ?"},
        },
        idempotency_key="agent-emploi-live-contract",
    )
    assert 0.0 <= float(result.answers["niveau"]["score"]) <= 3.0
    assert result.answers["stage"]["noul"] > 0.5
    assert result.tokens_in > 0
