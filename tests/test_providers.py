"""Ce que les fournisseurs doivent lire dans une réponse d'erreur.

Un HTTP 429 n'a pas la même conséquence qu'une panne : le routeur attend et
recommence au lieu de dépenser le quota de repli. Encore faut-il que le délai
annoncé soit extrait, et il ne l'est pas au même endroit chez les deux
fournisseurs gratuits — en-tête chez l'un, corps de la réponse chez l'autre.
"""

from __future__ import annotations

import json

import httpx
import pytest

from agent_emploi.llm.providers.base import (
    ProviderError,
    RateLimited,
    TransientError,
)
from agent_emploi.llm.providers.gemini import GeminiProvider
from agent_emploi.llm.providers.gemini import retry_delay as gemini_delay
from agent_emploi.llm.providers.groq import GroqProvider
from agent_emploi.llm.providers.groq import retry_delay as groq_delay

GROQ_429 = (
    '{"error":{"message":"Rate limit reached for model `llama-3.3-70b-versatile` '
    "in organization `org_x` service tier `on_demand` on tokens per minute (TPM): "
    'Limit 12000, Used 10625, Requested 4985. Please try again in 18.05s."}}'
)

GEMINI_429 = """{
  "error": {
    "code": 429,
    "message": "You exceeded your current quota.",
    "details": [
      {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "41s"}
    ]
  }
}"""


def response(status: int, text: str, headers: dict | None = None) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        text=text,
        headers=headers or {},
        request=httpx.Request("POST", "https://example.invalid"),
    )


class TestGroqRetryDelay:
    def test_reads_the_header_first(self):
        assert groq_delay(response(429, GROQ_429, {"retry-after": "7"})) == 7.0

    def test_falls_back_to_the_message(self):
        """Groq n'envoie pas toujours l'en-tête, mais chiffre toujours le texte."""
        assert groq_delay(response(429, GROQ_429)) == 18.05

    def test_none_when_nothing_is_announced(self):
        assert groq_delay(response(429, '{"error":{"message":"slow down"}}')) is None

    def test_ignores_an_unparsable_header(self):
        headers = {"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}
        assert groq_delay(response(429, GROQ_429, headers)) == 18.05


class TestGeminiRetryDelay:
    def test_reads_the_retry_info_block(self):
        assert gemini_delay(response(429, GEMINI_429)) == 41.0

    def test_none_when_the_quota_is_exhausted_without_a_delay(self):
        """Quota journalier épuisé : il n'y a rien à attendre dans cette passe."""
        body = '{"error":{"code":429,"message":"You exceeded your current quota."}}'
        assert gemini_delay(response(429, body)) is None


class TestErrorClassification:
    """429 doit se distinguer du reste ; 413 non — une requête trop grosse le
    restera à la seconde tentative."""

    @pytest.fixture
    def post(self, monkeypatch):
        def install(resp: httpx.Response):
            monkeypatch.setattr(httpx, "post", lambda *a, **k: resp)

        return install

    def test_groq_429_is_rate_limited(self, post):
        post(response(429, GROQ_429))
        with pytest.raises(RateLimited) as excinfo:
            GroqProvider("k").complete(model="m", prompt="p")
        assert excinfo.value.retry_after == 18.05

    def test_groq_413_is_a_plain_error(self, post):
        post(response(413, '{"error":{"message":"Request too large"}}'))
        with pytest.raises(ProviderError) as excinfo:
            GroqProvider("k").complete(model="m", prompt="p")
        assert not isinstance(excinfo.value, RateLimited)

    def test_gemini_503_is_transient(self, post):
        """« High demand » : Google sature un instant, la requête n'y est pour rien."""
        post(response(503, '{"error":{"code":503,"status":"UNAVAILABLE"}}'))
        with pytest.raises(TransientError):
            GeminiProvider("k").complete(model="m", prompt="p")

    def test_gemini_429_is_rate_limited(self, post):
        post(response(429, GEMINI_429))
        with pytest.raises(RateLimited) as excinfo:
            GeminiProvider("k").complete(model="m", prompt="p")
        assert excinfo.value.retry_after == 41.0


class TestTruncationFlag:
    """Le plafond de tokens atteint doit se voir dans la réponse : c'est ce qui
    permet au routeur de relancer avec plus de place au lieu de demander au
    modèle de corriger une syntaxe qui n'est pas fautive."""

    @pytest.fixture
    def post(self, monkeypatch):
        def install(payload: dict):
            monkeypatch.setattr(
                httpx,
                "post",
                lambda *a, **k: response(200, json.dumps(payload)),
            )

        return install

    def test_groq_reports_a_length_stop(self, post):
        post(
            {
                "choices": [
                    {"message": {"content": '{"a": 1'}, "finish_reason": "length"}
                ]
            }
        )
        assert GroqProvider("k").complete(model="m", prompt="p").truncated

    def test_groq_complete_answer_is_not_truncated(self, post):
        post(
            {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
        )
        assert not GroqProvider("k").complete(model="m", prompt="p").truncated

    def test_gemini_reports_max_tokens(self, post):
        post(
            {
                "candidates": [
                    {
                        "content": {"parts": [{"text": '{"a": 1'}]},
                        "finishReason": "MAX_TOKENS",
                    }
                ]
            }
        )
        assert GeminiProvider("k").complete(model="m", prompt="p").truncated


class TestTransportFailures:
    """Une panne de transport n'a pas la même conséquence qu'une panne de modèle.

    Le routeur recommence sur le même modèle au lieu de brûler le quota de
    repli — encore faut-il que les fournisseurs la nomment.
    """

    @pytest.fixture
    def failing(self, monkeypatch):
        def install(exc: Exception):
            def raise_it(*args, **kwargs):
                raise exc

            monkeypatch.setattr(httpx, "post", raise_it)

        return install

    def test_groq_flags_a_dns_failure_as_transient(self, failing):
        failing(httpx.ConnectError("[Errno -2] Name or service not known"))
        with pytest.raises(TransientError):
            GroqProvider("k").complete(model="m", prompt="p")

    def test_gemini_flags_a_timeout_as_transient(self, failing):
        failing(httpx.ReadTimeout("trop long"))
        with pytest.raises(TransientError):
            GeminiProvider("k").complete(model="m", prompt="p")

    def test_an_http_error_stays_a_plain_provider_error(self, monkeypatch):
        """Un 400 vient du modèle, pas du réseau : le rejouer ne sert à rien."""
        monkeypatch.setattr(
            httpx, "post", lambda *a, **k: response(400, '{"error":"modèle inconnu"}')
        )
        with pytest.raises(ProviderError) as excinfo:
            GroqProvider("k").complete(model="m", prompt="p")
        assert not isinstance(excinfo.value, TransientError)
