import json

import pytest
from pydantic import BaseModel

from agent_emploi.llm.providers.base import Completion, ProviderError
from agent_emploi.llm.router import LlmError, Router, extract_json
from agent_emploi.models import LlmUsage


class FakeProvider:
    """Fournisseur scripté : renvoie des réponses ou lève, dans l'ordre."""

    def __init__(self, name: str, responses: list) -> None:
        self.name = name
        self._responses = list(responses)
        self.calls: list[dict] = []

    def complete(self, **kwargs) -> Completion:
        self.calls.append(kwargs)
        if not self._responses:
            raise ProviderError(f"{self.name}: plus de réponse scriptée")
        result = self._responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture
def router(config, monkeypatch):
    """Routeur dont les fournisseurs sont injectés à la main."""
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    return Router(config)


def install(router: Router, **providers: FakeProvider) -> None:
    router._providers.update(providers)


class TestExtractJson:
    def test_plain_json(self):
        assert extract_json('{"a": 1}') == '{"a": 1}'

    def test_markdown_fenced(self):
        assert json.loads(extract_json('```json\n{"a": 1}\n```')) == {"a": 1}

    def test_surrounded_by_prose(self):
        raw = 'Voici le résultat :\n{"a": 1}\nJ\'espère que ça convient.'
        assert json.loads(extract_json(raw)) == {"a": 1}


class TestRouting:
    def test_uses_configured_model_for_task(self, router):
        groq = FakeProvider("groq", [Completion(text="ok", tokens_in=10, tokens_out=5)])
        install(router, groq=groq)

        assert router.complete("fit_check", "bonjour") == "ok"
        assert groq.calls[0]["model"] == "llama-test"

    def test_unknown_task_is_rejected(self, router):
        with pytest.raises(KeyError, match="tâche LLM inconnue"):
            router.complete("inexistante", "bonjour")

    def test_falls_back_when_primary_fails(self, router):
        groq = FakeProvider("groq", [ProviderError("quota atteint")])
        gemini = FakeProvider("gemini", [Completion(text="repli")])
        install(router, groq=groq, gemini=gemini)

        assert router.complete("fit_check", "bonjour") == "repli"
        assert gemini.calls[0]["model"] == "gemini-test"

    def test_raises_when_all_models_fail(self, router):
        install(
            router,
            groq=FakeProvider("groq", [ProviderError("ko")]),
            gemini=FakeProvider("gemini", [ProviderError("ko aussi")]),
        )
        with pytest.raises(LlmError, match="tous les modèles ont échoué"):
            router.complete("fit_check", "bonjour")

    def test_missing_api_key_falls_back(self, config, monkeypatch):
        """Une clé absente est une erreur de route : le repli doit jouer."""
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        router = Router(config)
        gemini = FakeProvider("gemini", [Completion(text="ok")])
        monkeypatch.setitem(router._providers, "gemini", gemini)

        assert router.complete("fit_check", "bonjour") == "ok"
        assert gemini.calls[0]["model"] == "gemini-test"

    def test_missing_api_key_everywhere_raises_llm_error(self, config, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        with pytest.raises(LlmError, match="GROQ_API_KEY"):
            Router(config).complete("fit_check", "bonjour")

    def test_task_without_fallback_does_not_retry(self, router):
        anthropic = FakeProvider("anthropic", [ProviderError("ko")])
        install(router, anthropic=anthropic)

        with pytest.raises(LlmError):
            router.complete("letter", "rédige")
        assert len(anthropic.calls) == 1


class TestUsageLogging:
    def test_successful_call_is_costed(self, router, config):
        install(
            router,
            anthropic=FakeProvider(
                "anthropic", [Completion(text="lettre", tokens_in=1000, tokens_out=1000)]
            ),
        )
        router.complete("letter", "rédige", job_id="abc")

        lines = config.paths.usage_file.read_text(encoding="utf-8").splitlines()
        usage = LlmUsage.model_validate_json(lines[-1])
        assert usage.task == "letter"
        assert usage.job_id == "abc"
        assert usage.cost_est == pytest.approx(0.03)  # 5 $/M in + 25 $/M out
        assert usage.ok

    def test_free_tier_call_costs_nothing(self, router, config):
        install(router, groq=FakeProvider("groq", [Completion(text="ok", tokens_in=900)]))
        router.complete("fit_check", "bonjour")

        usage = LlmUsage.model_validate_json(
            config.paths.usage_file.read_text(encoding="utf-8").splitlines()[-1]
        )
        assert usage.cost_est == 0.0

    def test_failed_call_is_logged_too(self, router, config):
        install(
            router,
            groq=FakeProvider("groq", [ProviderError("quota")]),
            gemini=FakeProvider("gemini", [Completion(text="ok")]),
        )
        router.complete("fit_check", "bonjour")

        entries = [
            LlmUsage.model_validate_json(line)
            for line in config.paths.usage_file.read_text(encoding="utf-8").splitlines()
        ]
        assert [entry.ok for entry in entries] == [False, True]
        assert "quota" in entries[0].error


class Verdict(BaseModel):
    score: int
    verdict: str


class TestStructured:
    def test_parses_valid_json(self, router):
        install(
            router,
            groq=FakeProvider("groq", [Completion(text='{"score": 80, "verdict": "apply"}')]),
        )
        result = router.structured("fit_check", "évalue", Verdict)
        assert result.score == 80

    def test_requests_json_mode(self, router):
        groq = FakeProvider("groq", [Completion(text='{"score": 1, "verdict": "skip"}')])
        install(router, groq=groq)
        router.structured("fit_check", "évalue", Verdict)
        assert groq.calls[0]["json_mode"] is True

    def test_retries_once_on_invalid_output(self, router):
        groq = FakeProvider(
            "groq",
            [
                Completion(text="désolé, je ne peux pas"),
                Completion(text='{"score": 70, "verdict": "maybe"}'),
            ],
        )
        install(router, groq=groq)

        result = router.structured("fit_check", "évalue", Verdict)
        assert result.verdict == "maybe"
        assert len(groq.calls) == 2
        # La relance rappelle au modèle ce qui n'allait pas.
        assert "invalide" in groq.calls[1]["prompt"]

    def test_gives_up_after_two_attempts(self, router):
        install(
            router,
            groq=FakeProvider("groq", [Completion(text="non"), Completion(text="toujours non")]),
            gemini=FakeProvider("gemini", [Completion(text="non"), Completion(text="non")]),
        )
        with pytest.raises(LlmError, match="JSON invalide"):
            router.structured("fit_check", "évalue", Verdict)
