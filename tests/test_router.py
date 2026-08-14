import json

import pytest
from pydantic import BaseModel

from agent_emploi.llm.providers.base import (
    Completion,
    ProviderError,
    RateLimited,
    TransientError,
)
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
def no_sleep(monkeypatch):
    """Enregistre les attentes du routeur au lieu de les subir."""
    waits: list[float] = []
    monkeypatch.setattr("agent_emploi.llm.router.time.sleep", waits.append)
    return waits


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


class TestRateLimit:
    """Une cadence dépassée n'est pas une panne : le même appel repassera.

    Basculer sur le repli dès le premier 429 dépense l'autre quota gratuit
    pour rien, et le laisse indisponible le jour où la panne est réelle.
    """

    def test_waits_the_announced_delay_then_retries_same_model(
        self, router, no_sleep
    ):
        groq = FakeProvider(
            "groq", [RateLimited("groq: 429", 5.0), Completion(text="ok")]
        )
        gemini = FakeProvider("gemini", [Completion(text="repli")])
        install(router, groq=groq, gemini=gemini)

        assert router.complete("fit_check", "bonjour") == "ok"
        assert no_sleep == [5.0]
        assert gemini.calls == []

    def test_falls_back_when_the_wait_is_too_long(self, router, no_sleep):
        """Un délai de plusieurs minutes est un quota épuisé, pas une cadence."""
        groq = FakeProvider("groq", [RateLimited("groq: 429", 3600.0)])
        gemini = FakeProvider("gemini", [Completion(text="repli")])
        install(router, groq=groq, gemini=gemini)

        assert router.complete("fit_check", "bonjour") == "repli"
        assert no_sleep == []

    def test_gives_up_on_the_model_after_two_waits(self, router, no_sleep):
        groq = FakeProvider("groq", [RateLimited("groq: 429", 1.0)] * 3)
        gemini = FakeProvider("gemini", [Completion(text="repli")])
        install(router, groq=groq, gemini=gemini)

        assert router.complete("fit_check", "bonjour") == "repli"
        assert len(groq.calls) == 3
        assert no_sleep == [1.0, 1.0]

    def test_uses_a_default_wait_when_no_delay_is_given(self, router, no_sleep):
        groq = FakeProvider("groq", [RateLimited("groq: 429"), Completion(text="ok")])
        install(router, groq=groq)

        assert router.complete("fit_check", "bonjour") == "ok"
        assert no_sleep == [10.0]


class TestTransientFailures:
    """Une coupure réseau ne dit rien du modèle, et rien de la route.

    Le repli emprunte le même réseau : y basculer ne règle rien. Et compter ces
    échecs comme une route cassée arrête une passe entière — c'est ce qui s'est
    produit sur une résolution DNS momentanément indisponible.
    """

    def test_retries_the_same_model_with_a_growing_wait(self, router, no_sleep):
        groq = FakeProvider(
            "groq",
            [TransientError("groq: ConnectError"), Completion(text="ok")],
        )
        gemini = FakeProvider("gemini", [Completion(text="repli")])
        install(router, groq=groq, gemini=gemini)

        assert router.complete("fit_check", "bonjour") == "ok"
        assert no_sleep == [2.0]
        assert gemini.calls == []

    def test_gives_up_after_three_retries(self, router, no_sleep):
        groq = FakeProvider("groq", [TransientError("groq: ConnectError")] * 4)
        gemini = FakeProvider("gemini", [Completion(text="repli")])
        install(router, groq=groq, gemini=gemini)

        assert router.complete("fit_check", "bonjour") == "repli"
        assert len(groq.calls) == 4
        assert no_sleep == [2.0, 4.0, 8.0]

    def test_marks_the_error_transient_when_nothing_but_the_network_failed(
        self, router, no_sleep
    ):
        """L'appelant doit pouvoir dire « réseau coupé » plutôt que « route morte »."""
        install(
            router,
            groq=FakeProvider("groq", [TransientError("groq: ConnectError")] * 4),
            gemini=FakeProvider("gemini", [TransientError("gemini: ConnectError")] * 4),
        )

        with pytest.raises(LlmError) as excinfo:
            router.complete("fit_check", "bonjour")
        assert excinfo.value.transient is True
        assert "réseau injoignable" in str(excinfo.value)

    def test_a_real_provider_error_is_not_transient(self, router, no_sleep):
        install(
            router,
            groq=FakeProvider("groq", [TransientError("groq: ConnectError")] * 4),
            gemini=FakeProvider("gemini", [ProviderError("gemini: modèle inconnu")]),
        )

        with pytest.raises(LlmError) as excinfo:
            router.complete("fit_check", "bonjour")
        assert excinfo.value.transient is False


class Reason(BaseModel):
    reason: str


class TestTruncatedOutput:
    """Un JSON coupé par le plafond de tokens n'est pas un JSON fautif.

    Redemander « corrige ton erreur » reproduirait la même coupure : c'est la
    place qui manque, pas la syntaxe.
    """

    def test_retries_with_more_room(self, router):
        groq = FakeProvider(
            "groq",
            [
                Completion(text='{"reason": "trop lo', truncated=True),
                Completion(text='{"reason": "ok"}'),
            ],
        )
        install(router, groq=groq)

        assert router.structured("fit_check", "juge", Reason).reason == "ok"
        assert groq.calls[1]["max_tokens"] == 2 * groq.calls[0]["max_tokens"]

    def test_asks_for_brevity_rather_than_a_correction(self, router):
        groq = FakeProvider(
            "groq",
            [
                Completion(text='{"reason": "trop lo', truncated=True),
                Completion(text='{"reason": "ok"}'),
            ],
        )
        install(router, groq=groq)
        router.structured("fit_check", "juge", Reason)

        assert "coupée avant la fin" in groq.calls[1]["prompt"]
        assert "invalide" not in groq.calls[1]["prompt"].rsplit("schéma", 1)[-1]

    def test_reports_the_truncation_when_it_persists(self, router):
        groq = FakeProvider(
            "groq", [Completion(text='{"reason": "trop lo', truncated=True)] * 2
        )
        install(router, groq=groq)

        with pytest.raises(LlmError, match="tronquée"):
            router.structured("fit_check", "juge", Reason)


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
