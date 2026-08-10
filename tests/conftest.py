import pytest

from agent_emploi.config import Config
from agent_emploi.llm.providers.base import Completion
from agent_emploi.profile import BannedPhrases, Profile

BASE_CONFIG = {
    "profile": {
        "cv_markdown": "profile/cv.md",
        "cv_fr": "profile/cv_fr.pdf",
        "cv_en": "profile/cv_en.pdf",
        "voice": "profile/voice.md",
        "banned_phrases": "profile/banned_phrases.txt",
    },
    "search": {"queries": ["ingénieur IA"]},
    "llm": {
        "tasks": {
            "fit_check": {
                "tier": "free",
                "provider": "groq",
                "model": "llama-test",
                "fallback": {"provider": "gemini", "model": "gemini-test"},
            },
            "letter": {"tier": "paid", "provider": "anthropic", "model": "claude-opus-5"},
            "review": {"tier": "free", "provider": "gemini", "model": "gemini-test"},
        },
        "pricing": {"claude-opus-5": {"input": 5.0, "output": 25.0}},
    },
    "budget": {"daily_usd": 1.0, "daily_calls": 100},
}


@pytest.fixture
def config(tmp_path) -> Config:
    """Configuration valide dont les chemins de données pointent vers tmp_path."""
    data = {
        **BASE_CONFIG,
        "paths": {
            "data": str(tmp_path / "data"),
            "outbox": str(tmp_path / "outbox"),
            "applications": str(tmp_path / "applications"),
        },
    }
    return Config.model_validate(data)


class ScriptedProvider:
    """Fournisseur qui renvoie des réponses préparées, dans l'ordre."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def complete(self, **kwargs) -> Completion:
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("appel LLM non prévu par le scénario de test")
        return Completion(text=self._responses.pop(0), tokens_in=900, tokens_out=250)


@pytest.fixture
def profile(tmp_path) -> Profile:
    """Profil minimal : un CV, une voix remplie, deux formules interdites."""
    directory = tmp_path / "profile"
    directory.mkdir()
    (directory / "cv_fr.pdf").write_bytes(b"%PDF-fr")
    return Profile(
        cv_text="Python, Tensorflow, LLM, RAG. École 42. Perceptron multicouche.",
        voice="## Échantillons\n\n```\nJe vouvoie toujours et je vais droit au but.\n```",
        banned=BannedPhrases(["fort de mon expérience", "passionné par"]),
        cv_fr=directory / "cv_fr.pdf",
        cv_en=directory / "cv_en.pdf",
    )
