"""Adaptateurs par fournisseur de modèles.

Tous exposent la même interface `Provider.complete()`, ce qui permet au routeur
de basculer d'un fournisseur à l'autre sans que les agents en sachent rien.
"""

from agent_emploi.llm.providers.base import (
    Completion,
    Provider,
    ProviderError,
    RateLimited,
    TransientError,
)
from agent_emploi.llm.providers.anthropic_provider import AnthropicProvider
from agent_emploi.llm.providers.gemini import GeminiProvider
from agent_emploi.llm.providers.groq import GroqProvider

#: Fournisseurs disponibles, indexés par le nom utilisé dans `config.yaml`.
REGISTRY: dict[str, type[Provider]] = {
    "anthropic": AnthropicProvider,
    "groq": GroqProvider,
    "gemini": GeminiProvider,
}

__all__ = [
    "REGISTRY",
    "AnthropicProvider",
    "Completion",
    "GeminiProvider",
    "GroqProvider",
    "Provider",
    "ProviderError",
    "RateLimited",
    "TransientError",
]
