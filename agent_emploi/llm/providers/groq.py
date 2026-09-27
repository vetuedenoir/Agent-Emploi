"""Fournisseur Groq — tier gratuit, API compatible OpenAI.

Sert le gros du volume : filtrage d'adéquation et revue.
"""

from __future__ import annotations

import re

import httpx

from agent_emploi.llm.providers.base import (
    Completion,
    ProviderError,
    RateLimited,
    TransientError,
)

ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
TIMEOUT = httpx.Timeout(60.0, connect=10.0)

#: Groq chiffre l'attente dans le corps de la réponse (« Please try again in
#: 18.05s ») en plus de l'en-tête `retry-after`, qu'il n'envoie pas toujours.
_RETRY_IN = re.compile(r"try again in ([\d.]+)s", re.IGNORECASE)

# `reasoning_effort` est laissé à sa valeur par défaut, délibérément. En
# « low », gpt-oss-120b sort trois fois moins de tokens (~280 contre ~790 par
# fit-check), mais sur 16 offres déjà notées, 3 ont changé de côté du seuil —
# dont une passée de 20 à 72, soit une lettre Opus pour rien. À l'effort par
# défaut, une seule a basculé.


def retry_delay(response: httpx.Response) -> float | None:
    """Délai d'attente annoncé par Groq, en secondes, ou `None`."""
    header = response.headers.get("retry-after")
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    match = _RETRY_IN.search(response.text)
    return float(match.group(1)) if match else None


class GroqProvider:
    name = "groq"

    def __init__(self, api_key: str) -> None:
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def complete(
        self,
        *,
        model: str,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> Completion:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            response = httpx.post(
                ENDPOINT, headers=self._headers, json=payload, timeout=TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            message = f"groq: HTTP {exc.response.status_code} — {exc.response.text[:300]}"
            if exc.response.status_code == 429:
                raise RateLimited(message, retry_delay(exc.response)) from exc
            raise ProviderError(message) from exc
        except httpx.TransportError as exc:
            # DNS, connexion, délai dépassé : la requête n'a pas abouti au
            # modèle. Elle est rejouable telle quelle.
            raise TransientError(f"groq: {exc.__class__.__name__} — {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"groq: {exc}") from exc

        try:
            choice = data["choices"][0]
            text = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"groq: réponse inattendue — {data}") from exc

        if not text or not text.strip():
            raise ProviderError("groq: réponse vide")

        usage = data.get("usage") or {}
        return Completion(
            text=text.strip(),
            tokens_in=usage.get("prompt_tokens", 0),
            tokens_out=usage.get("completion_tokens", 0),
            truncated=choice.get("finish_reason") == "length",
        )
