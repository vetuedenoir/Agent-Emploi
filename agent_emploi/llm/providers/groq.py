"""Fournisseur Groq — tier gratuit, API compatible OpenAI.

Sert le gros du volume : filtrage d'adéquation, revue, mapping de formulaire.
"""

from __future__ import annotations

import httpx

from agent_emploi.llm.providers.base import Completion, ProviderError

ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
TIMEOUT = httpx.Timeout(60.0, connect=10.0)


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
            raise ProviderError(
                f"groq: HTTP {exc.response.status_code} — {exc.response.text[:300]}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"groq: {exc}") from exc

        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"groq: réponse inattendue — {data}") from exc

        if not text or not text.strip():
            raise ProviderError("groq: réponse vide")

        usage = data.get("usage") or {}
        return Completion(
            text=text.strip(),
            tokens_in=usage.get("prompt_tokens", 0),
            tokens_out=usage.get("completion_tokens", 0),
        )
