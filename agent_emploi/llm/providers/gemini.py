"""Fournisseur Google Gemini — tier gratuit, second modèle du socle gratuit.

Sert de repli à Groq et inversement : deux quotas gratuits indépendants valent
mieux qu'un seul, une saturation ne bloque alors pas la passe.
"""

from __future__ import annotations

import httpx

from agent_emploi.llm.providers.base import Completion, ProviderError

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
TIMEOUT = httpx.Timeout(60.0, connect=10.0)


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def complete(
        self,
        *,
        model: str,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> Completion:
        generation_config: dict = {"maxOutputTokens": max_tokens}
        if json_mode:
            generation_config["responseMimeType"] = "application/json"

        payload: dict = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": generation_config,
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        try:
            response = httpx.post(
                f"{BASE_URL}/{model}:generateContent",
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": self._api_key,
                },
                json=payload,
                timeout=TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"gemini: HTTP {exc.response.status_code} — {exc.response.text[:300]}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"gemini: {exc}") from exc

        candidates = data.get("candidates") or []
        if not candidates:
            # Cas courant : la requête a été bloquée par un filtre de sécurité.
            feedback = data.get("promptFeedback", {})
            raise ProviderError(f"gemini: aucune réponse — {feedback or data}")

        parts = candidates[0].get("content", {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts).strip()
        if not text:
            reason = candidates[0].get("finishReason", "inconnu")
            raise ProviderError(f"gemini: réponse vide (finishReason={reason})")

        usage = data.get("usageMetadata") or {}
        return Completion(
            text=text,
            tokens_in=usage.get("promptTokenCount", 0),
            tokens_out=usage.get("candidatesTokenCount", 0),
        )
