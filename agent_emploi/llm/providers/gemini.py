"""Fournisseur Google Gemini — tier gratuit, second modèle du socle gratuit.

Sert de repli à Groq et inversement : deux quotas gratuits indépendants valent
mieux qu'un seul, une saturation ne bloque alors pas la passe.
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

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
TIMEOUT = httpx.Timeout(60.0, connect=10.0)

#: Gemini renvoie le délai dans un bloc `RetryInfo` du corps, au format
#: protobuf `Duration` (« 41s »). Absent quand le quota épuisé est journalier :
#: il n'y a alors rien à attendre dans la passe en cours.
_RETRY_DELAY = re.compile(r'"retryDelay"\s*:\s*"([\d.]+)s"')


def retry_delay(response: httpx.Response) -> float | None:
    """Délai d'attente annoncé par Gemini, en secondes, ou `None`."""
    header = response.headers.get("retry-after")
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    match = _RETRY_DELAY.search(response.text)
    return float(match.group(1)) if match else None


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
            message = (
                f"gemini: HTTP {exc.response.status_code} — {exc.response.text[:300]}"
            )
            if exc.response.status_code == 429:
                raise RateLimited(message, retry_delay(exc.response)) from exc
            if exc.response.status_code == 503:
                # « This model is currently experiencing high demand » : une
                # saturation passagère du côté de Google, pas un refus de la
                # requête. Mesuré sur les modèles flash du tier gratuit, elle
                # touche un appel sur deux aux heures chargées ; le même appel
                # passe souvent quelques secondes plus tard.
                raise TransientError(message) from exc
            raise ProviderError(message) from exc
        except httpx.TransportError as exc:
            raise TransientError(f"gemini: {exc.__class__.__name__} — {exc}") from exc
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
            truncated=candidates[0].get("finishReason") == "MAX_TOKENS",
        )
