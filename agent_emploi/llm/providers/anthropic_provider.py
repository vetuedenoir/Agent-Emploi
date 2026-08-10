"""Fournisseur Anthropic — tier payant, réservé à la rédaction de la lettre.

Utilise le SDK officiel. La réflexion adaptative est active par défaut sur
Claude Opus 5 : `max_tokens` plafonne la réflexion *et* le texte de réponse,
d'où une marge confortable par défaut.
"""

from __future__ import annotations

import anthropic

from agent_emploi.llm.providers.base import Completion, ProviderError

#: Le préfixe stable (consignes + CV + voix) est identique d'une offre à l'autre :
#: on le met en cache pour ne payer le plein tarif qu'une fois.
_CACHE_CONTROL = {"type": "ephemeral"}


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(
        self,
        *,
        model: str,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> Completion:
        params: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            # Bloc mis en cache : le préfixe ne change pas d'une offre à l'autre.
            params["system"] = [
                {"type": "text", "text": system, "cache_control": _CACHE_CONTROL}
            ]

        try:
            response = self._client.messages.create(**params)
        except anthropic.APIError as exc:
            raise ProviderError(f"anthropic: {exc}") from exc

        # Un refus renvoie un HTTP 200 avec un contenu vide : le vérifier avant
        # de lire `content`, sinon on lève un IndexError trompeur.
        if response.stop_reason == "refusal":
            detail = getattr(response.stop_details, "category", None)
            raise ProviderError(f"anthropic: requête refusée (catégorie={detail})")

        text = "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
        if not text:
            raise ProviderError(
                f"anthropic: réponse vide (stop_reason={response.stop_reason})"
            )

        usage = response.usage
        # Les tokens lus depuis le cache comptent dans l'entrée facturée, à tarif
        # réduit ; on les agrège ici pour que l'estimation reste majorante.
        tokens_in = (
            usage.input_tokens
            + (usage.cache_creation_input_tokens or 0)
            + (usage.cache_read_input_tokens or 0)
        )
        return Completion(
            text=text, tokens_in=tokens_in, tokens_out=usage.output_tokens
        )
