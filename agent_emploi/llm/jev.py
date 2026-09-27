"""Client de l'API Jev — un modèle de décision, pas de rédaction.

Jev ne rédige rien : il reçoit un état (≤ 8 000 caractères sérialisés) et des
questions typées, et renvoie pour chacune un choix, un score sur une échelle
ordonnée ou une probabilité oui/non (`noul`). C'est la forme exacte d'un étage
de filtrage, pour un coût d'entrée de 0,042 $ par million de tokens, la sortie
étant gratuite.

Le client ne sait rien des offres : il transporte, reprend sur les pannes
passagères et classe les échecs en deux familles, parce que l'appelant ne doit
pas les traiter de la même façon :

* `JevUnavailable` — le compte est en cause (clé refusée, crédits épuisés).
  Aucune requête suivante ne passera : la porte se désactive pour la passe.
* `JevError` — cette requête-là a échoué (requête refusée, reprises épuisées).
  L'offre suivante peut très bien passer.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

import httpx

logger = logging.getLogger(__name__)

ENDPOINT = "https://jevmodel.org/v1/systemone"
MODEL = "jev-latest"
TIMEOUT = httpx.Timeout(30.0, connect=10.0)

#: Plafond de l'API sur l'état sérialisé, en caractères.
MAX_STATE_CHARS = 8000

#: Reprises consenties sur 429, 502 et pannes de transport. Jev les documente
#: comme rejouables ; l'en-tête `Idempotency-Key` garantit qu'une reprise d'un
#: appel déjà facturé ne l'est pas deux fois.
MAX_RETRIES = 3

#: Première attente avant reprise, doublée à chaque essai : 2, 4, 8 secondes.
BACKOFF = 2.0

#: Statuts qui disent quelque chose du compte, pas de la requête.
_ACCOUNT_STATUSES = frozenset({401, 402})

#: Statuts rejouables tels quels.
_RETRY_STATUSES = frozenset({429, 502})


class JevUnavailable(RuntimeError):
    """Le compte ne permet plus d'appeler Jev : clé refusée ou crédits épuisés."""


class JevError(RuntimeError):
    """Échec d'une requête précise ; les suivantes peuvent réussir."""


@dataclass(frozen=True)
class JevResponse:
    """Réponses brutes, indexées par nom de question, et tokens facturés."""

    answers: dict[str, dict[str, Any]]
    tokens_in: int = 0


def serialized_length(state: Any) -> int:
    """Longueur de l'état, mesurée comme l'API pourrait la mesurer au pire.

    La documentation dit « 8 000 caractères » sans dire lesquels. Trois états
    mesurés à 7 999–8 000 en JSON compact ont été refusés (HTTP 422) : l'API
    compte donc plus large. Deux lectures plausibles l'expliquent, en unités
    UTF-16 (le `.length` de JavaScript, où un émoji compte double) ou en JSON
    avec espaces après `,` et `:`. On retient les deux à la fois : la marge
    coûte quelques dizaines de caractères de description, un refus coûte
    l'offre entière.
    """
    text = json.dumps(state, ensure_ascii=False)
    return len(text.encode("utf-16-le")) // 2


class JevClient:
    """Une requête `systemone` par appel, avec reprises sur les pannes passagères."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = MODEL,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._sleep = sleep

    def ask(
        self,
        state: Any,
        questions: dict[str, dict[str, Any]],
        *,
        idempotency_key: str,
    ) -> JevResponse:
        """Pose les questions sur l'état. Lève `JevUnavailable` ou `JevError`."""
        size = serialized_length(state)
        if size > MAX_STATE_CHARS:
            # Mieux vaut échouer ici qu'être facturé d'un 422.
            raise JevError(f"état de {size} caractères > {MAX_STATE_CHARS}")

        payload = {"model": self.model, "state": state, "questions": questions}
        headers = {**self._headers, "Idempotency-Key": idempotency_key[:100]}

        attempt = 0
        while True:
            try:
                response = httpx.post(
                    ENDPOINT, headers=headers, json=payload, timeout=TIMEOUT
                )
            except httpx.TransportError as exc:
                failure = f"{exc.__class__.__name__} — {exc}"
                retry_after = None
            else:
                status = response.status_code
                if status < 400:
                    return self._parse(response)
                failure = f"HTTP {status} — {response.text[:300]}"
                if status in _ACCOUNT_STATUSES:
                    raise JevUnavailable(f"jev: {failure}")
                if status not in _RETRY_STATUSES:
                    raise JevError(f"jev: {failure}")
                retry_after = _retry_after(response)

            attempt += 1
            if attempt > MAX_RETRIES:
                raise JevError(f"jev: {failure} (après {MAX_RETRIES} reprises)")
            delay = retry_after or BACKOFF * 2 ** (attempt - 1)
            logger.info(
                "jev: %s, reprise dans %.0fs (%d/%d)", failure, delay, attempt, MAX_RETRIES
            )
            self._sleep(delay)

    @staticmethod
    def _parse(response: httpx.Response) -> JevResponse:
        try:
            data = response.json()
            answers = data["answers"]
        except (ValueError, KeyError, TypeError) as exc:
            raise JevError(f"jev: réponse inattendue — {response.text[:300]}") from exc
        if not isinstance(answers, dict):
            raise JevError(f"jev: `answers` inattendu — {answers!r}")
        usage = data.get("usage") or {}
        return JevResponse(answers=answers, tokens_in=int(usage.get("input_tokens", 0)))


def _retry_after(response: httpx.Response) -> float | None:
    header = response.headers.get("retry-after")
    if not header:
        return None
    try:
        return float(header)
    except ValueError:
        return None
