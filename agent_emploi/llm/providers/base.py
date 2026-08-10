"""Interface commune aux fournisseurs de modèles."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class ProviderError(RuntimeError):
    """Échec d'appel côté fournisseur (réseau, quota, réponse inexploitable)."""


class Completion(BaseModel):
    """Résultat d'un appel, indépendant du fournisseur."""

    text: str
    tokens_in: int = 0
    tokens_out: int = 0


@runtime_checkable
class Provider(Protocol):
    """Contrat minimal qu'un fournisseur doit remplir.

    `json_mode` demande une réponse en JSON pur. Le schéma attendu est décrit
    dans le prompt et validé côté appelant par Pydantic : cette approche marche
    partout, alors que les modes « schéma natif » diffèrent d'un fournisseur à
    l'autre et refusent certaines constructions.
    """

    name: str

    def __init__(self, api_key: str) -> None: ...

    def complete(
        self,
        *,
        model: str,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> Completion: ...
