"""Interface commune aux fournisseurs de modèles."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class ProviderError(RuntimeError):
    """Échec d'appel côté fournisseur (réseau, quota, réponse inexploitable)."""


class TransientError(ProviderError):
    """Panne de transport — le même appel a des chances de passer tel quel.

    Résolution DNS en échec, connexion refusée, délai dépassé : rien de tout
    cela ne dit que le modèle est indisponible, seulement que le réseau l'était
    à cet instant. Basculer sur le repli ne sert à rien — il emprunte le même
    réseau — et compter cet échec comme une panne de route arrête une passe
    entière sur une coupure de quelques secondes. L'appelant recommence.
    """


class RateLimited(ProviderError):
    """Cadence dépassée — l'appel repassera tel quel après un délai.

    Distincte des autres échecs parce que la conduite à tenir n'est pas la
    même : un modèle qui répond « réessayez dans 18 secondes » n'est pas en
    panne, et basculer sur le repli à la première occurrence gaspille le seul
    autre quota gratuit disponible.

    `retry_after` est le délai annoncé par le fournisseur, en secondes, ou
    `None` s'il n'en donne pas — auquel cas l'appelant décide seul.
    """

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class Completion(BaseModel):
    """Résultat d'un appel, indépendant du fournisseur."""

    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    #: Vrai si la génération a été coupée par le plafond de tokens. Le texte est
    #: alors exploitable pour un humain mais pas pour `json.loads` : il manque
    #: la fin. L'appelant peut relancer avec un plafond plus haut.
    truncated: bool = False


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
