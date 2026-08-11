"""Étape 6 — candidature assistée : remplir le formulaire, s'arrêter avant l'envoi.

Le découpage suit la règle du projet : le déterministe d'un côté, le fragile de
l'autre.

    fields.py    appariement champ ↔ information — Python pur, sans navigateur
    identity.py  état civil, lu dans `profile/identity.yaml`
    browser.py   Playwright, contexte persistant — aucune fonction d'envoi
    login.py     connexion aux sites, mot de passe jamais écrit sur disque
    handoff.py   la fiche `candidature.md` remise quand on rend la main
    runner.py    la passe : sélection, remplissage, états, rapport

Rien dans ce package ne soumet un formulaire de candidature.
"""

from agent_emploi.apply.identity import Identity
from agent_emploi.apply.runner import (
    ApplyReport,
    PreparedApplication,
    mark_submitted,
    run_apply,
)

__all__ = [
    "ApplyReport",
    "Identity",
    "PreparedApplication",
    "mark_submitted",
    "run_apply",
]
