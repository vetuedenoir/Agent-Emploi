"""Actions sur une offre : lettre, décision, envoi, archivage.

Des formulaires classiques, suivis d'une redirection vers la fiche : ils
fonctionnent sans JavaScript, et recharger la page ne rejoue pas l'action.
"""

from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse

from agent_emploi.web import actions
from agent_emploi.web.routes.offers import render_detail

router = APIRouter()

def _done(job_id: str, notice: str) -> RedirectResponse:
    return RedirectResponse(f"/offres/{job_id}?fait={notice}", status_code=303)


def _note(value: str) -> str | None:
    return value.strip() or None


def _run(request: Request, job_id: str, notice: str, action, **extra):
    """Exécute une action sous le verrou d'écriture ; 409 si elle est refusée."""
    stores = request.app.state.stores
    with stores.write_lock:
        try:
            action(seen=stores.seen(), jobs=stores.jobs())
        except actions.ActionRefused as exc:
            return render_detail(request, job_id, status_code=409, error=str(exc), **extra)
    return _done(job_id, notice)


@router.post("/offres/{job_id}/lettre")
def save_letter(request: Request, job_id: str, text: str = Form("")):
    config = request.app.state.config
    return _run(
        request,
        job_id,
        "lettre",
        lambda **stores: actions.save_letter(job_id, text, config=config, **stores),
        # Le texte refusé est rendu tel quel : le perdre sur une erreur serait
        # pire que l'erreur.
        draft=text,
    )


@router.post("/offres/{job_id}/approuver")
def approve(request: Request, job_id: str, note: str = Form("")):
    config = request.app.state.config
    return _run(
        request,
        job_id,
        "approuvee",
        lambda **stores: actions.decide(
            job_id, approved=True, note=_note(note), config=config, **stores
        ),
    )


@router.post("/offres/{job_id}/rejeter")
def reject(request: Request, job_id: str, note: str = Form("")):
    config = request.app.state.config
    return _run(
        request,
        job_id,
        "rejetee",
        lambda **stores: actions.decide(
            job_id, approved=False, note=_note(note), config=config, **stores
        ),
    )


@router.post("/offres/{job_id}/envoyee")
def mark_sent(request: Request, job_id: str, note: str = Form("")):
    return _run(
        request,
        job_id,
        "envoyee",
        lambda **stores: actions.declare_sent(job_id, note=_note(note), **stores),
    )


@router.post("/offres/{job_id}/archiver")
def archive(request: Request, job_id: str):
    config = request.app.state.config
    return _run(
        request,
        job_id,
        "archivee",
        lambda **stores: actions.archive(job_id, config=config, **stores),
    )


@router.post("/offres/{job_id}/preparer")
def prepare(request: Request, job_id: str):
    fit_agent = request.app.state.fit_agent
    return _run(
        request,
        job_id,
        "preparee",
        lambda **stores: actions.prepare(job_id, fit_agent=fit_agent, **stores),
    )
