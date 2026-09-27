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

def busy_refusal(request: Request) -> str | None:
    """Motif de refus d'une écriture pendant une passe, sinon `None`.

    À appeler sous le verrou d'écriture : la passe le prend pour démarrer, donc
    ce qui est vu ici ne change pas avant la fin de l'action.
    """
    current = request.app.state.runner.current
    if current is None:
        return None
    return (
        f"une passe est en cours ({current.label}) : réessayez quand elle sera "
        "terminée, pour ne pas écrire deux fois sur la même offre"
    )


def _done(job_id: str, notice: str) -> RedirectResponse:
    return RedirectResponse(f"/offres/{job_id}?fait={notice}", status_code=303)


def _note(value: str) -> str | None:
    return value.strip() or None


def _run(request: Request, job_id: str, notice: str, action, **extra):
    """Exécute une action sous le verrou d'écriture ; 409 si elle est refusée."""
    stores = request.app.state.stores
    with stores.write_lock:
        refusal = busy_refusal(request)
        if refusal:
            return render_detail(request, job_id, status_code=409, error=refusal, **extra)
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
    """Lance le fit-check en arrière-plan et ouvre son suivi."""
    stores = request.app.state.stores
    with stores.write_lock:
        refusal = busy_refusal(request)
        if refusal:
            return render_detail(request, job_id, status_code=409, error=refusal)
        try:
            run = actions.start_prepare(
                job_id,
                seen=stores.seen(),
                jobs=stores.jobs(),
                config=request.app.state.config,
                runner=request.app.state.runner,
                fit_agent=request.app.state.fit_agent,
            )
        except actions.ActionRefused as exc:
            return render_detail(request, job_id, status_code=409, error=str(exc))
    return RedirectResponse(f"/passes/{run.id}", status_code=303)
