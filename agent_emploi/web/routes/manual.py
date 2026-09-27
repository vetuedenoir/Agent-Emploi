"""Ajout d'une offre vue ailleurs : `/offres/nouvelle`.

Un seul formulaire, trois boutons : « Pré-remplir » relit la page de l'offre
et réaffiche le formulaire complété, « Préparer une candidature » et « Suivre
seulement » enregistrent. Le pré-remplissage ne décide de rien : chaque champ
reste à relire et à corriger avant d'enregistrer.
"""

from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import ValidationError

from agent_emploi import manual
from agent_emploi.web import actions, jsonld
from agent_emploi.web.routes.actions import busy_refusal
from agent_emploi.web.routes.offers import render_detail

router = APIRouter()

FIELDS = ("url", "title", "company", "location", "contract", "apply_url", "description")


def _form(request: Request, values: dict, *, status_code: int = 200, **context):
    return request.app.state.templates.TemplateResponse(
        request,
        "offer_new.html",
        {"v": values, "notes": [], "error": None, "existing": None, **context},
        status_code=status_code,
    )


@router.get("/offres/nouvelle", response_class=HTMLResponse)
def new_offer(request: Request):
    values = dict.fromkeys(FIELDS, "")
    values["url"] = request.query_params.get("url", "")
    return _form(request, values)


@router.post("/offres/nouvelle", response_class=HTMLResponse)
def create_offer(
    request: Request,
    action: str = Form(...),
    url: str = Form(""),
    title: str = Form(""),
    company: str = Form(""),
    location: str = Form(""),
    contract: str = Form(""),
    apply_url: str = Form(""),
    description: str = Form(""),
):
    values = {
        "url": url.strip(),
        "title": title.strip(),
        "company": company.strip(),
        "location": location.strip(),
        "contract": contract.strip(),
        "apply_url": apply_url.strip(),
        "description": description.replace("\r\n", "\n").strip(),
    }

    if action == "prefill":
        try:
            found = jsonld.prefill(values["url"])
        except jsonld.PrefillError as exc:
            return _form(request, values, error=f"Pré-remplissage impossible : {exc}")
        # Ce que l'utilisateur a déjà tapé l'emporte sur ce que dit la page.
        for key in FIELDS[1:]:
            values[key] = values[key] or getattr(found, key) or ""
        return _form(request, values, notes=found.notes, prefilled=found.structured)

    if action not in manual.MODES:
        return _form(request, values, status_code=400, error="Action inconnue.")

    missing = [
        label
        for key, label in (("url", "l'URL"), ("title", "l'intitulé"), ("company", "l'entreprise"))
        if not values[key]
    ]
    if action == "prepare" and not values["description"]:
        missing.append("la description (le fit-check la lit)")
    if missing:
        return _form(
            request, values, status_code=400, error=f"Il manque {', '.join(missing)}."
        )

    try:
        job = manual.build_job(**values)
    except ValidationError:
        return _form(request, values, status_code=400, error="URL invalide.")

    stores = request.app.state.stores
    with stores.write_lock:
        refusal = busy_refusal(request)
        if refusal:
            return _form(request, values, status_code=409, error=refusal)
        try:
            manual.add(job, mode=action, seen=stores.seen(), jobs=stores.jobs())
        except manual.DuplicateOffer as exc:
            return _form(
                request, values, status_code=409, error=str(exc), existing=exc.existing
            )
        if action == "track":
            return RedirectResponse(f"/offres/{job.id}?fait=suivie", status_code=303)
        # Sous le même verrou que l'ajout : aucune passe ne peut s'intercaler
        # entre l'enregistrement de l'offre et le lancement de son fit-check.
        try:
            run = actions.start_prepare(
                job.id,
                seen=stores.seen(),
                jobs=stores.jobs(),
                config=request.app.state.config,
                runner=request.app.state.runner,
                fit_agent=request.app.state.fit_agent,
            )
        except actions.ActionRefused as exc:
            return render_detail(request, job.id, error=str(exc))
    return RedirectResponse(f"/passes/{run.id}", status_code=303)
