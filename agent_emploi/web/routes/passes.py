"""Passes en arrière-plan (`/passes`) et consommation LLM (`/conso`)."""

from __future__ import annotations

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from agent_emploi.web import views
from agent_emploi.web.passes import PassBusy, PassRunner, parse_int

router = APIRouter()

#: Périodes proposées sur la page de consommation, en jours (vide : tout).
USAGE_PERIODS = (1, 7, 30)


def _runner(request: Request) -> PassRunner:
    return request.app.state.runner


@router.get("/passes", response_class=HTMLResponse)
def passes(request: Request):
    runner = _runner(request)
    return request.app.state.templates.TemplateResponse(
        request,
        "passes.html",
        {"commands": runner.commands, "runs": runner.history(), "error": None},
    )


@router.post("/passes/{name}")
def start(
    request: Request,
    name: str,
    dry_run: str = Form(""),
    limit: str = Form(""),
    letters: str = Form(""),
):
    runner = _runner(request)
    if name not in runner.commands:
        raise HTTPException(status_code=404, detail="passe inconnue")
    try:
        run = runner.start(
            name,
            dry_run=bool(dry_run),
            limit=parse_int(limit),
            letters=parse_int(letters),
        )
    except PassBusy as exc:
        return request.app.state.templates.TemplateResponse(
            request,
            "passes.html",
            {"commands": runner.commands, "runs": runner.history(), "error": str(exc)},
            status_code=409,
        )
    return RedirectResponse(f"/passes/{run.id}", status_code=303)


@router.get("/passes/{run_id}", response_class=HTMLResponse)
def run_page(request: Request, run_id: str):
    run = _runner(request).get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="passe inconnue")
    # htmx interroge la même adresse toutes les 2 s : il ne reçoit que le
    # journal, qui cesse de s'interroger une fois la passe finie.
    template = "partials/pass_log.html" if request.headers.get("HX-Request") else "pass.html"
    return request.app.state.templates.TemplateResponse(request, template, {"run": run})


@router.get("/conso", response_class=HTMLResponse)
def usage(request: Request):
    days = parse_int(request.query_params.get("jours"))
    if days not in USAGE_PERIODS:
        days = None
    data = views.usage(request.app.state.config.paths.usage_file, days=days)
    return request.app.state.templates.TemplateResponse(
        request, "usage.html", {"u": data, "periods": USAGE_PERIODS}
    )
