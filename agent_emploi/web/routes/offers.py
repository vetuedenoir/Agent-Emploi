"""Historique des offres, recherche et fiche détaillée."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from agent_emploi.models import JobState
from agent_emploi.web import views

router = APIRouter()


def _filters(request: Request) -> views.OfferFilters:
    """Lit les critères dans l'URL, en ignorant les valeurs hors liste.

    Un lien ancien ou tapé à la main ne doit pas produire une erreur 422 : un
    critère inconnu est simplement sans effet.
    """
    params = request.query_params
    state = params.get("state", "")
    if state not in {s.value for s in JobState}:
        state = ""
    try:
        days = int(params.get("days", ""))
    except ValueError:
        days = None
    if days not in views.PERIODS:
        days = None
    sort = params.get("sort", "date")
    if sort not in views.SORTS:
        sort = "date"
    return views.OfferFilters(
        q=params.get("q", "").strip(),
        state=state,
        source=params.get("source", ""),
        days=days,
        sort=sort,
    )


@router.get("/offres", response_class=HTMLResponse)
def offers(request: Request):
    stores = request.app.state.stores
    seen = stores.seen()
    filters = _filters(request)
    rows = views.offer_rows(seen, stores.jobs(), filters)
    context = {
        "rows": rows,
        "filters": filters,
        "total": len(seen),
        "states": [state for state in JobState],
        "sources": sorted({entry.source for entry in seen.entries()}),
        "periods": views.PERIODS,
    }
    # Une frappe dans la recherche ne redemande que le tableau : le champ garde
    # le focus et le curseur, la page ne clignote pas. Un retour arrière que
    # htmx ne sait pas restaurer depuis son cache redemande, lui, la page entière.
    partial = request.headers.get("HX-Request") and not request.headers.get(
        "HX-History-Restore-Request"
    )
    template = "partials/offer_rows.html" if partial else "offers.html"
    return request.app.state.templates.TemplateResponse(request, template, context)


@router.get("/offres/{job_id}", response_class=HTMLResponse)
def offer(request: Request, job_id: str):
    stores = request.app.state.stores
    detail = views.offer_detail(
        job_id, stores.seen(), stores.jobs(), request.app.state.config
    )
    if detail is None:
        raise HTTPException(status_code=404, detail="offre inconnue")
    return request.app.state.templates.TemplateResponse(
        request, "offer_detail.html", {"d": detail}
    )
