"""Construction de l'application FastAPI.

Le serveur n'écoute que sur `127.0.0.1` (voir `cli.cmd_web`) : il n'a ni
authentification ni protection CSRF, et n'en a pas besoin tant qu'il ne sert
que la machine où il tourne.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from agent_emploi.config import Config
from agent_emploi.web import views
from agent_emploi.web.routes import offers
from agent_emploi.web.state import Stores

HERE = Path(__file__).parent


def local_time(value: datetime | None, pattern: str = "%d/%m/%Y %H:%M") -> str:
    """Les journaux sont en UTC ; l'affichage suit l'heure de la machine."""
    if value is None:
        return ""
    return value.astimezone().strftime(pattern)


def build_templates() -> Jinja2Templates:
    templates = Jinja2Templates(directory=HERE / "templates")
    templates.env.filters["local_time"] = local_time
    templates.env.globals["STATE_LABELS"] = views.STATE_LABELS
    return templates


def create_app(config: Config) -> FastAPI:
    app = FastAPI(title="Agent emploi", docs_url=None, redoc_url=None)
    app.state.config = config
    app.state.stores = Stores(config)
    app.state.templates = build_templates()
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request):
        stores: Stores = request.app.state.stores
        data = views.dashboard(stores.seen(), stores.jobs(), config)
        return request.app.state.templates.TemplateResponse(
            request, "dashboard.html", {"data": data}
        )

    app.include_router(offers.router)
    return app
