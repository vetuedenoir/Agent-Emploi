"""Construction de l'application FastAPI.

Le serveur n'écoute que sur `127.0.0.1` (voir `cli.cmd_web`) et n'a pas
d'authentification. Écouter en local ne suffit pourtant pas : n'importe quelle
page ouverte dans le navigateur peut envoyer un formulaire vers
`127.0.0.1:8000`, ou y pointer un nom de domaine à elle (DNS rebinding). Deux
gardes ferment ces portes : l'en-tête `Host` doit désigner la machine locale, et
une écriture doit venir des pages de l'interface elle-même.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from agent_emploi.agents.fit import FitAgent
from agent_emploi.config import Config
from agent_emploi.web import views
from agent_emploi.web.passes import PassRunner
from agent_emploi.web.routes import actions, manual, offers, passes
from agent_emploi.web.state import Stores

HERE = Path(__file__).parent

#: Noms sous lesquels le serveur accepte d'être joint.
LOCAL_HOSTS = ("127.0.0.1", "localhost")

#: Méthodes qui ne modifient rien, donc sans contrôle d'origine.
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def cross_site(request: Request) -> bool:
    """Vrai si une écriture vient d'une autre page que l'interface.

    `Sec-Fetch-Site` suffit sur un navigateur récent ; `Origin`, que les
    navigateurs joignent à tout POST de formulaire, couvre les autres. Une
    requête sans l'un ni l'autre ne vient pas d'un navigateur (curl, tests) :
    elle ne peut pas être forgée par une page, elle passe.
    """
    fetch_site = request.headers.get("sec-fetch-site")
    if fetch_site is not None:
        return fetch_site not in ("same-origin", "none")
    origin = request.headers.get("origin")
    if origin is None:
        return False
    return origin.split("://", 1)[-1] != request.headers.get("host")


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


def default_fit_agent(config: Config) -> Callable[[], FitAgent]:
    """Fabrique du fit-check, construit à la demande comme dans `cmd_screen`."""

    def build() -> FitAgent:
        from agent_emploi.llm.router import Router
        from agent_emploi.screen import load_cv_text

        return FitAgent(Router(config), load_cv_text(config), config.fit)

    return build


def create_app(
    config: Config,
    *,
    fit_agent: Callable[[], FitAgent] | None = None,
    runner: PassRunner | None = None,
) -> FastAPI:
    app = FastAPI(title="Agent emploi", docs_url=None, redoc_url=None)
    app.state.config = config
    app.state.fit_agent = fit_agent or default_fit_agent(config)
    app.state.stores = Stores(config)
    app.state.runner = runner or PassRunner(config, write_lock=app.state.stores.write_lock)
    app.state.templates = build_templates()
    # Le bandeau « passe en cours » de chaque page.
    app.state.templates.env.globals["runner"] = app.state.runner
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(LOCAL_HOSTS))

    @app.middleware("http")
    async def same_origin_writes(request: Request, call_next):
        if request.method not in SAFE_METHODS and cross_site(request):
            return PlainTextResponse("écriture refusée : origine étrangère", status_code=403)
        return await call_next(request)

    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request):
        stores: Stores = request.app.state.stores
        data = views.dashboard(stores.seen(), stores.jobs(), config)
        return request.app.state.templates.TemplateResponse(
            request,
            "dashboard.html",
            {"data": data, "runs": request.app.state.runner.history()[:5]},
        )

    # Avant `offers` : `/offres/nouvelle` serait sinon lu comme un identifiant.
    app.include_router(manual.router)
    app.include_router(offers.router)
    app.include_router(actions.router)
    app.include_router(passes.router)
    return app
