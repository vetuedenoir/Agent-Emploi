"""La boucle bout en bout : d'une recherche vide à des dossiers prêts à relire.

Une seule commande enchaîne les passes que l'on lançait jusqu'ici une par une :

    recherche → filtrage → fit-check → lettre → revue → dossier → **arrêt**

Le point d'arrêt n'est pas un réglage, c'est la conception : la boucle mène les
offres jusqu'à `awaiting_user` et s'y tient. Elle n'appelle pas `review`, qui
demande une décision humaine, et n'envoie rien : la candidature se fait à la
main, sur le site de l'offre.

Une passe d'archivage ferme la marche : elle range dans `applications/` les
candidatures que vous avez déclarées envoyées depuis la dernière fois. C'est du
rangement de fichiers, sans appel réseau ni changement d'état.

Rien n'est réimplémenté ici : chaque étape est la fonction de passe existante,
appelée dans l'ordre, avec son propre rapport. Une passe interrompue — plafond
de budget, fournisseur en panne — arrête la boucle proprement et laisse les
offres là où elles en sont ; la relance reprend au même point.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Mapping

from agent_emploi.agents.fit import FitAgent
from agent_emploi.agents.gate import GateAgent
from agent_emploi.agents.letter import LetterAgent
from agent_emploi.agents.review import ReviewAgent
from agent_emploi.config import Config
from agent_emploi.draft import DraftReport, run_draft
from agent_emploi.models import JobRecord
from agent_emploi.profile import CvText, Profile
from agent_emploi.review_cli import Pending, pending
from agent_emploi.screen import ScreenReport, run_screen
from agent_emploi.search import SearchReport, run_search
from agent_emploi.sources import JobSource
from agent_emploi.store.archive import ArchiveReport, run_archive
from agent_emploi.store.jobs import JobStore
from agent_emploi.store.seen import SeenStore

logger = logging.getLogger(__name__)


@dataclass
class PipelineReport:
    """Bilan de la boucle : un rapport par passe, plus ce qui vous attend."""

    search: SearchReport | None = None
    screen: ScreenReport | None = None
    draft: DraftReport | None = None
    archive: ArchiveReport | None = None
    #: Dossiers en attente de votre validation à la fin de la boucle — ceux de
    #: cette passe et ceux qui traînaient déjà.
    awaiting: list[Pending] = field(default_factory=list)
    #: Motif d'arrêt anticipé, hérité de la passe qui s'est arrêtée.
    stopped: str | None = None

    @property
    def errors(self) -> list[str]:
        """Toutes les erreurs rencontrées, dans l'ordre des passes."""
        collected: list[str] = []
        for report in (self.search, self.screen, self.draft, self.archive):
            if report is not None:
                collected += report.errors
        return collected

    @property
    def prepared(self) -> int:
        """Dossiers produits par cette passe."""
        return len(self.draft.prepared) if self.draft else 0


def run_pipeline(
    config: Config,
    *,
    seen: SeenStore,
    job_store: JobStore,
    profile: Profile,
    cvs: str | list[CvText],
    fit_agent: FitAgent,
    letter_agent: LetterAgent,
    review_agent: ReviewAgent,
    limit: int | None = None,
    draft_limit: int | None = None,
    record: bool = True,
    sources: Mapping[str, JobSource] | None = None,
    gate: GateAgent | None = None,
) -> PipelineReport:
    """Déroule la chaîne complète jusqu'aux dossiers à valider.

    `limit` borne la recherche (offres visées par requête et par source),
    `draft_limit` le nombre de lettres — la seule étape payante, qui suit
    `letter.max_per_day` si rien n'est précisé.

    `record=False` fait une passe à blanc de bout en bout : les appels LLM ont
    bien lieu, car c'est le seul moyen de voir ce que la chaîne produit, mais ni
    la mémoire, ni `jobs.jsonl`, ni `outbox/`, ni `applications/` ne sont
    touchés.
    """
    report = PipelineReport()

    report.search = run_search(config, seen, limit=limit, record=record)
    candidates = report.search.to_screen

    if candidates:
        report.screen = run_screen(
            config,
            candidates,
            seen=seen,
            job_store=job_store,
            fit_agent=fit_agent,
            cvs=cvs,
            sources=sources,
            record=record,
            gate=gate,
        )
        report.stopped = report.screen.stopped

    # En passe à blanc, le filtrage n'a rien persisté : les offres qu'il vient
    # de retenir n'existent que dans son rapport, et c'est de là qu'il faut les
    # reprendre pour que la chaîne se déroule vraiment jusqu'au bout.
    drafting: list[JobRecord] | None = None
    if not record:
        drafting = [
            JobRecord(job=job, fit=fit)
            for job, fit in (report.screen.accepted if report.screen else [])
        ]

    # La rédaction a lieu même si le filtrage s'est arrêté : les offres laissées
    # en `fit_ok` par une passe précédente sont déjà payées côté fit-check et
    # n'attendent plus que leur lettre. Si c'est le budget qui a coupé, la passe
    # de rédaction le constatera d'elle-même, avant tout appel.
    report.draft = run_draft(
        config,
        seen=seen,
        job_store=job_store,
        profile=profile,
        letter_agent=letter_agent,
        review_agent=review_agent,
        limit=draft_limit if draft_limit is not None else config.letter.max_per_day,
        record=record,
        candidates=drafting,
    )
    report.stopped = report.draft.stopped or report.stopped

    # Rangement de fin de passe : les candidatures déclarées envoyées depuis la
    # dernière boucle quittent `outbox/` pour `applications/`.
    report.archive = run_archive(
        config, seen=seen, job_store=job_store, record=record
    )

    report.awaiting = pending(job_store, seen, config)
    return report
