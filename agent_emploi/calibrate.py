"""Calibration de la porte Jev sur les verdicts de fit déjà rendus.

Les offres de `jobs.jsonl` notées par le fit-check forment un jeu étiqueté
gratuit : on sait lesquelles le LLM a retenues. Rejouer la porte dessus dit,
pour chaque jeu de seuils, combien d'offres retenues elle aurait perdues et
combien d'appels LLM elle aurait épargnés.

Les réponses de Jev sont mises en cache dans `data/gate_calibration.jsonl` :
les seuils s'appliquent après coup, on peut donc en essayer autant qu'on veut
sans repayer. Rien d'autre n'est écrit — ni la mémoire des offres, ni leurs
dossiers.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

from agent_emploi.agents.gate import GateAgent, JevError, JevUnavailable, decide
from agent_emploi.config import FitConfig, GateConfig
from agent_emploi.llm.budget import BudgetExceeded
from agent_emploi.models import GateVerdict, JobRecord

logger = logging.getLogger(__name__)

#: Seuils essayés. L'échelle d'adéquation va de 0 à 3.
MIN_SCORES = (0.5, 1.0, 1.5, 2.0, 2.5)
MAX_BLOCKINGS = (0.5, 0.7, 0.9)


def fit_accepted(record: JobRecord, config: FitConfig) -> bool:
    """L'étiquette : le fit-check a-t-il laissé passer l'offre ?"""
    fit = record.fit
    return (
        fit is not None
        and fit.verdict in config.accept_verdicts
        and fit.score >= config.min_score
    )


def sample(
    records: list[JobRecord], config: FitConfig, size: int, *, seed: int = 0
) -> list[JobRecord]:
    """Échantillon équilibré : autant d'offres retenues que possible.

    Elles sont rares (22 sur 202 à la première mesure) : un tirage uniforme
    n'en contiendrait presque pas, et c'est précisément sur elles qu'on veut
    savoir si la porte se trompe.
    """
    labelled = [r for r in records if r.fit is not None and r.job.is_enriched]
    labelled.sort(key=lambda r: r.job.id)
    rng = random.Random(seed)
    positives = [r for r in labelled if fit_accepted(r, config)]
    negatives = [r for r in labelled if not fit_accepted(r, config)]
    rng.shuffle(positives)
    rng.shuffle(negatives)
    kept = positives[: size // 2]
    return kept + negatives[: size - len(kept)]


def load_cache(path: Path) -> dict[str, GateVerdict]:
    cache: dict[str, GateVerdict] = {}
    if not path.exists():
        return cache
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            data = json.loads(line)
            cache[data["id"]] = GateVerdict.model_validate(data["verdict"])
        except (ValueError, KeyError):
            continue
    return cache


@dataclass
class Collected:
    verdicts: dict[str, GateVerdict] = field(default_factory=dict)
    queried: int = 0
    errors: list[str] = field(default_factory=list)
    stopped: str | None = None


def collect(gate: GateAgent, records: list[JobRecord], cache_path: Path) -> Collected:
    """Verdicts bruts de Jev pour chaque offre, cache d'abord."""
    cache = load_cache(cache_path)
    result = Collected()
    for record in records:
        job = record.job
        if job.id in cache:
            result.verdicts[job.id] = cache[job.id]
            continue
        try:
            verdict = gate.evaluate(job)
        except (BudgetExceeded, JevUnavailable) as exc:
            result.stopped = str(exc)
            break
        except JevError as exc:
            result.errors.append(f"{job.title}: {exc}")
            continue
        result.queried += 1
        result.verdicts[job.id] = verdict
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("a", encoding="utf-8") as handle:
            line = {"id": job.id, "verdict": verdict.model_dump(mode="json")}
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")
    return result


@dataclass(frozen=True)
class Row:
    min_score: float
    max_blocking: float
    #: Offres retenues par le fit que la porte laisse passer.
    kept: int
    #: Offres refusées par le fit que la porte écarte déjà.
    cut: int


def grid(
    verdicts: dict[str, GateVerdict],
    labels: dict[str, bool],
    min_contract: float,
) -> list[Row]:
    """Rejoue la décision pour chaque jeu de seuils, sans nouvel appel."""
    rows = []
    for min_score in MIN_SCORES:
        for max_blocking in MAX_BLOCKINGS:
            config = GateConfig(
                enabled=True,
                min_score=min_score,
                max_blocking=max_blocking,
                min_contract=min_contract,
            )
            kept = cut = 0
            for job_id, raw in verdicts.items():
                passed = decide(
                    config, raw.adequation, raw.experience_blocking, raw.contract_ok
                ).passed
                if labels[job_id] and passed:
                    kept += 1
                elif not labels[job_id] and not passed:
                    cut += 1
            rows.append(Row(min_score, max_blocking, kept, cut))
    return rows
