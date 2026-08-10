"""Interface en ligne de commande.

Les commandes suivent l'avancement du plan : diagnostic, recherche, filtrage,
rédaction, puis validation. La candidature arrive à l'étape suivante.

    python -m agent_emploi doctor    # vérifie l'installation
    python -m agent_emploi search    # cherche et mémorise les offres nouvelles
    python -m agent_emploi screen    # filtre, enrichit et note (fit-check LLM)
    python -m agent_emploi draft     # rédige les lettres et prépare outbox/
    python -m agent_emploi review    # soumet les dossiers à votre validation
    python -m agent_emploi status    # état des offres et consommation LLM
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from agent_emploi.config import Config, load_config, load_dotenv
from agent_emploi.llm.budget import BudgetTracker
from agent_emploi.llm.providers import REGISTRY
from agent_emploi.store.seen import SeenStore

OK = "  ok  "
WARN = " ⚠    "
FAIL = " ko   "


#: Marqueur laissé dans les gabarits de `profile/`. Sa présence signale un
#: fichier créé mais pas encore rempli — cas qui passerait sinon inaperçu.
TEMPLATE_MARKER = "À compléter"


def _is_template(path: Path, *, fenced: bool = False) -> bool:
    """Vrai si le fichier existe mais n'est encore qu'un gabarit.

    `fenced` vise `voice.md`, qui se remplit section par section : il est jugé
    sur ses blocs d'échantillons, pas sur une ligne de gabarit oubliée en bas.
    """
    from agent_emploi.profile import voice_is_filled

    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    return not voice_is_filled(content) if fenced else TEMPLATE_MARKER in content


def _check_profile(config: Config) -> bool:
    """Vérifie la présence des fichiers de profil. Seul le CV texte est bloquant."""
    healthy = True
    required = {"CV (markdown)": config.profile.cv_markdown}
    optional = {
        "CV français (PDF)": config.profile.cv_fr,
        "CV anglais (PDF)": config.profile.cv_en,
        "voix": config.profile.voice,
        "formules interdites": config.profile.banned_phrases,
    }

    for label, path in required.items():
        if not path.exists():
            print(f"{FAIL} {label}: {path} — absent (requis pour le filtrage lexical)")
            healthy = False
        elif _is_template(path):
            print(f"{FAIL} {label}: {path} — gabarit non rempli")
            healthy = False
        else:
            print(f"{OK} {label}: {path}")

    for label, path in optional.items():
        if not path.exists():
            print(f"{WARN} {label}: {path} — absent (requis à partir de l'étape 4)")
        elif _is_template(path, fenced=path == config.profile.voice):
            print(f"{WARN} {label}: {path} — gabarit non rempli (étape 4)")
        else:
            print(f"{OK} {label}: {path}")

    return healthy


def _check_providers(config: Config) -> None:
    """Signale les clés d'API manquantes, sans jamais afficher leur valeur."""
    used: set[str] = set()
    for route in config.llm.tasks.values():
        used.add(route.provider)
        if route.fallback is not None:
            used.add(route.fallback.provider)

    for provider in sorted(used):
        env_var = f"{provider.upper()}_API_KEY"
        if provider not in REGISTRY:
            print(f"{FAIL} {provider}: fournisseur inconnu")
        elif os.environ.get(env_var):
            print(f"{OK} {provider}: {env_var} définie")
        else:
            print(f"{WARN} {provider}: {env_var} absente")


def cmd_doctor(config: Config) -> int:
    print("Configuration")
    print(f"{OK} config.yaml chargée et valide")
    config.paths.ensure()
    print(f"{OK} répertoires de travail: {config.paths.data}, {config.paths.outbox}")

    print("\nProfil")
    profile_ok = _check_profile(config)

    print("\nModèles")
    for task, route in sorted(config.llm.tasks.items()):
        fallback = f" (repli: {route.fallback.provider})" if route.fallback else ""
        print(f"{OK} {task}: {route.provider}/{route.model} [{route.tier}]{fallback}")

    print("\nClés d'API")
    _check_providers(config)

    print("\nBudget")
    tracker = BudgetTracker(config.paths.usage_file, config.budget)
    print(f"{OK} {tracker.summary()}")

    if not profile_ok:
        print("\nAu moins un élément requis manque — voir les lignes 'ko' ci-dessus.")
        return 1
    print("\nSocle opérationnel.")
    return 0


def cmd_search(config: Config, *, limit: int | None, dry_run: bool) -> int:
    from agent_emploi.search import run_search

    config.paths.ensure()
    store = SeenStore(
        config.paths.seen_file, dedup_window_days=config.filters.dedup_window_days
    )
    known_before = len(store)

    report = run_search(config, store, limit=limit, record=not dry_run)

    for error in report.errors:
        print(f"{WARN} {error}")

    print(
        f"{report.found} résultats bruts, {report.duplicates} déjà connus ou "
        f"redondants, {len(report.new)} nouvelles offres"
    )
    if dry_run:
        print("(passe à blanc — rien n'a été enregistré)")

    for job in report.new:
        remote = f" · {job.remote}" if job.remote else ""
        print(f"\n  {job.title}")
        print(f"    {job.company} — {job.location or 'lieu non précisé'}{remote}")
        print(f"    {job.contract or 'contrat non précisé'} · {job.url}")

    if not dry_run:
        print(f"\nMémoire: {known_before} → {len(store)} offres connues")

    # Une source en échec ne doit pas passer inaperçue dans un enchaînement.
    return 1 if report.errors and not report.new else 0


def cmd_screen(config: Config, *, limit: int | None, dry_run: bool) -> int:
    """Cherche, filtre et note : la chaîne complète jusqu'au verdict d'adéquation."""
    from agent_emploi.agents.fit import FitAgent
    from agent_emploi.llm.router import Router
    from agent_emploi.screen import load_cv_text, run_screen
    from agent_emploi.search import run_search
    from agent_emploi.store.jobs import JobStore

    config.paths.ensure()
    try:
        cv_text = load_cv_text(config)
    except FileNotFoundError as exc:
        print(f"{FAIL} {exc}", file=sys.stderr)
        return 1

    store = SeenStore(
        config.paths.seen_file, dedup_window_days=config.filters.dedup_window_days
    )
    search_report = run_search(config, store, limit=limit, record=not dry_run)
    for error in search_report.errors:
        print(f"{WARN} {error}")
    reprises = (
        f", {len(search_report.pending)} reprises" if search_report.pending else ""
    )
    print(
        f"Recherche: {search_report.found} résultats bruts, "
        f"{len(search_report.new)} offres nouvelles{reprises}"
    )
    candidates = search_report.to_screen
    if not candidates:
        return 1 if search_report.errors else 0

    job_store = JobStore(config.paths.jobs_file)
    agent = FitAgent(Router(config), cv_text, config.fit)
    report = run_screen(
        config,
        candidates,
        seen=store,
        job_store=job_store,
        fit_agent=agent,
        cv_text=cv_text,
        record=not dry_run,
    )

    # Une route cassée produit la même erreur pour chaque offre : on n'en montre
    # qu'un échantillon, le compte fait le reste.
    for error in report.errors[:5]:
        print(f"{WARN} {error}")
    if len(report.errors) > 5:
        print(f"{WARN} (+{len(report.errors) - 5} autres erreurs)")

    print(
        f"Filtrage: {report.examined} examinées, {report.enriched} détails "
        f"téléchargés, {report.prescreened} pré-retenues, {report.evaluated} "
        f"évaluées par le LLM"
    )
    if report.reasons:
        motifs = ", ".join(
            f"{motif} {count}" for motif, count in report.reasons.most_common()
        )
        print(f"  écartées: {motifs}")
    if report.stopped:
        print(f"{WARN} passe interrompue — {report.stopped}")
    if dry_run:
        print("(passe à blanc — rien n'a été enregistré)")

    for job, verdict in report.accepted:
        print(f"\n  [{verdict.score:>3}] {job.title}")
        print(f"    {job.company} — {job.location or 'lieu non précisé'} · {job.url}")
        print(f"    {verdict.reason}")
        if verdict.matched:
            print(f"    atouts: {', '.join(verdict.matched[:6])}")
        if verdict.gaps:
            print(f"    manques: {', '.join(verdict.gaps[:6])}")

    if not report.accepted:
        print("\nAucune offre retenue sur cette passe.")

    tracker = BudgetTracker(config.paths.usage_file, config.budget)
    print(f"\nConsommation LLM — {tracker.summary()}")

    return 1 if report.errors and not report.accepted else 0


def cmd_draft(config: Config, *, limit: int | None, dry_run: bool) -> int:
    """Rédige, relit et prépare les dossiers des offres retenues."""
    from agent_emploi.agents.letter import LetterAgent
    from agent_emploi.agents.review import ReviewAgent
    from agent_emploi.draft import run_draft
    from agent_emploi.llm.router import Router
    from agent_emploi.profile import Profile
    from agent_emploi.store.jobs import JobStore

    config.paths.ensure()
    try:
        profile = Profile.load(config)
    except FileNotFoundError as exc:
        print(f"{FAIL} {exc}", file=sys.stderr)
        return 1

    if not profile.has_voice:
        print(
            f"{WARN} {config.profile.voice} vide ou non rempli — les lettres "
            "sonneront génériques (voir README)"
        )

    store = SeenStore(
        config.paths.seen_file, dedup_window_days=config.filters.dedup_window_days
    )
    job_store = JobStore(config.paths.jobs_file)
    router = Router(config)
    report = run_draft(
        config,
        seen=store,
        job_store=job_store,
        profile=profile,
        letter_agent=LetterAgent(router, profile, config.letter),
        review_agent=ReviewAgent(router, profile),
        # La rédaction est l'étape payante : sans consigne explicite, on s'en
        # tient au rythme de candidatures décidé dans `config.yaml`.
        limit=limit if limit is not None else config.apply.max_per_day,
        record=not dry_run,
    )

    for error in report.errors[:5]:
        print(f"{WARN} {error}")
    if len(report.errors) > 5:
        print(f"{WARN} (+{len(report.errors) - 5} autres erreurs)")

    reprises = f", {report.resumed} reprises" if report.resumed else ""
    print(
        f"Rédaction: {report.candidates} offres retenues{reprises}, "
        f"{report.generated} lettres générées ({report.regenerated} reprises), "
        f"{report.reviewed} relectures"
    )
    if report.stopped:
        print(f"{WARN} passe interrompue — {report.stopped}")
    if dry_run:
        print("(passe à blanc — les appels LLM ont eu lieu, rien n'a été écrit)")

    for item in report.prepared:
        fit = item.record.fit
        letter = item.record.letter
        print(f"\n  [{fit.score:>3}] {item.record.job.title}")
        print(f"    {item.record.job.company} · {letter.word_count} mots "
              f"({letter.language})")
        print(f"    {item.directory}")
        for warning in item.warnings:
            print(f"{WARN} {warning}")

    if not report.prepared:
        print("\nAucun dossier préparé — lancer `screen` pour retenir des offres.")
    else:
        print(
            f"\n{len(report.prepared)} dossier(s) prêt(s) à relire "
            f"({len(report.flagged)} avec réserves). Rien n'a été envoyé."
        )

    tracker = BudgetTracker(config.paths.usage_file, config.budget)
    print(f"\nConsommation LLM — {tracker.summary()}")

    return 1 if report.errors and not report.prepared else 0


def cmd_review(
    config: Config,
    *,
    ref: str | None,
    list_only: bool,
    decision: str | None,
    note: str | None,
) -> int:
    """Soumet les dossiers préparés à l'utilisateur. Rien n'avance sans lui."""
    from agent_emploi.profile import BannedPhrases
    from agent_emploi.review_cli import (
        Console,
        approve,
        pending,
        reject,
        resolve,
        run_review,
    )
    from agent_emploi.store.jobs import JobStore

    config.paths.ensure()
    store = SeenStore(
        config.paths.seen_file, dedup_window_days=config.filters.dedup_window_days
    )
    job_store = JobStore(config.paths.jobs_file)
    banned = BannedPhrases.load(config.profile.banned_phrases)

    queue = pending(job_store, store, config)
    if not queue:
        print("Aucun dossier en attente — lancer `draft` pour en préparer.")
        return 0

    if list_only:
        print(f"{len(queue)} dossier(s) en attente de validation :")
        for item in queue:
            score = f"{item.record.fit.score:>3}" if item.record.fit else "  ?"
            flag = f" ({len(item.concerns)} réserve(s))" if item.concerns else ""
            print(f"\n  [{score}] {item.label}{flag}")
            print(f"        {item.directory}")
            print(f"        réf: {item.job_id[:8]}")
        return 0

    # Décision non interactive : utile en script, et seul moyen de trancher un
    # dossier depuis un terminal sans entrée standard.
    if decision is not None:
        if ref is None:
            print(f"{FAIL} --{decision} exige une référence de dossier", file=sys.stderr)
            return 1
        try:
            item = resolve(queue, ref)
        except LookupError as exc:
            print(f"{FAIL} {exc}", file=sys.stderr)
            return 1
        if decision == "approve":
            decided = approve(
                item,
                seen=store,
                job_store=job_store,
                banned=banned,
                config=config,
                note=note,
            )
            for concern in decided.concerns:
                print(f"{WARN} {concern}")
            print(f"✓ approuvé — {item.label}\n  {item.directory}")
        else:
            reject(item, seen=store, job_store=job_store, note=note)
            print(f"✗ rejeté — {item.label}")
        return 0

    report = run_review(
        config,
        seen=store,
        job_store=job_store,
        banned=banned,
        console=Console(),
        only=ref,
    )

    for error in report.errors:
        print(f"{WARN} {error}")

    print(
        f"\nValidation : {len(report.approved)} approuvée(s), "
        f"{len(report.rejected)} rejetée(s), "
        f"{len(report.postponed)} laissée(s) en attente"
    )
    if report.approved:
        print("\nPrêtes pour la candidature (étape 6) :")
        for item in report.approved:
            print(f"  {item.directory}")
            if item.record.job.apply_url:
                print(f"    {item.record.job.apply_url}")
    print("\nRien n'a été envoyé.")
    return 1 if report.errors else 0


def cmd_status(config: Config) -> int:
    store = SeenStore(config.paths.seen_file, dedup_window_days=config.filters.dedup_window_days)
    print(f"Offres connues: {len(store)}")
    counts = store.count_by_state()
    if counts:
        for state, count in sorted(counts.items(), key=lambda item: item[0].value):
            print(f"  {state.value:<15} {count}")
    else:
        print("  (aucune — lancer une recherche)")

    from agent_emploi.store.jobs import JobStore

    retained = JobStore(config.paths.jobs_file)
    if len(retained):
        scored = retained.accepted()
        print(f"\nOffres pré-retenues: {len(retained)} ({len(scored)} notées)")
        for record in scored[:5]:
            fit = record.fit
            print(f"  [{fit.score:>3}] {fit.verdict:<6} {record.job.title}")

        # Un dossier écrit ne dit pas où il en est : c'est la mémoire qui
        # distingue ce qui attend une décision de ce qui l'a déjà reçue.
        from agent_emploi.models import JobState

        by_state: dict[JobState, list[str]] = {}
        for record in retained.records():
            if not record.outbox:
                continue
            entry = store.get(record.job.id)
            if entry is not None and entry.state in (
                JobState.AWAITING_USER,
                JobState.APPROVED,
            ):
                by_state.setdefault(entry.state, []).append(record.outbox)

        for state, label in (
            (JobState.AWAITING_USER, "Dossiers en attente de validation"),
            (JobState.APPROVED, "Dossiers approuvés, prêts pour la candidature"),
        ):
            paths = by_state.get(state, [])
            if paths:
                print(f"\n{label}: {len(paths)}")
                for path in paths[:5]:
                    print(f"  {path}")

    tracker = BudgetTracker(config.paths.usage_file, config.budget)
    print(f"\nConsommation LLM — {tracker.summary()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent-emploi", description="Recherche et candidature assistées."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("config.yaml"), help="chemin de config.yaml"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="vérifie l'installation et la configuration")
    sub.add_parser("status", help="état des offres et consommation LLM")

    search = sub.add_parser("search", help="cherche des offres et mémorise les nouvelles")
    search.add_argument(
        "--limit", type=int, default=None, help="offres visées par requête et par source"
    )
    search.add_argument(
        "--dry-run",
        action="store_true",
        help="affiche sans rien enregistrer (passe rejouable)",
    )

    screen = sub.add_parser(
        "screen", help="cherche, filtre et note les offres (fit-check LLM)"
    )
    screen.add_argument(
        "--limit", type=int, default=None, help="offres visées par requête et par source"
    )
    screen.add_argument(
        "--dry-run",
        action="store_true",
        help="affiche sans rien enregistrer (les appels LLM ont bien lieu)",
    )

    draft = sub.add_parser(
        "draft", help="rédige les lettres et prépare les dossiers outbox/"
    )
    draft.add_argument(
        "--limit",
        type=int,
        default=None,
        help="nombre de lettres à rédiger (défaut: apply.max_per_day)",
    )
    draft.add_argument(
        "--dry-run",
        action="store_true",
        help="n'écrit ni dossier ni mémoire (les appels LLM ont bien lieu)",
    )

    review = sub.add_parser(
        "review", help="valide les dossiers préparés (approuver / éditer / rejeter)"
    )
    review.add_argument(
        "ref",
        nargs="?",
        default=None,
        help="dossier visé : début d'identifiant ou fragment de nom de dossier",
    )
    review.add_argument(
        "--list",
        action="store_true",
        help="liste les dossiers en attente sans rien décider",
    )
    review.add_argument(
        "--approve",
        action="store_true",
        help="approuve le dossier désigné sans passer par le menu",
    )
    review.add_argument(
        "--reject",
        action="store_true",
        help="rejette le dossier désigné sans passer par le menu",
    )
    review.add_argument(
        "--note", default=None, help="motif de rejet ou remarque, joint à la décision"
    )

    args = parser.parse_args(argv)
    # Les clés vivent dans `.env`, à côté de `config.yaml` qui, lui, ne contient
    # aucun secret.
    load_dotenv(args.config.parent / ".env")

    try:
        config = load_config(args.config)
    except FileNotFoundError as exc:
        print(f"{FAIL} {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"{FAIL} config.yaml invalide:\n{exc}", file=sys.stderr)
        return 1

    if args.command == "search":
        return cmd_search(config, limit=args.limit, dry_run=args.dry_run)
    if args.command == "screen":
        return cmd_screen(config, limit=args.limit, dry_run=args.dry_run)
    if args.command == "draft":
        return cmd_draft(config, limit=args.limit, dry_run=args.dry_run)
    if args.command == "review":
        if args.approve and args.reject:
            print(f"{FAIL} --approve et --reject s'excluent", file=sys.stderr)
            return 1
        return cmd_review(
            config,
            ref=args.ref,
            list_only=args.list,
            decision="approve" if args.approve else "reject" if args.reject else None,
            note=args.note,
        )
    return {"doctor": cmd_doctor, "status": cmd_status}[args.command](config)


if __name__ == "__main__":
    raise SystemExit(main())
