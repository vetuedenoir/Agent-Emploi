"""Le dossier soumis à l'utilisateur — `outbox/<date>_<entreprise>_<poste>/`.

C'est là que le système s'arrête et rend la main. Le dossier doit se relire sans
lancer le programme : un navigateur pour `preview.html`, un éditeur de texte
pour le reste. Rien n'y est chiffré, condensé ni indexé — c'est un livrable pour
un humain, pas un format d'échange.

    offre.md      l'annonce complète, telle que la source l'a rendue
    lettre.md     la lettre, seul fichier destiné à être modifié à la main
    cv.pdf        copie du CV joint, dans la langue de l'annonce
    fit.json      verdict d'adéquation et score lexical
    review.json   revue de la lettre
    preview.html  tout le dossier sur une page, pour la relecture
    decision.json la décision de l'utilisateur, une fois le dossier validé

`lettre.md` est copié, pas lié : l'utilisateur peut le corriger sans que le
programme puisse écraser ses corrections.
"""

from __future__ import annotations

import html
import logging
import re
import shutil
import unicodedata
from datetime import datetime
from pathlib import Path

from agent_emploi.models import JobRecord, UserDecision, utcnow

logger = logging.getLogger(__name__)

#: Longueur maximale d'un fragment de nom de dossier. Les intitulés d'annonce
#: dépassent volontiers 100 caractères ; le chemin doit rester utilisable.
MAX_SLUG = 40


def slugify(text: str, *, max_length: int = MAX_SLUG) -> str:
    """Fragment de nom de fichier : sans accents, sans espaces, sans surprise.

    Coupé sur un tiret pour ne pas trancher au milieu d'un mot, ce qui rend les
    dossiers lisibles dans un `ls`.
    """
    decomposed = unicodedata.normalize("NFKD", text.lower())
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    slug = re.sub(r"[^a-z0-9]+", "-", stripped).strip("-")
    if len(slug) <= max_length:
        return slug or "sans-titre"
    cut = slug[:max_length]
    return cut.rsplit("-", 1)[0] if "-" in cut[1:] else cut


def bundle_path(root: Path, record: JobRecord, at: datetime | None = None) -> Path:
    """Chemin du dossier d'une offre : `<date>_<entreprise>_<poste>`.

    L'identifiant de l'offre n'apparaît pas : deux annonces de la même
    entreprise au même intitulé le même jour sont un doublon que le
    dédoublonnage a déjà dû écarter.
    """
    day = (at or utcnow()).strftime("%Y-%m-%d")
    return Path(root) / (
        f"{day}_{slugify(record.job.company)}_{slugify(record.job.title)}"
    )


def offer_markdown(record: JobRecord) -> str:
    """L'annonce en Markdown, avec ce qu'il faut pour candidater à la main."""
    job = record.job
    lines = [f"# {job.title}", "", f"**{job.company}**", ""]

    for label, value in (
        ("Lieu", job.location),
        ("Contrat", job.contract),
        ("Télétravail", job.remote),
        ("Salaire", job.salary),
        ("Publiée le", job.posted_at.strftime("%d/%m/%Y") if job.posted_at else None),
        ("Source", job.source),
        ("ATS", job.ats),
    ):
        if value:
            lines.append(f"- {label} : {value}")

    lines += ["", f"- Annonce : {job.url}"]
    if job.apply_url:
        lines.append(f"- Candidature : {job.apply_url}")

    lines += ["", "## Description", "", job.description or "_(non récupérée)_", ""]
    return "\n".join(lines)


def letter_markdown(record: JobRecord) -> str:
    """La lettre, précédée du strict nécessaire pour la retrouver.

    L'en-tête est en commentaire HTML : le fichier reste modifiable et
    copiable-collable tel quel dans un formulaire, sans avoir à retirer un
    en-tête Markdown que le recruteur verrait.
    """
    letter = record.letter
    header = (
        f"<!-- {record.job.company} — {record.job.title}\n"
        f"     {letter.word_count} mots · langue : {letter.language}\n"
        "     Ce fichier est à vous : corrigez-le librement, il ne sera pas "
        "réécrit. -->\n\n"
    )
    return header + letter.text.strip() + "\n"


#: En-tête de `lettre.md` — un commentaire HTML, donc invisible une fois collé.
_LETTER_HEADER = re.compile(r"\A\s*<!--.*?-->\s*", re.DOTALL)


def read_letter(directory: Path) -> str | None:
    """Relit `lettre.md`, en-tête retiré — l'inverse de `letter_markdown`.

    C'est par là que passent les corrections de l'utilisateur : le fichier lui
    appartient, et c'est sa version, pas celle du modèle, qui doit partir.
    Retourne `None` si le fichier a disparu.
    """
    path = Path(directory) / "lettre.md"
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    return _LETTER_HEADER.sub("", raw).strip()


def write_decision(directory: Path, decision: UserDecision) -> Path:
    """Dépose `decision.json` dans le dossier : la trace lisible du feu vert.

    La décision vit aussi dans `jobs.jsonl`, qui fait foi pour le programme ;
    ce fichier-ci est pour l'humain qui rouvre le dossier six mois plus tard.
    """
    path = Path(directory) / "decision.json"
    path.write_text(decision.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def _preview_html(record: JobRecord, cv_name: str | None) -> str:
    """Une page autonome, sans ressource externe : elle doit s'ouvrir hors ligne."""
    job, letter, fit, review = record.job, record.letter, record.fit, record.review

    def esc(value: object) -> str:
        return html.escape(str(value))

    meta = " · ".join(
        esc(value)
        for value in (job.location, job.contract, job.remote, job.salary)
        if value
    )

    alerts: list[str] = []
    if letter.banned_hits:
        alerts.append(
            "Formules interdites restées dans la lettre : "
            + ", ".join(esc(hit) for hit in letter.banned_hits)
        )
    if review is not None and not review.approved:
        alerts.append("La relecture a refusé cette lettre — voir les remarques.")
    if review is not None and review.unsupported_claims:
        alerts.append(
            "Affirmations non soutenues par le CV : "
            + ", ".join(esc(claim) for claim in review.unsupported_claims)
        )
    if cv_name is None:
        alerts.append(
            f"Aucun CV joint : le fichier pour la langue « {esc(letter.language)} » "
            "est introuvable."
        )

    def section(title: str, items: list[str]) -> str:
        if not items:
            return ""
        entries = "".join(f"<li>{esc(item)}</li>" for item in items)
        return f"<h3>{esc(title)}</h3><ul>{entries}</ul>"

    return f"""<!doctype html>
<html lang="fr">
<meta charset="utf-8">
<title>{esc(job.company)} — {esc(job.title)}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ max-width: 46rem; margin: 2rem auto; padding: 0 1.25rem;
         font: 16px/1.6 system-ui, sans-serif; }}
  h1 {{ margin-bottom: .2rem; font-size: 1.5rem; }}
  .meta {{ color: #6b7280; margin-top: 0; }}
  .alert {{ border-left: 3px solid #d97706; background: #d977061a;
            padding: .6rem .9rem; margin: .5rem 0; }}
  .letter {{ white-space: pre-wrap; border: 1px solid #9ca3af55;
             border-radius: .4rem; padding: 1.1rem; }}
  footer {{ color: #6b7280; font-size: .9rem; margin-top: 2rem; }}
</style>
<h1>{esc(job.title)}</h1>
<p class="meta">{esc(job.company)}{f" — {meta}" if meta else ""}<br>
  <a href="{esc(job.url)}">annonce d'origine</a>
  {f'· <a href="{esc(job.apply_url)}">formulaire de candidature</a>' if job.apply_url else ""}
</p>
{"".join(f'<p class="alert">{alert}</p>' for alert in alerts)}
<h2>Lettre — {letter.word_count} mots ({esc(letter.language)})</h2>
<div class="letter">{esc(letter.text)}</div>
<h2>Adéquation{f" — {fit.score}/100" if fit else ""}</h2>
<p>{esc(fit.reason) if fit else "non évaluée"}</p>
{section("Atouts", fit.matched) if fit else ""}
{section("Manques", fit.gaps) if fit else ""}
{section("Remarques de relecture", review.issues) if review else ""}
{section("Affirmations non soutenues", review.unsupported_claims) if review else ""}
<footer>CV joint : {esc(cv_name) if cv_name else "aucun"} —
  rien n'est envoyé tant que vous n'avez pas validé.</footer>
</html>
"""


def refresh_preview(directory: Path, record: JobRecord) -> Path | None:
    """Réécrit `preview.html` à partir du dossier tel qu'il est maintenant.

    `write_bundle` ne passe qu'une fois, à la rédaction : sans cela, une lettre
    corrigée à la main laisse une preview figée sur la version du modèle. Or
    c'est l'écran sur lequel on relit son propre travail — y voir l'ancien
    texte fait douter de ce qui partira, alors que `lettre.md` fait foi.

    Le CV n'est pas recopié : on constate seulement lequel est déjà là.
    """
    if record.letter is None or not directory.exists():
        return None

    cv_name = next((path.name for path in sorted(directory.glob("cv.*"))), None)
    preview = directory / "preview.html"
    preview.write_text(_preview_html(record, cv_name), encoding="utf-8")
    return preview


def write_bundle(
    root: Path,
    record: JobRecord,
    cv_pdf: Path | None,
    *,
    at: datetime | None = None,
) -> Path:
    """Écrit le dossier de candidature et retourne son chemin.

    Un dossier existant est réécrit : une reprise doit produire l'état courant,
    pas un doublon numéroté. Seule exception, `lettre.md` corrigé à la main —
    il est alors sauvegardé en `lettre.originale.md` avant réécriture.
    """
    if record.letter is None:
        raise ValueError(f"offre {record.job.id}: aucune lettre à livrer")

    directory = bundle_path(root, record, at)
    directory.mkdir(parents=True, exist_ok=True)

    letter_file = directory / "lettre.md"
    fresh = letter_markdown(record)
    if letter_file.exists() and letter_file.read_text(encoding="utf-8") != fresh:
        # Une correction de l'utilisateur ne doit jamais disparaître sans trace.
        letter_file.replace(directory / "lettre.originale.md")
        logger.info("lettre existante conservée en lettre.originale.md (%s)", directory)
    letter_file.write_text(fresh, encoding="utf-8")

    (directory / "offre.md").write_text(offer_markdown(record), encoding="utf-8")
    if record.fit is not None:
        (directory / "fit.json").write_text(
            record.fit.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
    if record.review is not None:
        (directory / "review.json").write_text(
            record.review.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )

    cv_name: str | None = None
    if cv_pdf is not None and cv_pdf.exists():
        cv_name = f"cv{cv_pdf.suffix}"
        shutil.copyfile(cv_pdf, directory / cv_name)

    (directory / "preview.html").write_text(
        _preview_html(record, cv_name), encoding="utf-8"
    )
    return directory
