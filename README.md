# Agent-Emploi

Système multi-agents qui cherche des offres d'emploi en IA, les filtre selon un CV, rédige
une lettre de motivation, et **prépare** la candidature — la validation et l'envoi final
restent toujours entre les mains de l'utilisateur.

## Principe

Le LLM n'intervient que là où il y a du jugement ou de la rédaction. Recherche,
dédoublonnage, scoring lexical, choix du CV et persistance sont du Python déterministe.
D'où un coût de quelques centimes par candidature : les tâches de volume tournent sur des
modèles gratuits, seule la lettre justifie un modèle payant.

## Installation

```bash
uv venv && uv pip install -e ".[dev]"
cp .env.example .env    # puis renseigner les clés d'API
python -m agent_emploi doctor
```

`doctor` liste ce qui manque : fichiers de profil, clés d'API, gabarits non remplis.

## Avancement

| Étape | Contenu | État |
|---|---|---|
| 1 | Socle : modèles, config, mémoire des offres, routeur LLM, budget | ✅ fait |
| 2 | Sourcing Welcome to the Jungle | ✅ fait |
| 3 | Filtrage déterministe + verdict d'adéquation | ✅ fait |
| 4 | Rédaction de la lettre + revue + choix du CV | ✅ fait |
| 5 | CLI de validation utilisateur | ✅ fait |
| 6 | Candidature assistée (Playwright, arrêt avant envoi) | à faire |
| 7 | Archivage + boucle bout en bout | à faire |
| 8 | Source secondaire : France Travail | à faire |

Plan détaillé : `~/.claude/plans/i-would-like-to-vast-catmull.md`

## Le profil

- `profile/cv.md` — version texte du CV (sert au scoring et aux prompts)
- `profile/cv_fr.pdf`, `profile/cv_en.pdf` — pièces jointes, choisies selon la
  langue de l'annonce
- `profile/voice.md` — **déterminant** : sans échantillons de votre écriture, les
  lettres sonneront générées quel que soit le modèle. Le fichier est jugé sur ses
  blocs d'échantillons, pas sur les consignes du gabarit ; celles-ci sont
  retirées avant d'atteindre le modèle.
- `profile/banned_phrases.txt` — déjà pré-rempli, à enrichir au fil des relectures

## Commandes

```bash
python -m agent_emploi doctor              # vérifie l'installation
python -m agent_emploi search --dry-run    # cherche sans rien enregistrer
python -m agent_emploi search --limit 20   # cherche et mémorise les nouvelles
python -m agent_emploi screen              # cherche, filtre et note les offres
python -m agent_emploi screen --dry-run    # idem sans rien enregistrer
python -m agent_emploi draft               # rédige les lettres, prépare outbox/
python -m agent_emploi draft --limit 2     # ne rédige que les 2 mieux notées
python -m agent_emploi review              # valide les dossiers, un par un
python -m agent_emploi review --list       # liste les dossiers en attente
python -m agent_emploi review <réf> --approve   # décide sans passer par le menu
python -m agent_emploi status              # état des offres et consommation LLM
pytest                                     # tests (réseau exclu par défaut)
pytest -m live                             # tests de contrat sur les vraies API
```

## Filtrage

Trois étages, du moins cher au plus cher, chacun ne voyant que ce que le
précédent a laissé passer :

| Étage | Coût | Ce qu'il tranche |
|---|---|---|
| `screen_metadata` | gratuit | termes exclus, contrat, télétravail, ancienneté, titre hors sujet |
| `screen_content` | 1 requête HTTP | terme requis dans la description, score lexical CV ↔ offre |
| `fit_check` | 1 appel LLM (gratuit) | jugement : niveau attendu, exigences bloquantes, langue de l'annonce |

Le score lexical est un cosinus sur vecteurs log-TF, sans dépendance externe :
le même couple (CV, offre) donne toujours la même valeur, ce qui rend le seuil
réglable. Sur le CV actuel, une offre hors sujet tombe vers 0.00 et une offre
pertinente se situe vers 0.06–0.12 ; les descriptions longues diluent le score,
d'où un seuil bas par défaut (`filters.min_lexical_score`).

`filters.require_in_title` est le réglage qui pèse le plus : à `true`, une offre
dont le titre ne contient aucun terme requis est écartée **avant** la requête de
détail. Le passer à `false` élargit la pêche au prix d'une requête HTTP par
offre remontée.

Chaque rejet est persisté dans `seen.jsonl` avec un motif court
(`exclu:commercial`, `lexical:0.031<0.040`, `fit:skip(20)`) : c'est en relisant
ces motifs qu'on règle les seuils. `screen` en affiche le décompte à chaque
passe.

Les offres retenues sont écrites en entier dans `data/jobs.jsonl` avec leur
verdict — l'étape de rédaction les reprend sans réinterroger la source. Une
passe interrompue (plafond de budget, fournisseur en panne) reprend là où elle
s'est arrêtée : les offres restées en `prescreened` sont réévaluées à la passe
suivante.

## Rédaction

`draft` reprend les offres laissées en `fit_ok` par `screen`, les mieux notées
d'abord, et s'arrête au nombre fixé par `apply.max_per_day` — c'est la seule
étape payante, elle ne part pas en volume.

| Étape | Coût | Rôle |
|---|---|---|
| lettre | 1 appel LLM payant | le seul livrable lu par un humain |
| contrôle | gratuit | formules interdites, longueur |
| revue | 1 appel LLM gratuit | inventions, banalité, ton |
| choix du CV | gratuit | `fit.language` → `cv_fr.pdf` ou `cv_en.pdf` |

Le prompt de rédaction est coupé en deux : consignes, CV, voix et liste noire
d'un côté — identiques d'une offre à l'autre, donc mis en cache par l'API — et
l'annonce de l'autre. À partir de la deuxième lettre d'une passe, seule la
partie variable est facturée plein tarif.

Le contrôle des formules interdites est en Python, pas dans le prompt :
demander au modèle de ne pas écrire « fort de mon expérience » en réduit la
fréquence sans la mettre à zéro ; le vérifier après coup, si. Un défaut
constaté — formule interdite, longueur hors bornes, ou revue négative —
déclenche une réécriture. Les deux motifs puisent dans le même compteur
(`letter.max_regenerations`, 1 par défaut) : passé ce quota, la lettre est
**livrée quand même**, avec ses réserves affichées dans le rapport, dans
`preview.html` et dans `review.json`. Rien n'est jeté en silence.

La lettre est écrite dans `jobs.jsonl` dès sa génération : une passe interrompue
entre l'appel payant et l'écriture du dossier ne le refait pas payer.

### Le dossier remis

```
outbox/2026-08-09_acme-ai_stage-ingenieur-ia-generative/
  offre.md      l'annonce complète, avec l'URL de candidature et l'ATS
  lettre.md     la lettre — le seul fichier destiné à être corrigé à la main
  cv.pdf        copie du CV joint, dans la langue de l'annonce
  fit.json      verdict d'adéquation
  review.json   revue de la lettre
  preview.html  tout le dossier sur une page, à ouvrir dans un navigateur
```

Le dossier se relit sans lancer le programme. Une relance réécrit le dossier,
sauf `lettre.md` corrigé à la main : votre version est alors conservée en
`lettre.originale.md`. **Rien n'est envoyé** — l'offre passe en `awaiting_user`
et attend votre validation.

## Validation

`review` reprend les dossiers en `awaiting_user`, les mieux notés d'abord, et
affiche pour chacun la lettre entière, le verdict d'adéquation et les réserves
restantes. Cinq issues :

| Touche | Effet |
|---|---|
| `a` | approuve : l'offre passe en `approved`, seule porte vers l'étape 6 |
| `e` | ouvre `lettre.md` dans `$EDITOR`, puis réaffiche le dossier corrigé |
| `o` | ouvre `preview.html` dans le navigateur |
| `r` | rejette, avec un motif libre |
| `p` / `q` | plus tard : le dossier reste en attente |

Trois points de conception :

- **`lettre.md` fait foi.** Vous corrigez le fichier, pas la base : à
  l'approbation la lettre est relue depuis le dossier, réécrite dans
  `jobs.jsonl` et c'est elle qui partira à l'étape 6. Les formules interdites
  sont revérifiées sur votre version — une correction à la main peut en
  réintroduire.
- **Les réserves n'empêchent pas d'approuver**, elles sont conservées dans la
  décision : une lettre approuvée malgré un signalement reste traçable.
- **Aucun chemin ne contourne cette étape.** La machine à états n'autorise
  `awaiting_user` qu'à aller vers `approved` ou `rejected` ; le remplissage de
  formulaire part de `approved`.

La décision est écrite dans `jobs.jsonl` — qui fait foi pour le programme — et
dans `decision.json` au sein du dossier, pour l'humain qui le rouvrira plus
tard. Les motifs de rejet atterrissent dans `seen.jsonl` avec les rejets
automatiques : c'est la même matière pour régler les seuils.

## Source Welcome to the Jungle

Deux services publics, aucun compte requis :

| Usage | Point d'entrée |
|---|---|
| Recherche | Index Algolia `wk_cms_jobs_production` (~90 000 offres) |
| Détail | `api.welcometothejungle.com/api/v3/organizations/{org}/jobs/{slug}` |

Trois particularités qui expliquent la conception :

1. **L'index ne contient pas les descriptions.** La recherche est donc en deux
   temps : métadonnées pour tout le monde, description seulement pour les offres
   ayant passé le pré-filtrage (une requête par offre).
2. **Une même offre est indexée une fois par job board partenaire qui la
   diffuse** — jusqu'à 5 copies avec des `objectID` différents mais les mêmes
   slugs. La source sur-échantillonne et dédoublonne, sans quoi une seule offre
   remplirait toute la page de résultats.
3. **Le détail expose `apply_url` et `ats`** : la majorité des offres redirigent
   vers un ATS externe (Greenhouse, Lever…). C'est ce qui pilotera l'étape 6.

Aucun de ces points d'entrée n'est contractuel. `pytest -m live` vérifie qu'ils
répondent toujours — en cas d'échec, relever les nouvelles valeurs plutôt que
contourner.

## Structure

```
agent_emploi/
  models.py      Job, JobState, machine à états, identifiants
  config.py      chargement et validation de config.yaml, lecture de .env
  cli.py         doctor / search / screen / draft / review / status
  search.py      passe de recherche : découverte et mémorisation
  filters.py     pré-filtrage déterministe, score lexical CV <-> offre
  screen.py      passe de filtrage : filtres -> enrichissement -> fit-check
  draft.py       passe de rédaction : lettre -> contrôle -> revue -> dossier
  review_cli.py  passe de validation : approuver / éditer / rejeter / plus tard
  profile.py     CV, voix, liste noire, choix du CV (aucun LLM)
  outbox.py      écriture du dossier remis à l'utilisateur
  agents/fit.py     verdict d'adéquation (LLM, tier gratuit)
  agents/letter.py  rédaction de la lettre (LLM, tier payant)
  agents/review.py  relecture de la lettre (LLM, tier gratuit)
  sources/       wttj.py (index Algolia + API v3)
  store/seen.py  mémoire des offres, dédoublonnage
  store/jobs.py  offres retenues : verdict, lettre, revue, dossier
  llm/           routeur, budget, fournisseurs (anthropic, groq, gemini)
config.yaml      tout le réglable : requêtes, filtres, modèles, plafonds
profile/         CV, voix, formules interdites
data/            seen.jsonl, jobs.jsonl, llm_usage.jsonl (non versionnés)
```
