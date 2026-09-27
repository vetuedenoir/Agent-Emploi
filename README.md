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

Pour l'étape 6, qui pilote un navigateur :

```bash
uv pip install -e ".[apply]" && playwright install chromium
python -m agent_emploi apply --init-identity   # crée profile/identity.yaml
```

## Avancement

| Étape | Contenu | État |
|---|---|---|
| 1 | Socle : modèles, config, mémoire des offres, routeur LLM, budget | ✅ fait |
| 2 | Sourcing Welcome to the Jungle | ✅ fait |
| 3 | Filtrage déterministe + verdict d'adéquation | ✅ fait |
| 4 | Rédaction de la lettre + revue + choix du CV | ✅ fait |
| 5 | CLI de validation utilisateur | ✅ fait |
| 6 | Candidature assistée (Playwright, arrêt avant envoi) | ✅ fait |
| 7 | Archivage + boucle bout en bout | ✅ fait |
| 8 | Source secondaire : France Travail | ✅ fait |

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
- `profile/identity.yaml` — état civil et liens, pour remplir les formulaires à
  l'étape 6. Créé par `apply --init-identity`, jamais versionné. Un champ laissé
  vide n'est pas une erreur : il restera simplement à remplir à la main.

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
python -m agent_emploi apply               # remplit les formulaires, sans envoyer
python -m agent_emploi apply --login wttj  # se connecte d'abord au site
python -m agent_emploi --no-ask search     # ne demande aucun identifiant à l'invite
python -m agent_emploi apply <réf> --sent  # enregistre un envoi que vous avez fait
python -m agent_emploi run                 # la chaîne entière, jusqu'à validation
python -m agent_emploi run --dry-run       # idem sans rien enregistrer
python -m agent_emploi run --letters 2     # borne la seule étape payante
python -m agent_emploi archive             # classe les candidatures envoyées
python -m agent_emploi archive --dry-run   # liste sans rien déplacer
python -m agent_emploi status              # état des offres et consommation LLM
python -m agent_emploi calibrate-gate      # compare les seuils de la porte Jev
python -m agent_emploi web                 # interface web sur 127.0.0.1:8000
pytest                                     # tests (réseau et navigateur exclus)
pytest -m live                             # tests de contrat sur les vraies API
pytest -m browser                          # tests sur un vrai Chromium
```

## Interface web

```bash
uv pip install -e ".[web]"
python -m agent_emploi web --port 8000
```

Le serveur n'écoute que sur `127.0.0.1` : les pages n'ont pas
d'authentification. Il refuse aussi un en-tête `Host` étranger (DNS
rebinding) et toute écriture venue d'une autre page que la sienne : sans cela,
un site ouvert dans le navigateur pourrait approuver un dossier. Il appelle
les mêmes fonctions que la CLI et relit
`seen.jsonl` et `jobs.jsonl` dès qu'ils changent. Une passe lancée dans un
terminal apparaît donc sans redémarrage.

- **Tableau de bord** : offres par état, dossiers en attente, consommation du
  jour, dernière activité.
- **Offres** : l'historique complet, rejets et motifs compris. Il se filtre
  par état, source et période, se trie par date ou par score, et se cherche
  par entreprise ou intitulé à la frappe (sans tenir compte des accents).
- **Fiche d'une offre** : liens vers l'annonce et la candidature, description,
  verdicts (Jev, adéquation, relecture), décision, résultat du remplissage,
  dossiers, et la frise de toutes les transitions. La lettre affichée est
  `lettre.md` quand le dossier existe, puisque c'est lui qui fait foi.

Depuis la fiche, les mêmes actions qu'en CLI :

- **Corriger la lettre**, avec un compteur de mots en direct. L'enregistrement
  réécrit `lettre.md` et `preview.html` dans le dossier, et `jobs.jsonl` avec
  `edited=True`. Les formules interdites sont revérifiées. La lettre reste
  modifiable jusqu'à l'approbation incluse, puis passe en lecture seule une
  fois le formulaire pré-rempli.
- **Approuver ou rejeter** un dossier en attente, comme `review`. Un motif de
  rejet sert à régler les filtres. Un dossier approuvé, pré-rempli ou rendu à
  la main peut encore être abandonné.
- **Déclarer envoyée** une candidature pré-remplie ou rendue à la main, comme
  `apply <réf> --sent`, puis **l'archiver** dans `applications/`.

Le remplissage lui-même (`apply`) reste en CLI : il ouvre un navigateur visible
et peut demander un mot de passe à l'invite. La fiche d'une offre approuvée
affiche la commande exacte à lancer. Une action devenue caduque (l'offre a
avancé entre-temps via une passe CLI) est refusée avec un message, sans rien
écrire.

## Filtrage

Quatre étages, du moins cher au plus cher, chacun ne voyant que ce que le
précédent a laissé passer :

| Étage | Coût | Ce qu'il tranche |
|---|---|---|
| `screen_metadata` | gratuit | termes exclus, contrat, télétravail, ancienneté, titre hors sujet |
| `screen_content` | 1 requête HTTP | terme requis dans la description, score lexical CV ↔ offre |
| porte Jev | 1 appel Jev (~0,0001 $) | expérience exigée bloquante, contrat, adéquation d'ensemble |
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

La porte Jev (`gate` dans `config.yaml`, clé `JEVMODEL_API_KEY`) existe parce
que le score lexical ne départage pas les offres qui arrivent jusqu'au LLM :
mesuré sur 202 verdicts, les offres retenues ont un score moyen de 0,101, les
refusées de 0,105 — et 9 sur 10 étaient refusées, surtout sur l'expérience
exigée ou le contrat. Jev répond à ces questions-là par des probabilités, pour
un coût négligeable. Elle est optionnelle et ne bloque jamais une passe : sans
clé, ou crédits épuisés, elle se retire et le fit-check tranche seul. Ses seuils
se règlent avec `calibrate-gate`, qui la rejoue sur les offres déjà notées et
met les réponses en cache (`data/gate_calibration.jsonl`) : essayer d'autres
seuils ne coûte rien.

Chaque rejet est persisté dans `seen.jsonl` avec un motif court
(`exclu:commercial`, `lexical:0.031<0.040`, `jev:experience(0.82)`, `fit:skip(20)`) : c'est en relisant
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

## Candidature

`apply` reprend les dossiers en `approved`, ouvre l'URL de candidature dans un
Chromium visible, remplit ce qu'il sait remplir, prend une capture — et
s'arrête. Les onglets restent ouverts : vous vérifiez, vous complétez, vous
envoyez.

**Aucune fonction d'envoi n'existe dans le code.** Ce n'est pas un réglage
qu'on pourrait inverser par erreur : `apply/browser.py` n'expose ni `submit` ni
`click`, et un test le vérifie à chaque passe. `apply.stop_before_submit` reste
dans `config.yaml` pour mémoire, sans effet.

L'appariement des champs est déterministe (`apply/fields.py`) : un tableau de
motifs sur le libellé, le `name`, l'`id` et le `placeholder`. Savoir que
« Prénom » attend un prénom ne justifie pas un appel LLM, et un tableau ne se
trompe pas deux fois de la même façon. Trois règles de prudence :

- **Les cases à cocher et les listes déroulantes ne sont jamais remplies**,
  même reconnues. Consentement RGPD, disponibilité, autorisation de travail :
  ce sont des déclarations, elles vous appartiennent — vous êtes devant
  l'écran. Elles apparaissent dans les « à faire ».
- **Un champ obligatoire non reconnu annule tout le remplissage.** Un
  formulaire à moitié rempli qu'on ne peut pas finir est plus déroutant qu'une
  page vierge accompagnée de la fiche.
- **La lettre et le CV viennent du dossier `outbox/`**, pas de la base : c'est
  la version que vous avez relue, corrections comprises, qui part.

Quand les motifs butent sur un libellé inattendu — « Comment devons-nous vous
joindre ? » — un appel du tier gratuit (`form_mapping`) est tenté en dernier
ressort. Son rôle est borné : **le modèle désigne un emplacement, jamais une
valeur.** Il répond « ce champ attend l'adresse électronique » ; c'est le code
qui va chercher l'adresse dans `identity.yaml` et qui la pose. Un modèle qui
divague ne peut donc pas inventer un numéro de téléphone — au pire il se trompe
de case, ce qui se voit dans le rapport comme dans le navigateur. Les fichiers
et les cases à cocher lui échappent entièrement, et un champ que les motifs ont
déjà reconnu n'est jamais réattribué.

L'appel n'a lieu que si le déterministe a échoué, et jamais sur un formulaire
qu'il a su remplir. Sans clé d'API, sans budget ou avec `--no-llm`, la passe se
déroule à l'identique : les champs inconnus repartent en `handoff`.

| Issue | État | Ce que vous trouvez |
|---|---|---|
| formulaire rempli | `prefilled` | l'onglet ouvert, `formulaire.png`, `candidature.md` |
| main rendue | `handoff` | `candidature.md` : URL directe, valeurs à recopier, lettre entière |

Captcha, connexion requise, formulaire méconnaissable, page inaccessible :
autant de `handoff`. Ce n'est pas un incident, c'est le fonctionnement normal
d'un système qui refuse de deviner — et jamais un blocage silencieux, la fiche
contient de quoi candidater à la main en cinq minutes.

`apply` ne fait pas passer une offre en `submitted` : le programme n'envoie
rien, il ne peut donc que prendre acte. Une fois la candidature envoyée de
votre main, `apply <réf> --sent` l'enregistre — et c'est cette déclaration qui
rend le dossier archivable.

### Connexion aux sites

Le contexte Chromium est persistant (`apply.browser.user_data_dir`) : une
session ouverte une fois est conservée d'une passe à l'autre. Vous pouvez donc
vous connecter à la main dans la fenêtre, ou laisser le programme le faire :

```bash
python -m agent_emploi apply --login wttj
python -m agent_emploi apply --login france_travail
```

Le mot de passe est lu dans `<SITE>_PASSWORD` (fichier `.env`, hors dépôt) ou
demandé à l'invite en saisie masquée. Il ne va **ni dans `config.yaml`, ni dans
un journal, ni dans une trace d'exception** — `apply/login.py` n'écrit aucun
fichier, et un test le vérifie. Un captcha ou une double authentification
interrompt la tentative et vous laisse finir dans la fenêtre ouverte : la
session ainsi obtenue est conservée comme si le programme l'avait faite.

La plupart des ATS (Greenhouse, Lever) acceptent une candidature sans compte :
la connexion ne sert que là où elle est exigée.

## La boucle

`run` enchaîne en une commande ce que l'on lançait passe par passe :

```
recherche → filtrage → fit-check → lettre → revue → dossier → arrêt
```

Le point d'arrêt n'est pas un réglage, c'est la conception : la boucle mène les
offres jusqu'à `awaiting_user` et s'y tient. Elle n'appelle ni `review`, qui
demande une décision humaine, ni `apply`, qui ouvre un navigateur —
`manager.py` n'importe rien de `apply/`, et un test le vérifie.

Rien n'y est réimplémenté : chaque étape est la passe existante, appelée dans
l'ordre avec son propre rapport. Une interruption — plafond de budget,
fournisseur en panne — arrête la boucle proprement et laisse chaque offre où
elle en est ; la relance reprend au même point.

Deux limites distinctes, parce que les étapes n'ont pas le même coût :
`--limit` borne la recherche (offres par requête et par source), `--letters`
borne la rédaction, seule étape payante, qui suit `apply.max_per_day` par
défaut.

`--dry-run` déroule la chaîne entière sans rien écrire. Les appels LLM ont bien
lieu — c'est le seul moyen de voir ce que la chaîne produit — mais comme le
filtrage ne persiste rien, la rédaction reprend ses offres directement dans le
rapport du filtrage plutôt que dans `jobs.jsonl`. Sans cela une passe à blanc
s'arrêterait au filtrage et ne montrerait jamais de lettre.

## Archivage

Une candidature envoyée n'a plus rien à faire dans `outbox/`, qui est la pile
des dossiers en cours. `archive` la déplace vers `applications/`, pièces
comprises :

```
applications/2026-08-11_acme-ai_stage-ingenieur-ia-generative/
  README.md     la fiche de suivi — relance, réponse, entretien
  lettre.md     la lettre telle qu'elle est partie
  cv.pdf        le CV joint
  offre.md      l'annonce, fit.json, review.json, decision.json…
```

Deux destinations, deux publics : `seen.jsonl` garde l'état final et son motif,
ce qui empêche de retraiter une offre et sert à régler les seuils ;
`applications/` garde le dossier lisible, pour vous.

- **Seul un envoi déclaré ferme une candidature.** `prefilled` et `handoff` ne
  sont pas des fins : la main est encore à vous. C'est `apply <réf> --sent` qui
  clôt le dossier, et donc lui seul qui le rend archivable. `--include-rejected`
  classe aussi les dossiers que vous avez rejetés avant envoi.
- **`README.md` n'est jamais réécrit.** Le programme l'écrit une fois — avec la
  date d'envoi, une date de relance à J+14, les liens et le verdict — puis n'y
  touche plus : une relance notée à la main ne doit pas disparaître à la passe
  suivante.
- **La copie précède le retrait.** Le dossier d'origine n'est supprimé qu'une
  fois toutes ses pièces recopiées ; une interruption le laisse au pire aux deux
  endroits, jamais à aucun. `--keep` conserve la copie dans `outbox/`.

L'archivage ne change aucun état : c'est un rangement de fichiers, pas une
transition. `run` en fait une passe à la fin de chaque boucle.

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

## Source France Travail

L'inverse de WTTJ, et c'est bien l'intérêt : une API officielle, documentée et
versionnée, qui continuera de répondre le jour où l'index Algolia fermera.

### Deux comptes France Travail, à ne pas confondre

C'est le piège de cette source, et il n'a rien d'évident :

| | Compte candidat | Compte développeur |
|---|---|---|
| Où | `francetravail.fr`, votre espace personnel | [`francetravail.io`](https://francetravail.io) |
| À quoi il sert | **postuler** sur les offres hébergées par France Travail | **chercher** des offres via l'API |
| Ce qu'on en tire | un e-mail et un mot de passe | un identifiant client et une clé secrète |
| Variables | `FRANCE_TRAVAIL_EMAIL` / `_PASSWORD` | `FRANCE_TRAVAIL_CLIENT_ID` / `_CLIENT_SECRET` |
| Où ça sert | `apply --login france_travail` (étape 6) | la source, dès `search` |

**Votre compte candidat n'ouvre pas l'API.** Le compte développeur est gratuit
et distinct : il se crée en deux minutes, on y déclare une application, on lui
ajoute l'API « Offres d'emploi v2 », et l'on récupère les deux valeurs.

### Fournir les identifiants d'API

Trois façons, dans l'ordre où elles sont tentées :

1. l'environnement (`export FRANCE_TRAVAIL_CLIENT_ID=…`) ;
2. le fichier `.env`, hors dépôt ;
3. **une saisie en début de commande**, si rien n'est trouvé :

```
$ python -m agent_emploi search
 ⚠    france_travail: identifiants absents. Saisissez-les pour cette commande,
      ou laissez vide pour ignorer la source. Rien n'est écrit sur disque.
FRANCE_TRAVAIL_CLIENT_ID : PAR_agentemploi_a1b2c3
FRANCE_TRAVAIL_CLIENT_SECRET :
```

C'est le même principe que `apply --login` : la clé secrète est saisie en
masqué, ne vit que le temps de la commande, et n'est écrite **ni dans un
fichier, ni dans un journal**. Elle n'est pas mémorisée d'une commande à
l'autre — pour ne la saisir qu'une fois, mieux vaut `.env`.

L'invite n'apparaît que sur un vrai terminal : sous `cron`, dans un tube ou
avec `--no-ask`, la question n'est pas posée et la source est simplement
ignorée — une tâche planifiée ne doit pas se bloquer sur une question que
personne ne lira. Laisser vide, ou faire Ctrl-C, revient au même : la passe
continue sans cette source.

**Sans identifiants, la source est ignorée** — annoncée une fois dans le
rapport de recherche, et signalée par `doctor` dans sa section « Sources ». La
passe continue sur les autres sources : une source non configurée ne fait
échouer personne. C'est le même mécanisme qui protège d'une faute de frappe
dans `search.sources`, à ceci près qu'une source *inconnue*, elle, fait échouer
`doctor` : rien ne la rattraperait à l'exécution.

### Postuler sur une offre France Travail

Une partie des offres se candidate sur `candidat.francetravail.fr`, qui demande
d'être connecté. L'étape 6 sait s'y connecter comme sur WTTJ :

```bash
python -m agent_emploi apply --login france_travail
```

L'e-mail vient de `FRANCE_TRAVAIL_EMAIL` ou de l'invite, le mot de passe de
`FRANCE_TRAVAIL_PASSWORD` ou d'une saisie masquée — jamais du disque, jamais
d'un journal. La connexion France Travail se fait en deux écrans et demande
souvent un code envoyé par courriel : dans ce cas le programme s'arrête et vous
laisse finir dans la fenêtre ouverte, et la session obtenue est conservée dans
le profil Chromium persistant comme si le programme l'avait faite. Ce n'est pas
un échec, c'est le comportement prévu.

Trois différences de fond avec WTTJ, qui se voient dans le code :

1. **La recherche renvoie déjà la description.** Le découpage en deux temps
   n'a donc rien à économiser ici : `enrich()` ne fait aucune requête quand
   l'offre est complète, ce qui est le cas courant. Le reste du système n'a pas
   à le savoir — l'interface est la même.
2. **Les codes de contrat sont des référentiels**, pas des chaînes libres.
   Plutôt que de figer `E2` ou `FS` dans le code, la source lit
   `/referentiel/typesContrats` et `/referentiel/naturesContrats` à l'exécution
   et apparie les libellés : « alternance » désigne apprentissage *et*
   professionnalisation (`E2` et `FS` aujourd'hui), quels que soient leurs codes
   du moment. Un code figé n'est utilisé qu'en dernier recours, et seulement là
   où il est stable (`CDI`, `CDD`, `MIS`…) — un code inventé ne ferait pas
   échouer la requête, il la viderait, ce qui est bien pire : on lirait
   « aucune offre » là où il fallait lire « panne ».
3. **`typeContrat` et `natureContrat` se combinent en ET.** Demander « CDI ou
   stage » en une requête ne renvoie donc rien. Quand les contrats configurés
   relèvent des deux familles — c'est le cas par défaut — aucun filtre de
   contrat n'est envoyé au serveur : plus d'offres transitent, et c'est le
   pré-filtrage local qui tranche. Le principe vaut aussi pour l'ancienneté :
   `publieeDepuis` n'accepte que 1, 3, 7, 14 ou 31 jours, et l'on choisit
   toujours la valeur **au-dessus** de celle demandée. Jamais de filtre serveur
   plus strict que le filtre local, sinon des offres disparaissent en silence.

Les libellés de contrat sont ramenés au vocabulaire de `config.yaml` (« CDI »,
« Stage », « Alternance ») : laisser passer « Contrat à durée indéterminée »
ferait rejeter par le filtre local une offre qu'il est censé accepter. Le nom
de l'entreprise est souvent masqué sur ces offres — c'est alors le diffuseur
(APEC, Indeed…) qui est affiché, et lui aussi qui devient l'`ats` de l'étape 6.

### Le cas du stage, ou pourquoi on ne fait pas confiance aux champs

Le référentiel des natures de contrat **ne contient aucun « stage »**, et les
offres relayent l'anomalie : « Stage : Ingénieur en informatique » est publiée
avec `typeContrat: CDI`, `natureContrat: Contrat travail`. Deux conséquences,
toutes deux vérifiées par des tests :

- **Aucun filtre serveur n'est possible sur le stage.** Toutes les offres
  reviennent et c'est le pré-filtrage local qui tranche.
- **Le libellé est déduit de l'intitulé**, pas des champs de contrat. Entre un
  champ démenti par les faits et un titre explicite, le titre gagne. L'ordre
  compte : l'alternance est testée d'abord, car elle est correctement typée
  (`natureContrat` apprentissage ou professionnalisation) — « Stage de césure »
  en contrat d'apprentissage est bien une alternance, pas un stage.

Un test `live` échouera le jour où France Travail ajoutera une nature « stage » :
elle serait alors plus fiable que cette heuristique, et il faudrait revenir ici.
Même vigilance sur les libellés qui se ressemblent : « Contrat durée déterminée
insertion » contient « durée déterminée » sans être ce qu'on entend par CDD, et
« CDI intérimaire » n'est pas de l'intérim — ces deux-là sont explicitement
écartés.

`pytest -m live` obtient un vrai jeton et vérifie que les libellés attendus
existent toujours dans les référentiels ; sans identifiants, ces tests sont
ignorés plutôt qu'en échec.

## Structure

```
agent_emploi/
  models.py      Job, JobState, machine à états, identifiants
  config.py      chargement et validation de config.yaml, lecture de .env
  cli.py         doctor / search / screen / draft / review / apply / run /
                 archive / status / web
  manager.py     la boucle bout en bout, jusqu'à votre validation — et pas plus
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
  apply/fields.py   appariement champ <-> information (aucun navigateur)
  apply/mapping.py  recours LLM sur un libellé inconnu (tier gratuit)
  apply/identity.py état civil, lu dans profile/identity.yaml
  apply/browser.py  Playwright, contexte persistant — aucune fonction d'envoi
  apply/login.py    connexion aux sites, mot de passe jamais écrit sur disque
  apply/handoff.py  la fiche candidature.md remise quand on rend la main
  apply/runner.py   passe de candidature : remplissage, états, rapport
  sources/       wttj.py (index Algolia + API v3)
                 france_travail.py (API officielle, OAuth2)
  store/seen.py  mémoire des offres, dédoublonnage
  store/jobs.py  offres retenues : verdict, lettre, revue, dossier
  store/archive.py  classement des candidatures closes + fiche de suivi
  llm/           routeur, budget, fournisseurs (anthropic, groq, gemini)
  web/           interface locale : FastAPI + Jinja2 + htmx (vendu, sans CDN)
config.yaml      tout le réglable : requêtes, filtres, modèles, plafonds
profile/         CV, voix, formules interdites, état civil
outbox/          dossiers en cours, en attente de décision ou d'envoi
applications/    candidatures closes, avec leur fiche de suivi
data/            seen.jsonl, jobs.jsonl, llm_usage.jsonl, browser/ (non versionnés)
```
