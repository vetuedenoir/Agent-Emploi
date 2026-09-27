# Agent-Emploi

Un assistant de recherche d'emploi qui tourne sur votre ordinateur. Il parcourt
les sites d'offres, garde celles qui correspondent à votre CV, rédige pour
chacune une lettre de motivation dans votre style, et vous prépare un dossier
prêt à envoyer.

**Il n'envoie jamais rien à votre place.** Chaque lettre passe par vous : vous
la relisez, la corrigez si besoin, puis vous postulez vous-même sur le site de
l'offre.

## Comment ça marche

1. **Recherche** — l'assistant interroge les sites d'emploi avec vos mots-clés.
2. **Tri** — il écarte les offres hors sujet, puis fait évaluer les autres par
   un modèle d'IA au regard de votre CV.
3. **Rédaction** — pour les meilleures offres, il écrit une lettre de
   motivation, la relit et choisit le CV à joindre (français ou anglais).
4. **Validation** — vous relisez chaque dossier et décidez : approuver,
   corriger ou rejeter.
5. **Candidature** — vous postulez sur le site de l'offre, puis vous l'indiquez
   à l'assistant, qui range le dossier et vous rappelle quand relancer.

Les étapes de tri utilisent des modèles gratuits ; seule la rédaction des
lettres passe par un modèle payant, pour quelques centimes par lettre. Un
plafond de dépense journalier évite toute mauvaise surprise.

## Installation

Il faut Python 3.11 ou plus récent et [uv](https://docs.astral.sh/uv/).

```bash
uv venv
uv pip install -e ".[web]"
```

Copiez ensuite les fichiers d'exemple :

```bash
cp .env.example .env                # vos clés d'API
cp config.example.yaml config.yaml  # vos critères de recherche
cp -r profile.example profile       # votre CV et votre style d'écriture
```

Puis vérifiez que tout est en place :

```bash
python -m agent_emploi doctor
```

Cette commande indique ce qui manque encore.

## Personnaliser

- **`.env`** : les clés d'accès aux services d'IA et aux sites d'emploi. Les
  commentaires du fichier indiquent à quoi sert chacune ; celles que vous
  laissez vides désactivent simplement le service correspondant.
- **`config.yaml`** : ce que vous cherchez (mots-clés, type de contrat, pays),
  le niveau d'exigence du tri, le nombre de lettres par jour et le budget.
- **`profile/`** : votre CV (en texte et en PDF), des exemples de votre
  écriture et les formules à bannir. Le fichier `voice.md` compte beaucoup :
  sans exemples de votre écriture, les lettres sonneront génériques.

Vos fichiers personnels (`.env`, `config.yaml`, `profile/`) et les données
produites par l'assistant restent sur votre machine et ne sont jamais
versionnés.

## Utilisation

### Interface web

```bash
python -m agent_emploi web
```

Ouvrez ensuite <http://127.0.0.1:8000>. Depuis l'interface, vous pouvez :

- lancer une recherche et suivre son avancement ;
- parcourir les offres trouvées, retenues ou écartées ;
- relire, corriger, approuver ou rejeter chaque lettre ;
- ajouter une offre repérée ailleurs en collant simplement son adresse ;
- suivre vos candidatures envoyées et votre consommation.

L'interface n'est accessible que depuis votre propre ordinateur.

### En ligne de commande

```bash
python -m agent_emploi run        # recherche, tri et rédaction en une fois
python -m agent_emploi review     # relire et valider les dossiers
python -m agent_emploi sent <réf> # indiquer qu'une candidature est envoyée
python -m agent_emploi archive    # ranger les candidatures envoyées
python -m agent_emploi status     # où en sont les offres
python -m agent_emploi --help     # toutes les commandes
```

Ajoutez `--dry-run` pour essayer une commande sans rien enregistrer.

## Ce que vous obtenez

Pour chaque offre retenue, un dossier dans `outbox/` contient :

- l'annonce complète et le lien pour postuler ;
- la lettre de motivation, modifiable à la main ;
- le CV à joindre, dans la langue de l'offre ;
- un aperçu de l'ensemble à ouvrir dans le navigateur.

Une fois la candidature envoyée, le dossier est déplacé dans `applications/`
avec une fiche de suivi (date d'envoi, date de relance, notes).

## Tests

```bash
uv pip install -e ".[dev]"
pytest
```
