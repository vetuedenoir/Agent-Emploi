# Profil — exemple

Copiez ce dossier en `profile/` puis remplacez chaque fichier par le vôtre :

```bash
cp -r profile.example profile
```

| Fichier | Rôle |
|---|---|
| `cv.md` | votre CV en texte : c'est lui que lit le programme |
| `cv_fr.pdf`, `cv_en.pdf` | votre CV en PDF, joint à la candidature selon la langue de l'offre (à ajouter vous-même) |
| `voice.md` | des exemples de votre écriture, pour que les lettres vous ressemblent |
| `banned_phrases.txt` | les formules que vous ne voulez jamais voir dans une lettre |

Pour viser plusieurs familles de postes, vous pouvez écrire plusieurs versions
de `cv.md` (par exemple une orientée développement, une orientée machine
learning) et les déclarer sous `profile.variants` dans `config.yaml`. Chaque
offre est alors rangée sous la version qui lui convient le mieux.

Le dossier `profile/` n'est jamais versionné.
