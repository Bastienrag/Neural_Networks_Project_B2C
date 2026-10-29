# Contribuer au projet

## Tout passe par une pull request

La branche `main` est protégée : personne ne peut y pousser directement.
Chaque modification arrive par une pull request, que seul @Bastienrag
peut approuver et merger.

1. Partir de `main` à jour et créer une branche :

   ```bash
   git switch main
   git pull
   git switch -c <type>/<sujet>
   ```

   Types de branche : `feat/` (nouveau code), `fix/` (correction),
   `exp/` (expérience d'entraînement), `docs/` (documentation),
   `config/` (configuration du dépôt).

2. Committer, pousser la branche puis ouvrir la PR :

   ```bash
   git push -u origin <type>/<sujet>
   gh pr create
   ```

3. Remplir le modèle de PR, en particulier « Comment je l'ai vérifié » :
   chaque chiffre annoncé doit avoir été mesuré.

4. Attendre la relecture. Un nouveau commit poussé après une approbation
   annule celle-ci. La branche est supprimée automatiquement après le merge.

## Commits

- Un commit = un changement logique. Ne pas mélanger code, documentation
  et configuration dans le même commit.

## Données

Les images WikiArt ne doivent jamais être committées : leur licence est
inconnue. Le `.gitignore` les exclut déjà.
