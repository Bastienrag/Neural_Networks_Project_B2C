# Guide intelligent du Musée

Projet final du cours *Neural Networks & Deep Learning Foundations*.

Un musée des beaux-arts français veut un guide intelligent, sur le
téléphone des visiteurs et sur des bornes dans les salles. Ce projet en
construit les briques neuronales :

- **reconnaître le style d'une œuvre** à partir d'une photo (Baroque,
  Cubism, Impressionism, Pop_Art, Ukiyo_e), avec un réseau dense puis un
  ResNet pré-entraîné ;
- **répondre aux questions des visiteurs** en langage naturel, en
  retrouvant les œuvres les plus proches de la question ;
- **recommander** à la direction un réseau de neurones ou un modèle
  classique, sachant que les bornes tournent sur des CPU modestes.

Le travail est réparti en six notebooks, un par jalon du cours, à
exécuter dans l'ordre. Ils partagent le chargement des données et le
découpage train / validation / test via le module `code/commun.py`.

Membres de l'équipe : Bastien Ragueneau, Colin Pietri, Cristian Larrain.

## Contenu du dépôt

```
.
├── code/
│   ├── commun.py                    # chargement, découpage, normalisation
│   ├── 01_donnees.ipynb             # contexte, exploration, découpage
│   ├── 02_baseline.ipynb            # régression logistique et réseau dense (m1)
│   ├── 03_diagnostic.ipynb          # suivi de l'entraînement et diagnostic (m2)
│   ├── 04_resnet.ipynb              # ResNet pré-entraîné (m3)
│   ├── 05_questions.ipynb           # recherche question → œuvre (m4)
│   ├── 06_recommandation.ipynb      # évaluation finale et recommandation
│   └── data/
│       ├── make_wikiart_subset.py   # extraction du sous-ensemble WikiArt
│       └── wikiart_style5/          # metadata.csv et subset_info.json (images non versionnées)
├── requirements.txt
└── CONTRIBUTING.md                  # workflow par pull request
```

## Lancer le projet

1. Cloner le dépôt et installer l'environnement (testé avec Python 3.14) :

   ```bash
   git clone https://github.com/Bastienrag/Neural_Networks_Project_B2C.git
   cd Neural_Networks_Project_B2C
   python3 -m venv .venv
   source .venv/bin/activate        # Windows : .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Télécharger les images. Elles ne sont pas versionnées car leur licence
   est inconnue, mais cette commande les régénère à l'identique :

   ```bash
   python code/data/make_wikiart_subset.py --label style \
       --classes Impressionism Baroque Cubism Ukiyo_e Pop_Art \
       --per-class 300 --size 128 --out code/data/wikiart_style5
   ```

   Le script lit environ 4,2 Go depuis Hugging Face pour garder 13 Mo
   d'images. Selon la connexion, cela prend de quelques minutes à plus de
   deux heures. Après une coupure, relancer la même commande reprend là où
   elle s'était arrêtée.

3. Ouvrir les notebooks et les exécuter dans l'ordre, de `01_donnees` à
   `06_recommandation`, chacun de haut en bas :

   ```bash
   jupyter lab code/
   ```

## Avancement

Chaque jalon du cours est marqué par un tag git.

| Jalon | Contenu | Notebook | État |
|---|---|---|---|
| Séance 1 | Dépôt, choix de 5 styles, extraction du sous-ensemble WikiArt | `01_donnees` | fait |
| `m1` | Régression logistique et réseau dense de référence | `02_baseline` | fait |
| `m2` | Courbes de suivi train / validation et diagnostic des échecs | `03_diagnostic` | fait |
| `m3` | ResNet pré-entraîné, comparé au réseau dense | `04_resnet` | fait |
| `m4` | Recherche par question et comparaison neuronal / classique | `05_questions` | fait |
| Rendu | Évaluation finale sur le test et recommandation | `06_recommandation` | à faire |

## Contribuer

Toute modification passe par une pull request : voir
[`CONTRIBUTING.md`](CONTRIBUTING.md).
