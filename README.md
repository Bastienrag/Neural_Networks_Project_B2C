# Guide intelligent du Musée

Projet final du cours *Neural Networks & Deep Learning Foundations*
(MSA-DATB09-03, Albert School, MSc 1).

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

Tout le travail est dans le notebook
[`code/guide_musee.ipynb`](code/guide_musee.ipynb).

Équipe : Bastien Ragueneau, Colin Pietri, Cristian Larrain.

## Contenu du dépôt

```
.
├── code/
│   ├── guide_musee.ipynb            # notebook de rendu
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

3. Ouvrir le notebook et l'exécuter de haut en bas :

   ```bash
   jupyter lab code/guide_musee.ipynb
   ```

## Avancement

Chaque jalon du cours est marqué par un tag git.

| Jalon | Contenu | État |
|---|---|---|
| Séance 1 | Dépôt, choix de 5 styles, extraction du sous-ensemble WikiArt | fait |
| `m1` | Réseau dense de référence, entraîné avec une boucle propre | à faire |
| `m2` | Courbes de suivi train / validation et diagnostic des échecs | à faire |
| `m3` | ResNet pré-entraîné, comparé au réseau dense | à faire |
| `m4` | Recherche par question et comparaison neuronal / classique | à faire |

## Contribuer

Toute modification passe par une pull request : voir
[`CONTRIBUTING.md`](CONTRIBUTING.md).
