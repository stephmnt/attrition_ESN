# attrition_ESN

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-stephanemanet-0A66C2?logo=linkedin&logoColor=white)](https://linkedin.com/in/stephanemanet)

Projet d'analyse RH pour identifier les causes de l'attrition au sein d'une ESN.
La mission consiste à explorer et croiser les donnees SIRH, évaluations annuelles et sondage bien-être, puis a construire un modèle de classification interprétable avec SHAP.

## Contexte

* Mandat RH pour comprendre un turnover élevé.
* Trois sources de donnees internes (SIRH, evaluations, sondage).
* Attendus : EDA, modelisation, interpretation des causes, support de presentation.

## Objectifs

* Comparer les profils "a quitté l'entreprise" vs "actif".
* Construire un modèle qui prédit la probabilité de démission.
* Identifier les facteurs explicatifs via SHAP.
* Formuler des leviers actionnables pour les RH.

## Données

* `datasets/extrait_sirh.csv` : fonction, age, salaire, ancienneté, données sociodémographiques.
* `datasets/extrait_eval.csv` : évaluations annuelles et notes de satisfaction.
* `datasets/extrait_sondage.csv` : sondage bien-être + cible `a_quitte_l_entreprise`.

## Contenu du depot

* `attrition_ESN.ipynb` : nettoyage, EDA, feature engineering, modelisation, SHAP.
* `attrition_ESN/` : paramètres et utilitaires (ex. generation SHAP).
* `output/` : graphiques et exports produits par le notebook.
* `mission.md` : descriptif complet de la mission.
* `pyproject.toml` / `requirements.txt` : dependances du projet.

## Prérequis

* Python >= 3.13 (voir `pyproject.toml`)
* Jupyter ou VS Code pour executer le notebook

## Installation

Option Poetry:
```sh
poetry install
```

Option pip:
```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Utilisation
```sh
jupyter lab
```

Ouvrir `attrition_ESN.ipynb` et executer les cellules dans l'ordre.
Les figures et exports sont ecrits dans `output/`.

## Livrables

* Analyse exploratoire et modelisation.
* Interpretation des facteurs via SHAP.
* Support de presentation a destination de la responsable Data Science.

## Brand
Le dossier `brand/` contient la charte graphique utilisee pour les figures du projet.

* `brand/brand.yml` : palette de couleurs, variantes, fond et dpi.
* `brand/brand.py` : utilitaires Matplotlib/Seaborn pour charger et appliquer le theme.
* Utilisé par le notebook et `attrition_ESN/shap_generator.py` pour uniformiser les graphiques.

Exemple d'utilisation:
```python
from brand.brand import configure_brand
configure_brand("brand/brand.yml")
```