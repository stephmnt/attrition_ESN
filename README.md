# README

Voici un notebook. J'ai besoin de le présenter sous la forme d'un powerpoint qui doit durer 15 minutes. 

* Rappel du contexte et présentation brève des dataset de départ (2 minutes)
* Présentation des insights clés de l’analyse exploratoire (5 mins)
* Présentation de l’approche de modélisation
* Méthodologie de construction et d’évaluation du modèle
* Performances du modèle retenu
* Lien entre le comportement du modèle et de ses features (feature importance globale et locale)

## Powerpoint attendu

* Les hypothèses de préparation de la donnée ;
* Les insights clés issus de la phase d’analyse exploratoire ;
* La méthodologie de modélisation finale et les indicateurs clés de fiabilité du modèle ;
* Les insights clés sur les causes d’attrition, issus de l’utilisation de la feature importance globale ;
* Des exemples d’explication de causes d’attrition, en utilisant la feature importance locale.

## Voici la liste des résultats attendus

### Étape 1 : Effectuez une analyse exploratoire des fichiers de données

* Un DataFrame central, issu d’une jointure entre les fichiers de départ.
* Des cellules au sein du notebook pour calculer des statistiques descriptives sur les fichiers de départ et le fichier central, dans l’objectif de faire ressortir des différences clés entre les employés.
* Des cellules au sein du notebook pour générer des graphiques, dans l’objectif de faire ressortir des différences clés entre les employés.

### Étape 2 : Préparez la donnée pour la modélisation

* Un DataFrame contenant vos features prêtes pour la modélisation (autrement dit : un DataFrame qui peut être injecté dans une méthode fit() de sklearn sans erreur). Ce DataFrame est traditionnellement appelé X.
* Un Pandas Series contenant votre colonne cible (traditionnellement appelée y).
* Idéalement, des fonctions Python permettant de réaliser les transformations sur le fichier central de l’étape 1 pour obtenir les deux données X et y. Le cas échéant, des cellules dans le notebook permettant de réaliser ces transformations.

### Étape 3 : Réalisation d'un premier modèle de classification

* Des jeux d’apprentissage et de test (traditionnellement nommés X_train, X_test, y_train, y_test).
* Trois modèles entraînés sur les jeux d’apprentissage : un modèle Dummy, un modèle linéaire, un modèle non-linéaire.
* Des métriques d’évaluation calculées pour chaque modèle, sur le jeu d’apprentissage et le jeu de test.
* Des cellules de code au sein du notebook, permettant d’obtenir les résultats attendus cités plus haut.

### Étape 4 : Améliorez l'approche de classification

* Une modélisation qui tient compte du déséquilibre des classes de votre jeu de données ainsi que du contexte métier. A savoir :
  1. Des jeux d’apprentissage et de test (traditionnellement nommés X_train, X_test, y_train, y_test) issus d’une stratification et avec des features supplémentaires par rapport aux données d’origine.
  2. Un modèle non-linéaire entraîné sur le jeu d’apprentissage et testé sur le jeu de test.
  3. Des métriques et graphiques d’évaluation calculés en validation croisée et interprétés de manière cohérente avec le contexte métier.
* Des cellules de code au sein du notebook, permettant d’obtenir les résultats attendus cités plus haut.

### Étape 5 : Optimisez et interprétez le comportement du modèle

* Un modèle non linéaire issu d’un process de fine-tuning.
* Des graphiques permettant de comprendre l’impact des features sur le modèle, sur le plan global comme local.
* Des cellules de code au sein de Jupyter permettant de :
  * Lancer une recherche d’hyperparamètres pour fine-tuner le modèle,
  * Lancer le calcul des features importances locale et globale.
