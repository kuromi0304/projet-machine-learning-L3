# 🎬 Prédiction du Succès des Films (Machine Learning)

Ce projet de Machine Learning a pour objectif de prédire un **score de succès** pour des films, en se basant sur leurs caractéristiques techniques (budget, réalisateur, producteurs, durée) et en utilisant des algorithmes d'apprentissage supervisé.

## 🚀 Installation et Lancement

### 1. Pré-requis
Le projet nécessite **Python** et les librairies suivantes. Vous pouvez les installer via le terminal :

```bash
pip install pandas numpy matplotlib scikit-learn
```
2. Exécuter le projet

Le projet est conçu pour fonctionner avec une seule commande. Placez-vous dans le dossier du projet et lancez :
```bash

python main.py
```
Assurez-vous que le fichier de données est bien situé dans data/DatasetFinal.csv.

📂 Organisation du Code

Le projet est structuré en trois modules pour séparer les responsabilités :
1. main.py (Programme Principal)

C'est le fichier exécutable. Il orchestre tout le processus :

    Chargement des données.

    Appel des fonctions de nettoyage et de transformation.

    Entraînement des deux modèles (Régression Linéaire et Gradient Boosting).

    Affichage des performances (R² et MSE).

    Génération et sauvegarde du graphique des prédictions.

2. utils.py (Boîte à outils)

Contient les fonctions de gestion de données :

    ouvrir_fichier : Chargement du CSV.

    nettoyer_donnees : Gestion des valeurs manquantes et formatage des nombres.

    transformer_texte_en_chiffre : Encodage des colonnes textes (Réalisateurs, Acteurs) en identifiants numériques.

    calculer_score_succes : Création de la variable cible (Target) score_final.

3. process.py (Cerveau IA)

Contient la logique de préparation pour le Machine Learning :

    preparer_ia : Sépare les variables explicatives (X) de la cible (y), supprime les colonnes inutiles ou tricheuses (revenu, vote, popularité) et standardise les données.

    selection_meilleures_colonnes : Utilise un test statistique (f_regression) pour ne garder que les variables les plus pertinentes pour le modèle.

📊 Résultats et Sorties

Une fois le script terminé, vous obtiendrez :

    Dans la console : Un bilan comparatif des performances.

        Exemple : Modèle Linéaire R²: 0.45 | Gradient Boosting R²: 0.52

    Un fichier image : resultat_predictions.png

        Ce graphique compare le score de succès réel (axe X) avec le score prédit par les modèles (axe Y). Plus les points sont proches de la ligne rouge, meilleure est la prédiction.

🧠 Méthodologie

    Algorithmes utilisés : Régression Linéaire (Baseline) et Gradient Boosting Regressor (Modèle avancé).

    Target (Cible) : Le score_final est un indicateur calculé combinant l'impact du budget et la puissance de l'équipe de production.

    Features (Variables) : Le modèle apprend principalement à partir du Budget, du nombre de films du Réalisateur (director_number), des Producteurs (producer_number) et de la durée (runtime).

🔗 Références et Sources

Certaines méthodes avancées et logiques mathématiques utilisées dans ce code (notamment pour le Clustering et la Prédiction de Revenus) s'inspirent des ressources techniques suivantes :

    Approche Clustering & Unsupervised Learning :

        https://github.com/ajitmane36/Netflix-Movies-and-Tv-Shows-Clustering-ML-Unsupervised/tree/main

        Utilisé pour comprendre la segmentation des données.

    Méthodologie de Prédiction de Revenus :

        https://github.com/Vikranth3140/Movie-Revenue-Prediction?utm_source=chatgpt.com

        Référence pour l'analyse des features et les modèles de régression.

Projet réalisé dans le cadre académique (L3).
