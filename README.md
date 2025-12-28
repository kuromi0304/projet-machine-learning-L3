# 🎬 Prédiction du Succès des Films (Machine Learning)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Library](https://img.shields.io/badge/Lib-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Validé-green)

> **Objectif :** Estimer le potentiel commercial et critique d'un film **avant sa sortie** en utilisant l'Intelligence Artificielle, sans utiliser de données futures (Anti-Data Leakage).

---

## 🚀 Démarrage Rapide

### 1. Installation
Assurez-vous d'avoir les librairies nécessaires :

```bash
pip install pandas numpy matplotlib scikit-learn

2. Lancer l'analyse

Le projet est entièrement automatisé. Exécutez simplement :
Bash

python main.py

📂 Résultat : Le script va nettoyer les données, entraîner les modèles et générer automatiquement le graphique resultat_predictions.png dans le dossier courant.
🏗️ Architecture du Projet

Le code est modulaire pour respecter les bonnes pratiques de développement :
Fichier	Rôle Principal
main.py	Exécutable. Pilote le chargement, l'entraînement et la sauvegarde des résultats.
process.py	Intelligence. Prépare les données (X, y) et filtre les variables pour éviter la triche (Data Leakage).
utils.py	Outils. Gère le nettoyage des données, le formatage des nombres et le calcul du score cible.
🧠 Méthodologie IA

Nous comparons deux approches pour prédire le score :

    Régression Linéaire : Modèle de référence (Baseline).

    Gradient Boosting : Modèle avancé (Non-linéaire, souvent plus performant).

🛡️ Stratégie Anti-Triche (Data Leakage)

Pour garantir une prédiction réaliste, nous excluons volontairement les données connues uniquement après la sortie :

    ❌ Revenu Box-Office

    ❌ Popularité

    ❌ Notes des spectateurs

Nous utilisons uniquement les données de production (disponibles avant la sortie) :

    ✅ Budget

    ✅ Casting & Équipe technique (Réalisateur, Producteurs - transformés en IDs)

    ✅ Durée (Runtime) & Saisonnalité

🔗 Références & Crédits

Ce projet s'inspire de méthodes avancées de Feature Engineering et de Clustering issues de la recherche open-source :

    Clustering & Segmentation :

        Netflix Movies & TV Shows Clustering

        Utilisé pour comprendre la segmentation des données.

    Prédiction de Revenus :

        Movie Revenue Prediction

        Référence pour l'analyse des features et les modèles de régression.

Projet réalisé dans le cadre académique (L3).
