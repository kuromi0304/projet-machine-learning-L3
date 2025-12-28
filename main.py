import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Import de nos fonctions personnalisées
from utils import ouvrir_fichier, nettoyer_donnees, transformer_texte_en_chiffre, calculer_score_succes
from process import preparer_ia, selection_meilleures_colonnes, creer_clusters_films

# Chemin vers le fichier CSV
CHEMIN_DATA = "data\DatasetFinal.csv"

# Chargement du fichier
donnees = ouvrir_fichier(CHEMIN_DATA)

# Nettoyage des colonnes numériques (gestion des virgules et des vides)
colonnes_num = ['budget', 'director_number', 'producer_number', 'production_companies_number', 'runtime']
donnees = nettoyer_donnees(donnees, colonnes_num)

# Transformation des colonnes textes en identifiants (IDs) pour le modèle
colonnes_txt = ['cast', 'director', 'production_companies']
donnees = transformer_texte_en_chiffre(donnees, colonnes_txt)

# Création de notre indicateur de réussite (la "Target" y)
donnees = calculer_score_succes(donnees)

# Ajout des groupes de films (Clustering) pour l'analyse de segmentation
donnees, kmeans = creer_clusters_films(donnees, k=4)

# On prépare les données (X) et la cible (y)
X, y = preparer_ia(donnees)

# On demande à l'algorithme de garder les variables les plus importantes
X = selection_meilleures_colonnes(X, y, nb_colonnes=8)

# Séparation : 80% pour entraîner le modèle , 20% pour tester sa précision
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle 1 : Régression Linéaire (Modèle de base)
modele_simple = LinearRegression()
modele_simple.fit(X_train, y_train)
pred_simple = modele_simple.predict(X_test)

# Modèle 2 : Gradient Boosting (Modèle plus complexe avec optimisation)
gbr = GradientBoostingRegressor(random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1]
}
grid = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=3, n_jobs=-1)
grid.fit(X_train, y_train)
modele_complexe = grid.best_estimator_
pred_complexe = modele_complexe.predict(X_test)

# Calcul des performances
r2_simple = r2_score(y_test, pred_simple)
r2_complexe = r2_score(y_test, pred_complexe)
mse_simple = mean_squared_error(y_test, pred_simple)
mse_complexe = mean_squared_error(y_test, pred_complexe)

# Affichage des résultats finaux
print("\n" + "="*45)
print(" BILAN DES PERFORMANCES ")
print("="*45)
print(f"Modèle Linéaire      R²: {r2_simple:.3f} | Erreur Quadratique : {mse_simple:.2f}")
print(f"Gradient Boosting    R²: {r2_complexe:.3f} | Erreur Quadratique : {mse_complexe:.2f}")
print("="*45)

# GRAPHIQUE DES PRÉDICTIONS
plt.figure(figsize=(10, 6))
plt.scatter(y_test, pred_simple, color='orange', alpha=0.5, label=f'Linéaire (R²={r2_simple:.2f})')
plt.scatter(y_test, pred_complexe, color='purple', alpha=0.5, label=f'Gradient Boosting (R²={r2_complexe:.2f})')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', label='Prédiction Parfaite')

plt.title("Capacité du modèle à prédire le succès (données réelles vs prédites)")
plt.xlabel("Score de Succès Réel")
plt.ylabel("Score de Succès Prédit")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.savefig("resultat_predictions.png")
print("Graphique enregistré : resultat_predictions.png")

# VISUALISATION DES CLUSTERS (DONUT)
plt.figure(figsize=(7,7))
cluster_counts = donnees['cluster'].value_counts().sort_index()
plt.pie(
    cluster_counts.values,
    labels=[f"Cluster {c}" for c in cluster_counts.index],
    autopct='%1.1f%%',
    startangle=90,
    wedgeprops=dict(width=0.4)
)
plt.title("Répartition des films par cluster")
plt.axis('equal')
plt.tight_layout()

plt.savefig("repartition_clusters.png")
print("Graphique enregistré : repartition_clusters.png")

plt.show()
