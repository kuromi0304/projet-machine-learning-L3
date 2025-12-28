import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import des fonctions personnalisées depuis tes modules
from utils import ouvrir_fichier, nettoyer_donnees, calculer_score_succes
from process import creer_clusters_films

CHEMIN_DATA = "C:/Users/keita/Downloads/projet-film/data/DatasetFinal.csv"

# Chargement du fichier
df_brut = ouvrir_fichier(CHEMIN_DATA)

# Nettoyage et préparation des données pour la visualisation
colonnes_num = ['budget', 'director_number', 'producer_number', 'production_companies_number', 'runtime', 'revenue', 'popularity', 'vote_average']
df_propre = nettoyer_donnees(df_brut.copy(), colonnes_num)

# Calcul du score de succès (la variable cible du modèle)
df_propre = calculer_score_succes(df_propre)

# Création des groupes de films (Clusters) pour l'analyse
df_propre, kmeans_model = creer_clusters_films(df_propre, k=4)

#%% 
def tracer_repartition_langues(df):
    """Figure 1 : Analyse de la diversité linguistique des films"""
    langues = df['original_language'].value_counts().head(5)
    couleurs = ['#4682B4', '#5F9EA0', '#8FBC8F', '#BDB76B', '#CD853F']
    
    plt.figure(figsize=(8, 8))
    plt.pie(langues, labels=langues.index, autopct='%1.1f%%', startangle=140, colors=couleurs)
    plt.title("Figure 1 : Répartition par langue d'origine (Top 5)")
    plt.show()

tracer_repartition_langues(df_brut)

#%% 
def tracer_donnees_manquantes(df):
    """Figure 2 : Audit technique des valeurs manquantes dans le dataset"""
    nb_nulles = df.isnull().sum()
    nb_nulles = nb_nulles[nb_nulles > 0] 
    
    plt.figure(figsize=(10, 6))
    plt.bar(nb_nulles.index, nb_nulles.values, color='#A52A2A', edgecolor='black')
    plt.xticks(rotation=45, ha='right')
    plt.title("Figure 2 : Audit des données manquantes")
    plt.ylabel("Nombre de trous détectés")
    plt.tight_layout()
    plt.show()

tracer_donnees_manquantes(df_brut)

#%%
def tracer_repartition_clusters(df):
    """Figure 3 : Visualisation de la segmentation des films (Clusters)"""
    cluster_counts = df['cluster'].value_counts().sort_index()
    couleurs_clusters = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']
    
    plt.figure(figsize=(8, 8))
    plt.pie(
        cluster_counts.values,
        labels=[f"Groupe {c}" for c in cluster_counts.index],
        autopct='%1.1f%%',
        startangle=90,
        colors=couleurs_clusters,
        wedgeprops=dict(width=0.4, edgecolor='white') # Style Donut
    )
    plt.title("Figure 3 : Répartition des films par segments (Clusters)")
    plt.axis('equal')
    plt.show()

tracer_repartition_clusters(df_propre)

#%%
def tracer_comparaison_succes(df, threshold=50):
    """Figure 4 : Distribution des scores de succès calculés par le modèle"""
    petits = df[df['score_final'] <= threshold]['score_final']
    grands = df[df['score_final'] > threshold]['score_final']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.hist(petits, bins=20, color='#FF9999', edgecolor='black')
    ax1.set_title(f"Distribution des 'Petits Succès' (Score <= {threshold})")
    
    ax2.hist(grands, bins=20, color='#99FF99', edgecolor='black')
    ax2.set_title(f"Distribution des 'Grands Succès' (Score > {threshold})")
    
    plt.tight_layout()
    plt.show()

tracer_comparaison_succes(df_propre)

#%% 
def tracer_domination_studios(df):
    """Figure 5 : Analyse des studios produisant les films à plus haut score"""
    top_films = df[df['score_final'] > df['score_final'].quantile(0.8)]
    studios = top_films['production_companies'].value_counts().head(10)
    
    plt.figure(figsize=(12, 6))
    # Inversion pour avoir le plus grand en haut
    plt.barh(studios.index[::-1], studios.values[::-1], color='#4B0082', edgecolor='white')
    plt.title("Figure 5 : Top 10 des studios dominant le segment 'Grand Succès'")
    plt.xlabel("Nombre de films produits")
    plt.tight_layout()
    plt.show()

tracer_domination_studios(df_propre)

#%%
def tracer_correlation_budget_score(df):
    """Figure 6 : Relation entre le budget investi et le score final du modèle"""
    plt.figure(figsize=(10, 6))
    # Utilisation de hexbin pour une meilleure visibilité de la densité
    hb = plt.hexbin(df['budget'], df['score_final'], gridsize=30, cmap='YlGnBu', bins='log')
    plt.colorbar(hb, label='Nombre de films (échelle Log)')
    plt.title("Figure 6 : Corrélation entre Budget et Score de Succès")
    plt.xlabel("Budget ($)")
    plt.ylabel("Score de Succès")
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.show()

tracer_correlation_budget_score(df_propre)
