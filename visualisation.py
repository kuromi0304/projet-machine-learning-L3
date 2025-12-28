#%% 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import des fonctions personnalisées
from utils import ouvrir_fichier, nettoyer_donnees, calculer_score_succes


CHEMIN_DATA = "C:/Users/keita/Downloads/projet-film/data/DatasetFinal.csv"

# Chargement des DataFrames pour analyse comparative
df_brut = ouvrir_fichier(CHEMIN_DATA)

# Nettoyage et ingénierie de variables (Feature Engineering)
df_propre = nettoyer_donnees(df_brut.copy(), 
                            ['budget', 'director_number', 'producer_number', 
                             'production_companies_number', 'runtime'])

# Calcul de la variable cible (Target)
df_propre = calculer_score_succes(df_propre)

#%% 
def tracer_repartition_langues(df):
    """Analyse descriptive de la diversité linguistique du dataset"""
    langues = df['original_language'].value_counts().head(5)
    
    plt.figure(figsize=(8, 8))
    plt.pie(langues, labels=langues.index, autopct='%1.1f%%', startangle=140, 
            colors=['#4682B4', '#5F9EA0', '#8FBC8F', '#BDB76B', '#CD853F'])
    plt.title("Figure 1 : Répartition par langue d'origine (Top 5)")
    plt.show()

tracer_repartition_langues(df_brut)

#%% 
def tracer_donnees_manquantes(df):
    """Audit technique : Identification des valeurs manquantes (NaN) avant imputation"""
    nb_nulles = df.isnull().sum()
    nb_nulles = nb_nulles[nb_nulles > 0] 
    
    plt.figure(figsize=(10, 6))
    plt.bar(nb_nulles.index, nb_nulles.values, color='#A52A2A', edgecolor='black')
    plt.xticks(rotation=45, ha='right')
    plt.title("Figure 2 : Audit des données manquantes (Vides / NaN)")
    plt.ylabel("Fréquence des valeurs manquantes")
    plt.tight_layout()
    plt.show()

tracer_donnees_manquantes(df_brut)

#%%
def tracer_comparaison_succes(df, threshold=50):
    """Analyse comparative de la distribution des scores selon un seuil de réussite"""
    petits = df[df['score_final'] <= threshold]['score_final']
    grands = df[df['score_final'] > threshold]['score_final']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogramme pour la classe 'Échecs / Succès modérés'
    ax1.hist(petits, bins=20, color='#FF9999', edgecolor='black')
    ax1.set_title(f"Distribution des 'Petits Succès' (Score <= {threshold})")
    ax1.set_xlabel("Indice de Succès")
    ax1.set_ylabel("Effectif")
    
    # Histogramme pour la classe 'Succès majeurs'
    ax2.hist(grands, bins=20, color='#99FF99', edgecolor='black')
    ax2.set_title(f"Distribution des 'Grands Succès' (Score > {threshold})")
    ax2.set_xlabel("Indice de Succès")
    
    plt.tight_layout()
    plt.show()

tracer_comparaison_succes(df_propre)

#%% 
def tracer_domination_studios(df):
    """Analyse de la concentration du marché : Domination des studios (Top 20% du score)"""
    top_films = df[df['score_final'] > df['score_final'].quantile(0.8)]
    studios = top_films['production_companies'].value_counts().head(10)
    
    plt.figure(figsize=(12, 6))
    # Inversion de l'index pour un affichage décroissant en barres horizontales
    plt.barh(studios.index[::-1], studios.values[::-1], color='#4B0082', edgecolor='white')
    plt.title("Figure 4 : Studios dominant le segment 'Grand Succès'")
    plt.xlabel("Volume de production dans le Top quantile")
    plt.tight_layout()
    plt.show()

tracer_domination_studios(df_propre)

#%%
def tracer_genres_clairs(df):
    """Extraction et analyse des catégories cinématographiques prépondérantes"""
    df_temp = df.copy()
    # Imputation des données manquantes pour la catégorisation
    df_temp['genres'] = df_temp['genres'].fillna("Inconnu")
    
    # Extraction du genre atomique (premier élément de la liste)
    df_temp['genre_unique'] = df_temp['genres'].astype(str).apply(
        lambda x: x.split(',')[0].split('|')[0].strip()
    )
    
    top_genres = df_temp['genre_unique'].value_counts().head(10)
    
    plt.figure(figsize=(12, 6))
    plt.bar(top_genres.index, top_genres.values, color='#DAA520', edgecolor='black')
    plt.title("Figure 5 : Analyse par Genre Principal (Top 10)")
    plt.ylabel("Nombre d'occurrences")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

tracer_genres_clairs(df_propre)

#%% 
def tracer_correlation_budget_score(df):
    """Analyse de corrélation par densité : Relation entre capital investi et succès"""
    plt.figure(figsize=(10, 6))
    hb = plt.hexbin(df['budget'], df['score_final'], gridsize=30, cmap='YlGnBu', bins='log')
    plt.colorbar(hb, label='Logarithme du volume de films')
    plt.title("Figure 6 : Densité de corrélation Budget / Score de Succès")
    plt.xlabel("Budget ($)")
    plt.ylabel("Score de Succès (Index 0-100)")
    plt.show()

tracer_correlation_budget_score(df_propre)

#%% 
def tracer_importance_variables(model, noms_colonnes):
    """Interprétabilité du modèle : Analyse de l'importance des variables """
    importances = model.feature_importances_
    indices = np.argsort(importances)
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices], color='#2ECC71', align='center', edgecolor='black')
    plt.yticks(range(len(indices)), [noms_colonnes[i] for i in indices])
    plt.title("Figure 7 : Influence des variables prédictives (Poids du modèle)")
    plt.xlabel("Importance relative (Feature Importance Score)")
    plt.tight_layout()
    plt.show()