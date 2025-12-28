import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

def preparer_ia(tab):
    """
    But : Isoler les données pour l'apprentissage et les mettre à l'échelle.
    Entrée : tab (DataFrame complet).
    Sortie : X (données d'entrée scalées), y (le score à prédire).
    """
    # La cible (Target) : ce que le modèle doit essayer de deviner
    y = tab['score_final']

    # On retire les colonnes qui ont servi à calculer le score (pour ne pas tricher)
    # et les données qui ne sont connues qu'après la sortie du film (Revenus, Votes)
    colonnes_a_exclure = [
        'score_final', 'budget', 'producer_number', 
        'production_companies_number', 'director_number',
        'revenue', 'popularity', 'vote_average', 'vote_count'
    ]
    
    # On ne garde que les colonnes numériques (les IDs et les nombres)
    # Les colonnes de texte pur (titres, résumés) sont ignorées par le modèle ici
    X = tab.drop(columns=colonnes_a_exclure, errors='ignore').select_dtypes(include=['number'])

    # Mise à l'échelle (Standardisation) : 
    # On transforme les données pour qu'elles aient une moyenne de 0 et un écart-type de 1.
    # C'est indispensable pour que les modèles comme la Régression Linéaire traitent 
    # toutes les variables avec la même importance.
    mise_a_echelle = StandardScaler()
    X_prepare = mise_a_echelle.fit_transform(X.fillna(0))

    return pd.DataFrame(X_prepare, columns=X.columns), y

def selection_meilleures_colonnes(X, y, nb_colonnes=8):
    """
    But : Sélectionner statistiquement les variables qui influencent le plus le score.
    Entrée : X (données), y (cible), nb_colonnes (combien on en veut).
    Sortie : Tableau X réduit aux meilleures variables.
    """
    # SÉCURITÉ : On vérifie combien de colonnes on a réellement sous la main
    nb_disponibles = X.shape[1]
    
    # Si on demande plus de colonnes qu'il n'en existe, on prend le max possible
    k_final = min(nb_colonnes, nb_disponibles)
    
    # On utilise SelectKBest avec un test statistique (f_regression) 
    # pour mesurer la corrélation entre chaque colonne et le score final.
    selecteur = SelectKBest(score_func=f_regression, k=k_final)
    X_reduit = selecteur.fit_transform(X, y)
    
    # On reconstruit le DataFrame avec les noms des colonnes sélectionnées
    colonnes_gardees = X.columns[selecteur.get_support()]
    
    return pd.DataFrame(X_reduit, columns=colonnes_gardees)

def creer_clusters_films(df, k=4):
    """
    But : Créer des groupes de films similaires pour faciliter l'analyse.
    Étapes :
        1. Choisir les attributs pour regrouper les films
        2. Remplacer les NaN par la médiane (SimpleImputer)
        3. Mettre les colonnes sur la même échelle (StandardScaler)
        4. KMeans : créer k groupes de films similaires
    """
    # On sélectionne les critères de ressemblance
    cluster_attributes = ['budget', 'score_final']
    
    # Vérification des colonnes disponibles
    cols_existantes = [c for c in cluster_attributes if c in df.columns]
    X_cluster = df[cols_existantes].copy()

    # SimpleImputer : remplace les NaN par la médiane
    imputer = SimpleImputer(strategy='median')
    X_cluster_imputed = imputer.fit_transform(X_cluster)

    # StandardScaler : met toutes les colonnes sur la même échelle
    scaler = StandardScaler()
    X_cluster_scaled = scaler.fit_transform(X_cluster_imputed)

    # KMeans : créer k groupes de films similaires
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_cluster_scaled)

    return df, kmeans
