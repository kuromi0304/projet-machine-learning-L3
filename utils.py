import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
<<<<<<< HEAD

def ouvrir_fichier(chemin):
    """
    But : Lire le fichier CSV.
    Entrée : chemin vers le fichier.
    Sortie : Le tableau de données (DataFrame).
    """
    return pd.read_csv(chemin, low_memory=False)

def nettoyer_donnees(tab, colonnes_chiffres):
    """
    But : Transformer les textes en nombres et boucher les trous (NaN).
    Entrée : tab (tableau), colonnes_chiffres (liste des noms de colonnes).
    Sortie : Tableau propre.
    """
    for col in colonnes_chiffres:
        if col in tab.columns:
            # On remplace les virgules par des points et on force le type float
            tab[col] = pd.to_numeric(tab[col].astype(str).str.replace(',', '.'), errors='coerce')
            # On remplace les valeurs vides par la médiane pour ne pas fausser les calculs
            tab[col] = tab[col].fillna(tab[col].median())
    
    # On garde seulement les films avec un budget minimum
    tab = tab[tab['budget'] > 1000].copy()
    return tab

def transformer_texte_en_chiffre(tab, colonnes_texte):
    """
    But : Transformer les catégories  en numéros.
    Outil : LabelEncoder (donne un ID unique à chaque texte).
    """
    for col in colonnes_texte:
        if col in tab.columns:
            tab[col] = tab[col].astype(str).fillna('Inconnu')
            codeur = LabelEncoder()
            tab[col] = codeur.fit_transform(tab[col])
    return tab

def calculer_score_succes(tab):
    """
    But : Créer notre indicateur de réussite (Score de 0 à 100).
    Logique : 50% Budget, 30% Partenaires, 20% Réalisateur.
    """
    # On utilise log1p pour que les énormes budgets ne cassent pas l'échelle
    score_budget = 0.5 * np.log1p(tab['budget'])
    
    # On additionne le nombre de producteurs et de boites de prod
    force_prod = 0.3 * (tab['producer_number'] + tab['production_companies_number'])
    
    # On ajoute le poids du réalisateur
    poids_tech = 0.2 * tab['director_number']
    
    tab['score_final'] = score_budget + force_prod + poids_tech
    
    # On ramène tout entre 0 et 100 (Normalisation Min-Max)
    mini = tab['score_final'].min()
    maxi = tab['score_final'].max()
    tab['score_final'] = 100 * (tab['score_final'] - mini) / (maxi - mini)
    
    return tab
=======

def ouvrir_fichier(chemin):
    """
    But : Lire le fichier CSV.
    Entrée : chemin vers le fichier.
    Sortie : Le tableau de données (DataFrame).
    """
    return pd.read_csv(chemin, low_memory=False)

def nettoyer_donnees(tab, colonnes_chiffres):
    """
    But : Transformer les textes en nombres et boucher les trous (NaN).
    Entrée : tab (tableau), colonnes_chiffres (liste des noms de colonnes).
    Sortie : Tableau propre.
    """
    for col in colonnes_chiffres:
        if col in tab.columns:
            # On remplace les virgules par des points et on force le type float
            tab[col] = pd.to_numeric(tab[col].astype(str).str.replace(',', '.'), errors='coerce')
            # On remplace les valeurs vides par la médiane pour ne pas fausser les calculs
            tab[col] = tab[col].fillna(tab[col].median())
    
    # On garde seulement les films avec un budget minimum
    tab = tab[tab['budget'] > 1000].copy()
    return tab

def transformer_texte_en_chiffre(tab, colonnes_texte):
    """
    But : Transformer les catégories  en numéros.
    Outil : LabelEncoder (donne un ID unique à chaque texte).
    """
    for col in colonnes_texte:
        if col in tab.columns:
            tab[col] = tab[col].astype(str).fillna('Inconnu')
            codeur = LabelEncoder()
            tab[col] = codeur.fit_transform(tab[col])
    return tab

def calculer_score_succes(tab):
    """
    But : Créer notre indicateur de réussite (Score de 0 à 100).
    Logique : 50% Budget, 30% Partenaires, 20% Réalisateur.
    """
    # On utilise log1p pour que les énormes budgets ne cassent pas l'échelle
    score_budget = 0.5 * np.log1p(tab['budget'])
    
    # On additionne le nombre de producteurs et de boites de prod
    force_prod = 0.3 * (tab['producer_number'] + tab['production_companies_number'])
    
<<<<<<< HEAD
    for col in categorical_cols:
        df[col] = df[col].astype(str).fillna('Unknown')
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df


#On crée un score unique qui mesure le succès d'un film.

#Entrée :
   # df : le tableau de données déjà nettoyé
#Sortie :
    #df : le même tableau mais avec une nouvelle colonne 'succes_score'
#Fonction principale :
    #combiner le revenu, la popularité et la note moyenne du film en un seul score facile à utiliser


def compute_succes_score(df):
   
    # Logarithme du revenu pour réduire la variance
    df['revenue_log'] = np.log1p(df['revenue'])

    # Normalisation de la popularité
    df['popularity_norm'] = (
        (df['popularity'] - df['popularity'].min()) /
        (df['popularity'].max() - df['popularity'].min())
    )

    # Normalisation du vote moyen
    df['vote_norm'] = df['vote_average'] / 10

    # Score final pondéré
    df['succes_score'] = (
        0.5 * df['revenue_log'] +
        0.3 * df['popularity_norm'] +
        0.2 * df['vote_norm']
    )

    # Supprimer les lignes avec succès non calculable
    df = df.dropna(subset=['succes_score'])
    return df


>>>>>>> 2410ef01 (Ajout du module de visualisation)
=======
    # On ajoute le poids du réalisateur
    poids_tech = 0.2 * tab['director_number']
    
    tab['score_final'] = score_budget + force_prod + poids_tech
    
    # On ramène tout entre 0 et 100 (Normalisation Min-Max)
    mini = tab['score_final'].min()
    maxi = tab['score_final'].max()
    tab['score_final'] = 100 * (tab['score_final'] - mini) / (maxi - mini)
    
    return tab
>>>>>>> 2777b94b (Ajout du module de visualisation L3)
