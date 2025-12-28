import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def ouvrir_fichier(chemin):
    """Lire un fichier CSV et retourner un DataFrame."""
    return pd.read_csv(chemin, low_memory=False)


def nettoyer_donnees(tab, colonnes_chiffres=None):
    """Convertit des colonnes numériques et comble les NaN.

    colonnes_chiffres: liste ou None. Si None, aucune conversion n'est faite.
    """
    if colonnes_chiffres:
        for col in colonnes_chiffres:
            if col in tab.columns:
                tab[col] = pd.to_numeric(
                    tab[col].astype(str).str.replace(',', '.'),
                    errors='coerce'
                )
                tab[col] = tab[col].fillna(tab[col].median())

    if 'budget' in tab.columns:
        tab = tab[tab['budget'] > 1000].copy()

    return tab


def transformer_texte_en_chiffre(tab, colonnes_texte):
    """Encode des colonnes catégorielles en entiers avec LabelEncoder."""
    for col in colonnes_texte:
        if col in tab.columns:
            tab[col] = tab[col].astype(str).fillna('Inconnu')
            le = LabelEncoder()
            tab[col] = le.fit_transform(tab[col])
    return tab


def calculer_score_succes(tab):
    """Calcule un indicateur de succès (0-100) basé sur budget, producteurs, réalisateur.

    Nécessite les colonnes :
    'budget', 'producer_number', 'production_companies_number', 'director_number'
    """
    required = [
        'budget',
        'producer_number',
        'production_companies_number',
        'director_number'
    ]

    for c in required:
        if c not in tab.columns:
            raise KeyError(f"Colonne requise manquante : {c}")

    score_budget = 0.5 * np.log1p(tab['budget'])
    force_prod = 0.3 * (
        tab['producer_number'] + tab['production_companies_number']
    )
    poids_tech = 0.2 * tab['director_number']

    tab['score_final'] = score_budget + force_prod + poids_tech

    mini = tab['score_final'].min()
    maxi = tab['score_final'].max()

    if pd.isna(mini) or pd.isna(maxi) or maxi == mini:
        tab['score_final'] = 0
    else:
        tab['score_final'] = 100 * (
            tab['score_final'] - mini
        ) / (maxi - mini)

    return tab


def compute_succes_score(df):
    """Alternative : combine revenue, popularity et vote_average (NON utilisée pour le ML)."""
    if 'revenue' in df.columns:
        df['revenue_log'] = np.log1p(df['revenue'])
    else:
        df['revenue_log'] = 0

    if 'popularity' in df.columns:
        pop_min = df['popularity'].min()
        pop_max = df['popularity'].max()
        if pop_max == pop_min:
            df['popularity_norm'] = 0
        else:
            df['popularity_norm'] = (
                df['popularity'] - pop_min
            ) / (pop_max - pop_min)
    else:
        df['popularity_norm'] = 0

    df['vote_norm'] = (
        df['vote_average'] / 10
        if 'vote_average' in df.columns
        else 0
    )

    df['succes_score'] = (
        0.5 * df['revenue_log']
        + 0.3 * df['popularity_norm']
        + 0.2 * df['vote_norm']
    )

    df = df.dropna(subset=['succes_score'])
    return df
