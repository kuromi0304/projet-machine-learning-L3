import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

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
    if 'budget' in tab.columns:
        tab = tab[tab['budget'] > 1000].copy()
    return tab

def transformer_texte_en_chiffre(tab, colonnes_texte):
    """
    But : Transformer les catégories en numéros.
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
    Logique : Mélange des moyens mis en œuvre et de la réception du public.
    """
    # On utilise log1p pour que les énormes budgets ne cassent pas l'échelle
    score_budget = 0.5 * np.log1p(tab['budget'])
    # On additionne le nombre de producteurs et de boites de prod
    force_prod = 0.3 * (tab['producer_number'] + tab['production_companies_number'])
    # On ajoute le poids du réalisateur
    poids_tech = 0.2 * tab['director_number']
    
    potentiel = score_budget + force_prod + poids_tech

    # 2. Calcul du succès réel (si les colonnes revenue/popularity existent)
    if 'revenue' in tab.columns and 'popularity' in tab.columns:
        revenue_log = np.log1p(tab['revenue'])
        pop_norm = (tab['popularity'] - tab['popularity'].min()) / (tab['popularity'].max() - tab['popularity'].min())
        vote_norm = tab['vote_average'] / 10 if 'vote_average' in tab.columns else 0.5
        
        succes_reel = (0.5 * revenue_log) + (0.3 * pop_norm) + (0.2 * vote_norm)
        # Moyenne entre le potentiel et le succès réel
        tab['score_final'] = (potentiel + succes_reel) / 2
    else:
        tab['score_final'] = potentiel
    
    # On ramène tout entre 0 et 100 (Normalisation Min-Max)
    mini = tab['score_final'].min()
    maxi = tab['score_final'].max()
    tab['score_final'] = 100 * (tab['score_final'] - mini) / (maxi - mini)
    
    # Supprimer les lignes avec score non calculable
    return tab.dropna(subset=['score_final'])
