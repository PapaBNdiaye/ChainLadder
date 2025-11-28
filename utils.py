import warnings
import pandas as pd

warnings.filterwarnings("ignore", message=".*DataFrameGroupBy.apply operated on the grouping columns.*")


def garantir_valeurs_cumulees(
    df,
    colonne_montant,
    colonne_annee_accident='AccidentYear',
    colonne_annee_developpement='DevelopmentYear'
):
    """
    Vérifie, pour chaque année d'accident, si les montants de `colonne_montant` 
    sont bien cumulés dans le temps (c'est-à-dire non décroissants au fil du développement).
    Si une diminution est détectée, on considère que les données sont incrémentales 
    et on les transforme en cumuls.

    Paramètres :
    - df : DataFrame contenant l'historique de sinistres
    - colonne_montant : nom de la colonne à vérifier (ex. 'Paid' ou 'Incurred')
    - colonne_annee_accident : colonne indiquant l'année de survenance
    - colonne_annee_developpement : colonne indiquant l'année d'observation du développement

    Retourne :
    - Un nouveau DataFrame où `colonne_montant` est garanti cumulé.
    - Un booléen indiquant si des modifications ont été effectuées.
    """
    df = df.copy()
    df = df.sort_values([colonne_annee_accident, colonne_annee_developpement]).reset_index(drop=True)

    modifications_effectuees = False

    def traiter_groupe_par_annee_accident(groupe):
        nonlocal modifications_effectuees
        # IMPORTANT : Trier le groupe par année de développement pour s'assurer de l'ordre correct
        groupe = groupe.sort_values(colonne_annee_developpement).reset_index(drop=True)
        montants_originaux = groupe[colonne_montant].values.copy()
        
        # Cas trivial : une seule valeur ou moins
        if len(montants_originaux) <= 1:
            return groupe
        
        # Vérifie si la série est strictement non décroissante (caractéristique des données cumulées)
        # Pas de tolérance : si une diminution est détectée, les données sont considérées comme incrémentales
        est_non_decroissante = all(
            montants_originaux[i] <= montants_originaux[i + 1] 
            for i in range(len(montants_originaux) - 1)
        )
        
        if est_non_decroissante:
            # Les données sont déjà cumulées (strictement non décroissantes), on ne fait RIEN
            return groupe
        
        # Les données ne sont pas non décroissantes, donc elles sont incrémentales
        # On applique cumsum pour les transformer en cumulées
        modifications_effectuees = True
        groupe = groupe.copy()
        groupe[colonne_montant] = montants_originaux.cumsum()
        return groupe

    # Pas d'include_groups=False pour compatibilité pandas < 2.0
    df_corrige = df.groupby(colonne_annee_accident, group_keys=False).apply(traiter_groupe_par_annee_accident)
    
    return df_corrige, modifications_effectuees


def formater_resultats_export(cl_model, mack_model, colonne_origin):
    """
    Formate les résultats des modèles Chainladder et Mack Chainladder 
    en un DataFrame prêt pour l'export CSV.

    Paramètres :
    - cl_model : modèle Chainladder ajusté
    - mack_model : modèle Mack Chainladder ajusté
    - colonne_origin : nom de la colonne d'origine (pour les index)

    Retourne :
    - DataFrame avec les colonnes : Année d'origine, Ultimate CL, IBNR CL, 
      Ultimate Mack, IBNR Mack, Erreur standard Mack
    """
    # Extraction des données Chainladder
    ultimate_cl = cl_model.ultimate_.to_frame()
    ibnr_cl = cl_model.ibnr_.to_frame()
    
    # Extraction des données Mack Chainladder
    ultimate_mack = mack_model.ultimate_.to_frame()
    ibnr_mack = mack_model.ibnr_.to_frame()
    mack_std_err = mack_model.mack_std_err_.to_frame()
    
    # Création du DataFrame de résultats
    resultats = pd.DataFrame({
        'Année_origine': ultimate_cl.index,
        'Ultimate_CL': ultimate_cl.iloc[:, 0].values,
        'IBNR_CL': ibnr_cl.iloc[:, 0].values,
        'Ultimate_Mack': ultimate_mack.iloc[:, 0].values,
        'IBNR_Mack': ibnr_mack.iloc[:, 0].values,
        'Erreur_standard_Mack': mack_std_err.iloc[:, 0].values
    })
    
    return resultats

