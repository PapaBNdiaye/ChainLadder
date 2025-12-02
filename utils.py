import warnings
import pandas as pd

warnings.filterwarnings("ignore", message=".*DataFrameGroupBy.apply operated on the grouping columns.*")


def dedoublonner_donnees(df, colonne_origin, colonne_development):
    df = df.copy()
    nb_lignes_avant = len(df)
    
    df_dedoublonne = df.drop_duplicates(
        subset=[colonne_origin, colonne_development],
        keep='last'
    ).reset_index(drop=True)
    
    nb_lignes_apres = len(df_dedoublonne)
    nb_doublons = nb_lignes_avant - nb_lignes_apres
    
    return df_dedoublonne, nb_doublons


def garantir_valeurs_cumulees(
    df,
    colonne_montant,
    colonne_annee_accident,
    colonne_annee_developpement
):
    df = df.copy()
    df = df.sort_values([colonne_annee_accident, colonne_annee_developpement]).reset_index(drop=True)

    modifications_effectuees = False

    def traiter_groupe_par_annee_accident(groupe):
        nonlocal modifications_effectuees
        montants = groupe[colonne_montant].values
        est_cumule = all(montants[i] <= montants[i + 1] for i in range(len(montants) - 1))
        if not est_cumule:
            modifications_effectuees = True
            groupe = groupe.copy()
            groupe[colonne_montant] = montants.cumsum()
        return groupe

    df_corrige = df.groupby(colonne_annee_accident, group_keys=False).apply(traiter_groupe_par_annee_accident)
    
    return df_corrige, modifications_effectuees


def formater_resultats_export(cl_model, mack_model, colonne_origin):
    ultimate_cl = cl_model.ultimate_.to_frame()
    ibnr_cl = cl_model.ibnr_.to_frame()
    
    ultimate_mack = mack_model.ultimate_.to_frame()
    ibnr_mack = mack_model.ibnr_.to_frame()
    mack_std_err_df = mack_model.mack_std_err_.to_frame()
    
    if hasattr(mack_model, 'summary_'):
        try:
            summary_df = mack_model.summary_.to_frame()
            if 'Mack Std Err' in summary_df.columns:
                mack_std_err_values = summary_df['Mack Std Err'].values
            elif len(summary_df.columns) > 0:
                numeric_cols = summary_df.select_dtypes(include=['float64', 'int64']).columns
                if len(numeric_cols) > 0:
                    mack_std_err_values = summary_df.iloc[:, -1].values
                else:
                    mack_std_err_values = mack_std_err_df.sum(axis=1).values
            else:
                mack_std_err_values = mack_std_err_df.sum(axis=1).values
        except (AttributeError, KeyError, ValueError, TypeError) as e:
            mack_std_err_values = mack_std_err_df.sum(axis=1).values
    else:
        mack_std_err_values = mack_std_err_df.sum(axis=1).values
    
    annees_origine = ultimate_cl.index
    if isinstance(annees_origine, pd.DatetimeIndex) or (hasattr(annees_origine, 'dtype') and pd.api.types.is_datetime64_any_dtype(annees_origine)):
        annees_formatees = [annee.year if hasattr(annee, 'year') else pd.to_datetime(annee).year for annee in annees_origine]
    else:
        annees_formatees = []
        for annee in annees_origine:
            if hasattr(annee, 'year'):
                annees_formatees.append(annee.year)
            elif isinstance(annee, str) and ('T' in annee or ' ' in annee):
                try:
                    annees_formatees.append(pd.to_datetime(annee).year)
                except (ValueError, TypeError, pd.errors.ParserError):
                    annees_formatees.append(annee)
            else:
                annees_formatees.append(annee)
    
    resultats = pd.DataFrame({
        'Année_origine': annees_formatees,
        'Ultimate_CL': ultimate_cl.iloc[:, 0].values,
        'IBNR_CL': ibnr_cl.iloc[:, 0].values,
        'Ultimate_Mack': ultimate_mack.iloc[:, 0].values,
        'IBNR_Mack': ibnr_mack.iloc[:, 0].values,
        'Erreur_standard_Mack': mack_std_err_values
    })
    
    return resultats

