import streamlit as st
import pandas as pd
import chainladder as cl
import warnings
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

from utils import garantir_valeurs_cumulees, formater_resultats_export, dedoublonner_donnees

warnings.filterwarnings("ignore")
plt.ioff()


def formater_nombre(nombre, decimales=2):
    if nombre is None:
        return "-"
    if decimales == 0:
        format_str = f"{nombre:,.0f}"
    else:
        format_str = f"{nombre:,.{decimales}f}"
    return format_str.replace(',', ' ')


def formater_annee(valeur):
    if valeur is None:
        return None
    
    if isinstance(valeur, int):
        return valeur
    
    if hasattr(valeur, 'year'):
        return valeur.year
    
    if isinstance(valeur, str):
            try:
                if 'T' in valeur or ' ' in valeur:
                    from datetime import datetime
                    dt = pd.to_datetime(valeur)
                    return dt.year
                elif len(valeur) == 4 and valeur.isdigit():
                    return int(valeur)
            except (ValueError, TypeError, pd.errors.ParserError):
                pass
    
    if isinstance(valeur, pd.Timestamp):
        return valeur.year
    
    try:
        dt = pd.to_datetime(valeur)
        return dt.year
    except (ValueError, TypeError, pd.errors.ParserError):
        return valeur


def formater_dataframe_numerique(df, colonnes_numeriques, decimales=2):
    df_formate = df.copy()
    
    for col in colonnes_numeriques:
        if col in df_formate.columns:
            df_formate[col] = df_formate[col].apply(lambda x: formater_nombre(x, decimales) if pd.notna(x) else "-")
    
    if 'Année_origine' in df_formate.columns:
        df_formate['Année_origine'] = df_formate['Année_origine'].apply(formater_annee)
    
    return df_formate


def afficher_triangle(triangle_obj, titre="Triangle"):
    try:
        df = triangle_obj.to_frame()
        
        if df.empty:
            st.warning("Le triangle est vide.")
            return
        
        if titre:
            st.caption(titre)
        
        colonnes_numeriques = df.select_dtypes(include=[np.number]).columns.tolist()
        
        df_formate = df.copy()
        if isinstance(df.index, pd.DatetimeIndex) or (hasattr(df.index, 'dtype') and pd.api.types.is_datetime64_any_dtype(df.index)):
            df_formate.index = df.index.map(formater_annee)
        elif hasattr(df.index, 'map'):
            try:
                df_formate.index = df.index.map(formater_annee)
            except (ValueError, TypeError, AttributeError):
                pass
        
        if colonnes_numeriques:
            df_formate = formater_dataframe_numerique(df_formate, colonnes_numeriques, 2)
            st.dataframe(df_formate, width='stretch')
        else:
            st.dataframe(df_formate, width='stretch')
        
    except Exception as e:
        st.error(f"Impossible d'afficher le triangle : {str(e)}")


def carte_kpi_model(titre, couleur, kpi_principal, kpis_secondaires=None, sous_titre=None):
    kpis_secondaires = kpis_secondaires or []
    
    style_card = f"border: 3px solid {couleur}; border-radius: 10px; padding: 20px; margin: 10px 0; background: linear-gradient(135deg, {couleur}15 0%, {couleur}05 100%); box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;"
    style_header = f"border-bottom: 2px solid {couleur}; padding-bottom: 10px; margin-bottom: 15px;"
    style_title = f"color: {couleur}; margin: 0; font-size: 18px; font-weight: bold;"
    style_label = "font-size: 12px; color: #666; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 0.5px;"
    style_value = f"font-size: 32px; font-weight: bold; color: {couleur}; margin-bottom: 5px;"
    style_subtitle = "font-size: 11px; color: #888; font-style: italic;"
    style_secondary_label = "font-size: 11px; color: #666; margin-bottom: 3px; text-transform: uppercase; letter-spacing: 0.5px;"
    style_secondary_value = "font-size: 20px; font-weight: 600; color: #333;"
    
    html = f'<div style="{style_card}">'
    html += f'<div style="{style_header}"><h3 style="{style_title}">{titre}</h3></div>'
    html += '<div style="margin-bottom: 15px;">'
    html += f'<div style="{style_label}">{kpi_principal["label"]}</div>'
    html += f'<div style="{style_value}">{kpi_principal["valeur"]}</div>'
    if sous_titre:
        html += f'<div style="{style_subtitle}">{sous_titre}</div>'
    html += '</div>'
    
    if kpis_secondaires:
        html += '<div style="margin-top: 20px; padding-top: 15px; border-top: 1px solid #ddd;">'
        for kpi in kpis_secondaires:
            html += '<div style="margin-bottom: 12px;">'
            html += f'<div style="{style_secondary_label}">{kpi["label"]}</div>'
            html += f'<div style="{style_secondary_value}">{kpi["valeur"]}</div>'
            html += '</div>'
        html += '</div>'
    
    html += '</div>'
    
    st.markdown(html, unsafe_allow_html=True)

# Configuration de la page
st.set_page_config(
    page_title="Application Chainladder",
    page_icon="",
    layout="wide"
)

st.title(":bar_chart: Application Chainladder")
st.markdown("""
Cette application permet d'estimer les réserves de sinistres en utilisant les méthodes Chainladder (déterministe) 
et Mack Chainladder (stochastique).
""")

# Initialisation des variables de session
if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'df_traite' not in st.session_state:
    st.session_state.df_traite = None
if 'df_dedoublonne' not in st.session_state:
    st.session_state.df_dedoublonne = None
if 'df_choisi' not in st.session_state:
    st.session_state.df_choisi = None
if 'choix_donnees' not in st.session_state:
    st.session_state.choix_donnees = None
if 'modif_detectee' not in st.session_state:
    st.session_state.modif_detectee = False
if 'triangle' not in st.session_state:
    st.session_state.triangle = None
if 'cl_model' not in st.session_state:
    st.session_state.cl_model = None
if 'mack_model' not in st.session_state:
    st.session_state.mack_model = None
if 'colonne_origin' not in st.session_state:
    st.session_state.colonne_origin = None
if 'colonne_development' not in st.session_state:
    st.session_state.colonne_development = None
if 'colonne_montant' not in st.session_state:
    st.session_state.colonne_montant = None

# Sidebar pour la configuration
with st.sidebar:
    st.header("Configuration")
    
    # Chargement du fichier
    st.subheader("1. Chargement des données")
    uploaded_file = st.file_uploader(
        "Choisir un fichier CSV",
        type=['csv'],
        help="Chargez votre fichier CSV contenant les données de sinistres",
        key="file_uploader"
    )
    
    if 'uploaded_file_id' not in st.session_state:
        st.session_state.uploaded_file_id = None
    
    if uploaded_file is not None:
        MAX_FILE_SIZE = 50 * 1024 * 1024
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error(f"Fichier trop volumineux. Taille maximale autorisée : {MAX_FILE_SIZE / (1024*1024):.0f} MB")
        else:
            file_id = uploaded_file.file_id if hasattr(uploaded_file, 'file_id') else id(uploaded_file)
            
            if st.session_state.uploaded_file_id != file_id:
                try:
                    df = pd.read_csv(uploaded_file, nrows=1000000)
                    if len(df) >= 1000000:
                        st.warning("Le fichier contient plus de 1 million de lignes. Seules les 1 000 000 premières lignes ont été chargées.")
                    
                    if len(df.columns) < 3:
                        st.error("Le fichier doit contenir au moins 3 colonnes (origin, development, montant)")
                    elif df.empty:
                        st.error("Le fichier CSV est vide")
                    else:
                        st.session_state.df_original = df
                        st.session_state.uploaded_file_id = file_id
                        st.session_state.choix_donnees = None
                        st.session_state.df_choisi = None
                        st.session_state.df_traite = None
                        st.session_state.modif_detectee = False
                        st.session_state.triangle = None
                        st.session_state.cl_model = None
                        st.session_state.mack_model = None
                        st.success(f"Fichier chargé : {formater_nombre(len(df), 0)} lignes, {formater_nombre(len(df.columns), 0)} colonnes")
                except pd.errors.EmptyDataError:
                    st.error("Le fichier CSV est vide ou corrompu")
                except pd.errors.ParserError as e:
                    st.error(f"Erreur de parsing CSV : {str(e)}")
                except UnicodeDecodeError:
                    st.error("Erreur d'encodage : Le fichier doit être en UTF-8")
                except Exception as e:
                    st.error(f"Erreur lors du chargement : {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        
        if st.session_state.df_original is not None:
            with st.expander("Aperçu des données"):
                st.dataframe(st.session_state.df_original.head(10))
                st.caption(f"Colonnes disponibles : {', '.join(st.session_state.df_original.columns.tolist())}")
    
    # Configuration des colonnes
    if st.session_state.df_original is not None:
        st.subheader("2. Configuration des colonnes")
        
        colonnes_disponibles = st.session_state.df_original.columns.tolist()
        
        colonne_origin = st.selectbox(
            "Colonne année de survenance (origin)",
            options=colonnes_disponibles,
            index=0 if colonnes_disponibles else None,
            help="Sélectionnez la colonne contenant l'année de survenance"
        )
        
        colonne_development = st.selectbox(
            "Colonne année de développement (development)",
            options=colonnes_disponibles,
            index=min(1, len(colonnes_disponibles) - 1) if colonnes_disponibles else None,
            help="Sélectionnez la colonne contenant l'année ou période de développement"
        )
        
        colonne_montant = st.selectbox(
            "Colonne montant",
            options=colonnes_disponibles,
            index=min(2, len(colonnes_disponibles) - 1) if colonnes_disponibles else None,
            help="Sélectionnez la colonne contenant les montants (sera vérifiée et convertie en cumul si nécessaire)"
        )
        
        st.session_state.colonne_origin = colonne_origin
        st.session_state.colonne_development = colonne_development
        st.session_state.colonne_montant = colonne_montant
        
        colonnes_changees = (
            st.session_state.colonne_origin != colonne_origin or 
            st.session_state.colonne_development != colonne_development or
            st.session_state.colonne_montant != colonne_montant
        )
        
        if colonnes_changees and st.session_state.triangle is not None:
            st.session_state.choix_donnees = None
            st.session_state.df_choisi = None
            st.session_state.df_traite = None
            st.session_state.modif_detectee = False
            st.session_state.triangle = None
            st.session_state.cl_model = None
            st.session_state.mack_model = None
        
        # Bouton pour traiter les données
        if st.button("Traiter les données", type="primary"):
            with st.spinner("Traitement en cours..."):
                try:
                    if colonne_origin not in st.session_state.df_original.columns:
                        st.error(f"La colonne '{colonne_origin}' n'existe pas dans le DataFrame")
                        st.stop()
                    if colonne_development not in st.session_state.df_original.columns:
                        st.error(f"La colonne '{colonne_development}' n'existe pas dans le DataFrame")
                        st.stop()
                    if colonne_montant not in st.session_state.df_original.columns:
                        st.error(f"La colonne '{colonne_montant}' n'existe pas dans le DataFrame")
                        st.stop()
                    
                    st.session_state.cl_model = None
                    st.session_state.mack_model = None
                    st.session_state.triangle = None
                    st.session_state.choix_donnees = None
                    st.session_state.df_choisi = None
                    
                    df_dedoublonne, nb_doublons = dedoublonner_donnees(
                        st.session_state.df_original,
                        colonne_origin,
                        colonne_development
                    )
                    
                    st.session_state.df_dedoublonne = df_dedoublonne
                    
                    if nb_doublons > 0:
                        st.info(f":information_source: {formater_nombre(nb_doublons, 0)} doublon(s) détecté(s) et supprimé(s) basé(s) sur les colonnes {colonne_origin} et {colonne_development}.")
                    
                    df_traite, modif = garantir_valeurs_cumulees(
                        df_dedoublonne,
                        colonne_montant,
                        colonne_origin,
                        colonne_development
                    )
                    
                    st.session_state.df_traite = df_traite
                    st.session_state.modif_detectee = modif
                    
                    if modif:
                        st.warning(":warning: Les données n'étaient pas en cumul. Le cumul a été calculé automatiquement.")
                        st.info(":bulb: Veuillez choisir les données à utiliser ci-dessous.")
                    else:
                        st.info(":information_source: Les données étaient déjà en cumul.")
                        st.session_state.df_choisi = df_dedoublonne
                        st.session_state.choix_donnees = "original"
                        triangle = cl.Triangle(
                            data=st.session_state.df_choisi,
                            origin=colonne_origin,
                            development=colonne_development,
                            columns=[colonne_montant],
                            cumulative=True
                        )
                        st.session_state.triangle = triangle
                        st.success(":white_check_mark: Triangle construit avec succès !")
                    
                except Exception as e:
                    st.error(f"Erreur lors du traitement : {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        
        if st.session_state.modif_detectee and st.session_state.df_traite is not None and st.session_state.triangle is None:
            st.subheader("3. Choix des données à utiliser")
            st.markdown("""
            **Choisissez les données à utiliser pour construire le triangle :**
            - **Données originales** : Les données telles qu'elles ont été chargées (format incrémental)
            - **Données cumulées** : Les données avec cumul calculé automatiquement
            """)
            
            if st.session_state.choix_donnees is None:
                st.session_state.choix_donnees = "cumule"
            
            choix = st.radio(
                "Sélectionnez le type de données :",
                options=["original", "cumule"],
                format_func=lambda x: "Données originales (incrémentales)" if x == "original" else "Données cumulées (nettoyées)",
                index=0 if st.session_state.choix_donnees == "original" else 1,
                key="choix_donnees_radio"
            )
            
            st.session_state.choix_donnees = choix
            
            col_prev1, col_prev2 = st.columns(2)
            with col_prev1:
                st.caption("**Aperçu données originales**")
                st.dataframe(st.session_state.df_original.head(5), width='stretch')
            with col_prev2:
                st.caption("**Aperçu données cumulées**")
                st.dataframe(st.session_state.df_traite.head(5), width='stretch')
            
            if st.button("Construire le triangle avec les données choisies", type="primary", key="btn_construire_triangle"):
                with st.spinner("Construction du triangle en cours..."):
                    try:
                        st.session_state.cl_model = None
                        st.session_state.mack_model = None
                        
                        choix_actuel = st.session_state.choix_donnees
                        
                        if choix_actuel == "original":
                            st.session_state.df_choisi = st.session_state.df_dedoublonne if st.session_state.df_dedoublonne is not None else st.session_state.df_original
                            use_cumulative = False
                        else:
                            st.session_state.df_choisi = st.session_state.df_traite
                            use_cumulative = True
                        
                        triangle = cl.Triangle(
                            data=st.session_state.df_choisi,
                            origin=st.session_state.colonne_origin,
                            development=st.session_state.colonne_development,
                            columns=[st.session_state.colonne_montant],
                            cumulative=use_cumulative
                        )
                        
                        st.session_state.triangle = triangle
                        
                        type_utilise = "originales (incrémentales)" if choix_actuel == "original" else "cumulées (nettoyées)"
                        st.success(f":white_check_mark: Triangle construit avec succès en utilisant les données {type_utilise} !")
                        st.info(":bulb: Vous pouvez maintenant faire défiler vers le bas pour visualiser le triangle et estimer les réserves.")
                        
                    except Exception as e:
                        st.error(f"Erreur lors de la construction du triangle : {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
        
        elif st.session_state.triangle is not None:
            st.subheader("3. Statut")
            st.success(":white_check_mark: Triangle construit avec succès !")
            if st.session_state.choix_donnees is not None and st.session_state.modif_detectee:
                type_utilise = "originales (incrémentales)" if st.session_state.choix_donnees == "original" else "cumulées (nettoyées)"
                st.info(f"Dataset utilisé : Données {type_utilise}")
            st.info(":bulb: Faites défiler vers le bas pour visualiser le triangle et estimer les réserves.")

# Corps principal de l'application
if st.session_state.triangle is not None:
    if st.session_state.choix_donnees is not None:
        if st.session_state.modif_detectee:
            type_utilise = "originales (incrémentales)" if st.session_state.choix_donnees == "original" else "cumulées (nettoyées)"
            st.info(f":information_source: **Dataset utilisé** : Données {type_utilise}")
        else:
            st.info(":information_source: **Dataset utilisé** : Données originales (déjà en cumul)")
    
    st.header("Visualisation du triangle")
    
    st.subheader("Triangle de développement")
    afficher_triangle(st.session_state.triangle, titre="Triangle de développement")
    
    # Estimation des réserves
    st.header("Estimation des réserves")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Estimer avec Chainladder (déterministe)", type="primary", key="btn_chainladder"):
            with st.spinner("Calcul en cours..."):
                try:
                    cl_model = cl.Chainladder().fit(
                        st.session_state.triangle
                    )
                    st.session_state.cl_model = cl_model
                    st.success(":white_check_mark: Modèle Chainladder ajusté")
                except Exception as e:
                    st.error(f"Erreur : {str(e)}")
    
    with col2:
        if st.button("Estimer avec Mack Chainladder (stochastique)", type="primary", key="btn_mack"):
            with st.spinner("Calcul en cours..."):
                try:
                    mack_model = cl.MackChainladder().fit(
                        st.session_state.triangle
                    )
                    st.session_state.mack_model = mack_model
                    st.success(":white_check_mark: Modèle Mack Chainladder ajusté")
                except Exception as e:
                    st.error(f"Erreur : {str(e)}")
    
    # Affichage des résultats Chainladder
    if st.session_state.cl_model is not None:
        with st.expander(":chart_with_upwards_trend: Résultats Chainladder (déterministe)", expanded=True):
            st.subheader("Triangle ultime")
            ultimate_obj = st.session_state.cl_model.ultimate_
            afficher_triangle(ultimate_obj, titre="Triangle ultime - Chainladder")
            
            st.subheader("Réserves IBNR")
            ibnr_obj = st.session_state.cl_model.ibnr_
            afficher_triangle(ibnr_obj, titre="Réserves IBNR - Chainladder")
            
            st.subheader("Tableau récapitulatif")
            ultimate_df = st.session_state.cl_model.ultimate_.to_frame()
            ibnr_df = st.session_state.cl_model.ibnr_.to_frame()
            
            resume_cl = pd.DataFrame({
                'Année_origine': ultimate_df.index,
                'Ultimate': ultimate_df.iloc[:, 0].values,
                'IBNR': ibnr_df.iloc[:, 0].values
            })
            
            resume_cl_formate = formater_dataframe_numerique(resume_cl, ['Ultimate', 'IBNR'], 2)
            st.dataframe(resume_cl_formate, width='stretch')
            
            col_tot1, col_tot2 = st.columns(2)
            with col_tot1:
                st.metric("Total Ultimate", formater_nombre(resume_cl['Ultimate'].sum(), 2))
            with col_tot2:
                st.metric("Total IBNR", formater_nombre(resume_cl['IBNR'].sum(), 2))
    
    # Affichage des résultats Mack Chainladder
    if st.session_state.mack_model is not None:
        with st.expander(":bar_chart: Résultats Mack Chainladder (stochastique)", expanded=True):
            st.subheader("Triangle ultime")
            ultimate_mack_obj = st.session_state.mack_model.ultimate_
            afficher_triangle(ultimate_mack_obj, titre="Triangle ultime - Mack Chainladder")
            
            st.subheader("Réserves IBNR")
            ibnr_mack_obj = st.session_state.mack_model.ibnr_
            afficher_triangle(ibnr_mack_obj, titre="Réserves IBNR - Mack Chainladder")
            
            st.subheader("Tableau récapitulatif avec erreurs standards")
            
            if hasattr(st.session_state.mack_model, 'summary_'):
                try:
                    summary_mack = st.session_state.mack_model.summary_
                    if hasattr(summary_mack, 'to_frame'):
                        summary_mack_df = summary_mack.to_frame()
                        colonnes_numeriques_summary = summary_mack_df.select_dtypes(include=[np.number]).columns.tolist()
                        if colonnes_numeriques_summary:
                            summary_mack_df_formate = formater_dataframe_numerique(summary_mack_df, colonnes_numeriques_summary, 2)
                            st.dataframe(summary_mack_df_formate, width='stretch')
                        else:
                            st.dataframe(summary_mack_df, width='stretch')
                    else:
                        st.dataframe(summary_mack, width='stretch')
                except Exception as e:
                    ultimate_mack_df = st.session_state.mack_model.ultimate_.to_frame()
                    ibnr_mack_df = st.session_state.mack_model.ibnr_.to_frame()
                    mack_std_err_df = st.session_state.mack_model.mack_std_err_.to_frame()
                    
                    resume_mack = pd.DataFrame({
                        'Année_origine': ultimate_mack_df.index,
                        'Ultimate': ultimate_mack_df.iloc[:, 0].values,
                        'IBNR': ibnr_mack_df.iloc[:, 0].values,
                        'Erreur_standard': mack_std_err_df.iloc[:, 0].values
                    })
                    
                    resume_mack_formate = formater_dataframe_numerique(resume_mack, ['Ultimate', 'IBNR', 'Erreur_standard'], 2)
                    st.dataframe(resume_mack_formate, width='stretch')
                    st.warning(f"Affichage du résumé manuel. Erreur avec summary_ : {str(e)}")
            else:
                ultimate_mack_df = st.session_state.mack_model.ultimate_.to_frame()
                ibnr_mack_df = st.session_state.mack_model.ibnr_.to_frame()
                mack_std_err_df = st.session_state.mack_model.mack_std_err_.to_frame()
                
                resume_mack = pd.DataFrame({
                    'Année_origine': ultimate_mack_df.index,
                    'Ultimate': ultimate_mack_df.iloc[:, 0].values,
                    'IBNR': ibnr_mack_df.iloc[:, 0].values,
                    'Erreur_standard': mack_std_err_df.iloc[:, 0].values
                })
                
                resume_mack_formate = formater_dataframe_numerique(resume_mack, ['Ultimate', 'IBNR', 'Erreur_standard'], 2)
                st.dataframe(resume_mack_formate, width='stretch')
            
            col_tot1, col_tot2, col_tot3 = st.columns(3)
            ultimate_mack_total = st.session_state.mack_model.ultimate_.sum().sum()
            ibnr_mack_total = st.session_state.mack_model.ibnr_.sum().sum()
            
            with col_tot1:
                st.metric("Total Ultimate", formater_nombre(ultimate_mack_total, 2))
            with col_tot2:
                st.metric("Total IBNR", formater_nombre(ibnr_mack_total, 2))
            with col_tot3:
                if hasattr(st.session_state.mack_model, 'total_mack_std_err_'):
                    try:
                        total_std_err = st.session_state.mack_model.total_mack_std_err_
                        if hasattr(total_std_err, 'sum'):
                            total_std_err_value = total_std_err.sum().sum() if hasattr(total_std_err.sum(), 'sum') else total_std_err.sum()
                        elif hasattr(total_std_err, 'iloc'):
                            total_std_err_value = total_std_err.iloc[0, 0]
                        else:
                            total_std_err_value = float(total_std_err)
                        st.metric("Erreur standard totale", formater_nombre(total_std_err_value, 2))
                    except Exception as e:
                        st.metric("Erreur standard totale", "N/A")
                        st.caption(f"Impossible d'afficher l'erreur standard : {str(e)}")
    
    # Export des résultats
    if st.session_state.cl_model is not None and st.session_state.mack_model is not None:
        st.header("Export des résultats")
        
        try:
            resultats_export = formater_resultats_export(
                st.session_state.cl_model,
                st.session_state.mack_model,
                st.session_state.colonne_origin
            )
            
            st.subheader("Résultats combinés")
            colonnes_numeriques_export = ['Ultimate_CL', 'IBNR_CL', 'Ultimate_Mack', 'IBNR_Mack', 'Erreur_standard_Mack']
            resultats_export_formate = formater_dataframe_numerique(resultats_export, colonnes_numeriques_export, 2)
            st.dataframe(resultats_export_formate, width='stretch')
            
            csv = resultats_export.to_csv(index=False, encoding='utf-8-sig')
            
            st.download_button(
                label=":inbox_tray: Télécharger les résultats en CSV",
                data=csv,
                file_name="resultats_chainladder.csv",
                mime="text/csv",
                help="Téléchargez les résultats des deux modèles au format CSV"
            )
        except Exception as e:
            st.error(f"Erreur lors de la préparation de l'export : {str(e)}")
    
    # Comparaison des modèles
    if st.session_state.cl_model is not None and st.session_state.mack_model is not None:
        st.header("Comparaison des modèles")
        
        ibnr_cl_total = st.session_state.cl_model.ibnr_.sum().sum()
        ibnr_mack_total = st.session_state.mack_model.ibnr_.sum().sum()
        difference = abs(ibnr_cl_total - ibnr_mack_total)
        
        total_std_err_value = None
        if hasattr(st.session_state.mack_model, 'total_mack_std_err_'):
            try:
                total_std_err = st.session_state.mack_model.total_mack_std_err_
                if hasattr(total_std_err, 'sum'):
                    total_std_err_value = total_std_err.sum().sum() if hasattr(total_std_err.sum(), 'sum') else total_std_err.sum()
                elif hasattr(total_std_err, 'iloc'):
                    total_std_err_value = total_std_err.iloc[0, 0]
                else:
                    total_std_err_value = float(total_std_err)
            except (AttributeError, ValueError, TypeError, KeyError, IndexError):
                total_std_err_value = None
        
        if difference < 0.01:
            st.success("Les deux modèles donnent les mêmes réserves IBNR - c'est normal.")
        
        col_header1, col_header2 = st.columns(2)
        
        with col_header1:
            carte_kpi_model(
                titre="Chainladder (déterministe)",
                couleur="#1f77b4",
                kpi_principal={
                    'label': 'Réserves IBNR',
                    'valeur': formater_nombre(ibnr_cl_total, 2)
                },
                sous_titre="Estimation ponctuelle uniquement"
            )
        
        with col_header2:
            if total_std_err_value is not None:
                cv = (total_std_err_value / ibnr_mack_total) * 100 if ibnr_mack_total > 0 else 0
                carte_kpi_model(
                    titre="Mack Chainladder (stochastique)",
                    couleur="#2ca02c",
                    kpi_principal={
                        'label': 'Réserves IBNR',
                        'valeur': formater_nombre(ibnr_mack_total, 2)
                    },
                    kpis_secondaires=[
                        {
                            'label': 'Erreur standard',
                            'valeur': formater_nombre(total_std_err_value, 2)
                        },
                        {
                            'label': 'Coefficient de variation',
                            'valeur': f"{cv:.2f}%"
                        }
                    ]
                )
            else:
                carte_kpi_model(
                    titre="Mack Chainladder (stochastique)",
                    couleur="#2ca02c",
                    kpi_principal={
                        'label': 'Réserves IBNR',
                        'valeur': formater_nombre(ibnr_mack_total, 2)
                    },
                    sous_titre="Erreur standard non disponible"
                )
        
        st.divider()
        
        tab1, tab2, tab3 = st.tabs(["Comparaison détaillée", "Visualisation", "Interprétation"])
        
        with tab1:
            st.subheader("Tableau comparatif")
            
            if total_std_err_value is not None:
                ic_95_inf = ibnr_mack_total - 1.96 * total_std_err_value
                ic_95_sup = ibnr_mack_total + 1.96 * total_std_err_value
                ic_99_inf = ibnr_mack_total - 2.58 * total_std_err_value
                ic_99_sup = ibnr_mack_total + 2.58 * total_std_err_value
                cv = (total_std_err_value / ibnr_mack_total) * 100 if ibnr_mack_total > 0 else 0
                
                comparaison_df = pd.DataFrame({
                    'Modèle': ['Chainladder', 'Mack Chainladder'],
                    'Réserves IBNR': [ibnr_cl_total, ibnr_mack_total],
                    'Erreur standard': ['-', formater_nombre(total_std_err_value, 2)],
                    'Coefficient de variation': ['-', f"{cv:.2f}%"],
                    'IC 95% (inférieur)': ['-', formater_nombre(ic_95_inf, 2)],
                    'IC 95% (supérieur)': ['-', formater_nombre(ic_95_sup, 2)],
                    'IC 99% (inférieur)': ['-', formater_nombre(ic_99_inf, 2)],
                    'IC 99% (supérieur)': ['-', formater_nombre(ic_99_sup, 2)]
                })
                
                comparaison_df_formate = formater_dataframe_numerique(comparaison_df, ['Réserves IBNR'], 2)
                st.dataframe(comparaison_df_formate, width='stretch', hide_index=True)
            else:
                st.warning("L'erreur standard n'est pas disponible. Le tableau comparatif complet ne peut pas être affiché.")
                comparaison_simple_df = pd.DataFrame({
                    'Modèle': ['Chainladder', 'Mack Chainladder'],
                    'Réserves IBNR': [ibnr_cl_total, ibnr_mack_total]
                })
                comparaison_simple_df_formate = formater_dataframe_numerique(comparaison_simple_df, ['Réserves IBNR'], 2)
                st.dataframe(comparaison_simple_df_formate, width='stretch', hide_index=True)
        
        with tab2:
            st.subheader("Visualisation de l'incertitude")
            
            if total_std_err_value is not None:
                ic_95_inf = ibnr_mack_total - 1.96 * total_std_err_value
                ic_95_sup = ibnr_mack_total + 1.96 * total_std_err_value
                ic_99_inf = ibnr_mack_total - 2.58 * total_std_err_value
                ic_99_sup = ibnr_mack_total + 2.58 * total_std_err_value
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=['Chainladder'],
                    y=[ibnr_cl_total],
                    name='Chainladder',
                    marker_color='#1f77b4',
                    text=[formater_nombre(ibnr_cl_total, 0)],
                    textposition='outside',
                    textfont=dict(size=14, color='black', family='Arial Black'),
                    width=0.5,
                    hovertemplate='<b>Chainladder</b><br>Réserves IBNR: ' + formater_nombre(ibnr_cl_total, 0) + '<extra></extra>'
                ))
                
                fig.add_trace(go.Bar(
                    x=['Mack Chainladder'],
                    y=[ibnr_mack_total],
                    name='Mack Chainladder',
                    marker_color='#2ca02c',
                    text=[formater_nombre(ibnr_mack_total, 0)],
                    textposition='outside',
                    textfont=dict(size=14, color='black', family='Arial Black'),
                    width=0.5,
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array=[ic_95_sup - ibnr_mack_total],
                        arrayminus=[ibnr_mack_total - ic_95_inf],
                        color='red',
                        thickness=4,
                        width=8
                    ),
                    hovertemplate='<b>Mack Chainladder</b><br>Réserves IBNR: %{y}<br>IC 95%: [' + f'{formater_nombre(ic_95_inf, 0)} - {formater_nombre(ic_95_sup, 0)}]<extra></extra>'
                ))
                
                fig.update_layout(
                    xaxis=dict(
                        title=dict(text='Modèle', font=dict(size=16, color='black', family='Arial')),
                        tickfont=dict(size=13, color='black'),
                        gridcolor='lightgray',
                        linecolor='black',
                        linewidth=2,
                        title_standoff=20
                    ),
                    yaxis=dict(
                        title=dict(text='Réserves IBNR', font=dict(size=16, color='black', family='Arial')),
                        tickformat='.0f',
                        tickfont=dict(size=12, color='black'),
                        gridcolor='lightgray',
                        linecolor='black',
                        linewidth=2,
                        title_standoff=30
                    ),
                    height=550,
                    margin=dict(l=120, r=80, t=80, b=120),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    showlegend=True,
                    legend=dict(
                        yanchor="top",
                        y=0.98,
                        xanchor="left",
                        x=1.02,
                        bgcolor='white',
                        bordercolor='black',
                        borderwidth=2,
                        font=dict(size=13, color='black', family='Arial')
                    )
                )
                
                fig.add_annotation(
                    x=1, xref='x',
                    y=ic_95_inf - (ic_95_sup - ic_95_inf) * 0.15, yref='y',
                    text=f'IC 95%: [{formater_nombre(ic_95_inf, 0)} - {formater_nombre(ic_95_sup, 0)}]',
                    showarrow=False,
                    font=dict(size=11, color='red', family='Arial'),
                    bgcolor='yellow',
                    bordercolor='red',
                    borderwidth=2,
                    xshift=0,
                    yshift=-10
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.caption(f"**Intervalle de confiance à 95% pour Mack Chainladder** : [{formater_nombre(ic_95_inf, 0)} ; {formater_nombre(ic_95_sup, 0)}]")
            else:
                st.info("L'erreur standard n'est pas disponible pour créer la visualisation.")
        
        with tab3:
            st.subheader("Interprétation des résultats")
            
            if difference < 0.01:
                st.markdown("""
                **Les deux modèles donnent les mêmes réserves IBNR - c'est normal.**
                
                **Chainladder (déterministe)** :
                - Calcule les réserves IBNR à partir des facteurs de développement moyens
                - Simple et rapide
                - Ne fournit pas d'information sur l'incertitude
                
                **Mack Chainladder (stochastique)** :
                - Donne les mêmes réserves IBNR que Chainladder
                - Ajoute l'erreur standard : mesure de l'incertitude sur les réserves
                - Permet de construire des intervalles de confiance
                - Coefficient de variation : indique le niveau d'incertitude relatif
                
                **En pratique, Mack vous permet de :**
                - Quantifier l'incertitude sur vos réserves
                - Construire des intervalles de confiance pour la planification financière
                - Prendre des décisions éclairées en connaissant les risques
                - Répondre aux exigences réglementaires (Solvabilité II, etc.)
                """)
                
                if total_std_err_value is not None:
                    ic_95_inf = ibnr_mack_total - 1.96 * total_std_err_value
                    ic_95_sup = ibnr_mack_total + 1.96 * total_std_err_value
                    ic_99_inf = ibnr_mack_total - 2.58 * total_std_err_value
                    ic_99_sup = ibnr_mack_total + 2.58 * total_std_err_value
                    
                    st.markdown("---")
                    st.markdown("#### Interprétation des intervalles de confiance")
                    st.markdown(f"""
                    **Intervalle de confiance à 95%** : [{formater_nombre(ic_95_inf, 0)} ; {formater_nombre(ic_95_sup, 0)}]
                    
                    **Intervalle de confiance à 99%** : [{formater_nombre(ic_99_inf, 0)} ; {formater_nombre(ic_99_sup, 0)}]
                    
                    **Interprétation correcte (fréquentiste) :**
                    
                    Un intervalle de confiance à 95% signifie que **si on répétait le calcul des réserves un grand nombre de fois** avec des échantillons similaires, environ **95% des intervalles de confiance calculés** contiendraient la vraie valeur des réserves.
                    
                    **Points importants :**
                    - Les vraies réserves sont une valeur fixe mais inconnue
                    - L'intervalle est calculé à partir de vos données spécifiques
                    - Soit l'intervalle contient les vraies réserves, soit il ne les contient pas (on ne peut pas le savoir avec certitude)
                    - Le niveau de confiance (95%) se réfère à la **méthode d'estimation** : sur de nombreux échantillons similaires, 95% des intervalles contiendraient les vraies réserves
                    
                    **En pratique :**
                    L'intervalle de confiance vous donne une indication de la précision de votre estimation. Un intervalle plus étroit indique une estimation plus précise, tandis qu'un intervalle plus large indique plus d'incertitude.
                    """)
            else:
                st.warning(f"Différence entre les deux modèles : {formater_nombre(difference, 2)}")
                st.info("""
                **Note** : Normalement, les deux modèles devraient donner les mêmes réserves IBNR car ils utilisent les mêmes facteurs de développement.
                Une différence peut indiquer un problème dans les données ou dans le calcul.
                """)

else:
    st.info("Veuillez charger un fichier CSV et configurer les colonnes dans la barre latérale pour commencer.")

