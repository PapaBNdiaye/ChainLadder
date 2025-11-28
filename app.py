import streamlit as st
import pandas as pd
import chainladder as cl
import warnings
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Ajouter le répertoire parent au PYTHONPATH pour permettre l'import de app.utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import garantir_valeurs_cumulees, formater_resultats_export

warnings.filterwarnings("ignore")
plt.ioff()  # Désactiver le mode interactif pour éviter les problèmes avec Streamlit

# Configuration de la page
st.set_page_config(
    page_title="Application Chainladder",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Application Chainladder")
st.markdown("""
Cette application permet d'estimer les réserves de sinistres en utilisant les méthodes Chainladder (déterministe) 
et Mack Chainladder (stochastique).
""")

# Initialisation des variables de session
if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'df_traite' not in st.session_state:
    st.session_state.df_traite = None
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
        help="Chargez votre fichier CSV contenant les données de sinistres"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df_original = df
            st.success(f"✅ Fichier chargé : {len(df)} lignes, {len(df.columns)} colonnes")
            
            # Aperçu des données
            with st.expander("Aperçu des données"):
                st.dataframe(df.head(10))
                st.caption(f"Colonnes disponibles : {', '.join(df.columns.tolist())}")
        except Exception as e:
            st.error(f"Erreur lors du chargement : {str(e)}")
    
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
            index=1 if len(colonnes_disponibles) > 1 else None,
            help="Sélectionnez la colonne contenant l'année ou période de développement"
        )
        
        colonne_montant = st.selectbox(
            "Colonne montant",
            options=colonnes_disponibles,
            index=2 if len(colonnes_disponibles) > 2 else None,
            help="Sélectionnez la colonne contenant les montants (sera vérifiée et convertie en cumul si nécessaire)"
        )
        
        st.session_state.colonne_origin = colonne_origin
        st.session_state.colonne_development = colonne_development
        st.session_state.colonne_montant = colonne_montant
        
        # Bouton pour traiter les données
        if st.button("Traiter les données et construire le triangle", type="primary"):
            with st.spinner("Traitement en cours..."):
                try:
                    # Vérification et calcul du cumul
                    df_traite, modif = garantir_valeurs_cumulees(
                        st.session_state.df_original,
                        colonne_montant,
                        colonne_origin,
                        colonne_development
                    )
                    
                    st.session_state.df_traite = df_traite
                    
                    if modif:
                        st.warning("⚠️ Les données n'étaient pas en cumul. Le cumul a été calculé automatiquement.")
                    else:
                        st.info("ℹ️ Les données étaient déjà en cumul.")
                    
                    # Construction du triangle
                    triangle = cl.Triangle(
                        data=df_traite,
                        origin=colonne_origin,
                        development=colonne_development,
                        columns=[colonne_montant],
                        cumulative=True
                    )
                    
                    st.session_state.triangle = triangle
                    st.success("✅ Triangle construit avec succès !")
                    
                except Exception as e:
                    st.error(f"Erreur lors du traitement : {str(e)}")

# Corps principal de l'application
if st.session_state.triangle is not None:
    st.header("Visualisation du triangle")
    
    # Affichage du triangle avec heatmap
    st.subheader("Triangle de développement")
    try:
        triangle_obj = st.session_state.triangle[st.session_state.colonne_montant]
        # Essayer d'afficher le heatmap
        heatmap_result = triangle_obj.heatmap()
        # Si c'est une figure matplotlib, l'afficher
        if hasattr(heatmap_result, 'savefig') or isinstance(heatmap_result, plt.Figure):
            st.pyplot(heatmap_result)
        else:
            # Sinon, afficher le tableau
            st.dataframe(triangle_obj.to_frame(), width='stretch')
            st.info("Heatmap non disponible, affichage du tableau à la place.")
    except Exception as e:
        # Fallback : afficher le tableau
        try:
            triangle_df = st.session_state.triangle[st.session_state.colonne_montant].to_frame()
            st.dataframe(triangle_df, width='stretch')
            st.warning(f"Affichage du tableau à la place du heatmap. Erreur : {str(e)}")
        except:
            st.error(f"Erreur lors de l'affichage : {str(e)}")
    
    # Estimation des réserves
    st.header("Estimation des réserves")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Estimer avec Chainladder (déterministe)", type="primary"):
            with st.spinner("Calcul en cours..."):
                try:
                    cl_model = cl.Chainladder().fit(
                        st.session_state.triangle[st.session_state.colonne_montant]
                    )
                    st.session_state.cl_model = cl_model
                    st.success("✅ Modèle Chainladder ajusté")
                except Exception as e:
                    st.error(f"Erreur : {str(e)}")
    
    with col2:
        if st.button("Estimer avec Mack Chainladder (stochastique)", type="primary"):
            with st.spinner("Calcul en cours..."):
                try:
                    mack_model = cl.MackChainladder().fit(
                        st.session_state.triangle[st.session_state.colonne_montant]
                    )
                    st.session_state.mack_model = mack_model
                    st.success("✅ Modèle Mack Chainladder ajusté")
                except Exception as e:
                    st.error(f"Erreur : {str(e)}")
    
    # Affichage des résultats Chainladder
    if st.session_state.cl_model is not None:
        with st.expander("📈 Résultats Chainladder (déterministe)", expanded=True):
            st.subheader("Triangle ultime")
            try:
                ultimate_obj = st.session_state.cl_model.ultimate_
                heatmap_ult = ultimate_obj.heatmap()
                if hasattr(heatmap_ult, 'savefig') or isinstance(heatmap_ult, plt.Figure):
                    st.pyplot(heatmap_ult)
                else:
                    st.dataframe(ultimate_obj.to_frame(), width='stretch')
            except Exception as e:
                st.dataframe(st.session_state.cl_model.ultimate_.to_frame(), width='stretch')
                st.warning(f"Heatmap non disponible pour le triangle ultime : {str(e)}")
            
            st.subheader("Réserves IBNR")
            try:
                ibnr_obj = st.session_state.cl_model.ibnr_
                heatmap_ibnr = ibnr_obj.heatmap()
                if hasattr(heatmap_ibnr, 'savefig') or isinstance(heatmap_ibnr, plt.Figure):
                    st.pyplot(heatmap_ibnr)
                else:
                    st.dataframe(ibnr_obj.to_frame(), width='stretch')
            except Exception as e:
                st.dataframe(st.session_state.cl_model.ibnr_.to_frame(), width='stretch')
                st.warning(f"Heatmap non disponible pour les réserves IBNR : {str(e)}")
            
            # Tableau récapitulatif
            st.subheader("Tableau récapitulatif")
            ultimate_df = st.session_state.cl_model.ultimate_.to_frame()
            ibnr_df = st.session_state.cl_model.ibnr_.to_frame()
            
            resume_cl = pd.DataFrame({
                'Année_origine': ultimate_df.index,
                'Ultimate': ultimate_df.iloc[:, 0].values,
                'IBNR': ibnr_df.iloc[:, 0].values
            })
            
            st.dataframe(resume_cl, width='stretch')
            
            # Totaux
            col_tot1, col_tot2 = st.columns(2)
            with col_tot1:
                st.metric("Total Ultimate", f"{resume_cl['Ultimate'].sum():,.2f}")
            with col_tot2:
                st.metric("Total IBNR", f"{resume_cl['IBNR'].sum():,.2f}")
    
    # Affichage des résultats Mack Chainladder
    if st.session_state.mack_model is not None:
        with st.expander("📊 Résultats Mack Chainladder (stochastique)", expanded=True):
            st.subheader("Triangle ultime")
            try:
                ultimate_mack_obj = st.session_state.mack_model.ultimate_
                heatmap_ult_mack = ultimate_mack_obj.heatmap()
                if hasattr(heatmap_ult_mack, 'savefig') or isinstance(heatmap_ult_mack, plt.Figure):
                    st.pyplot(heatmap_ult_mack)
                else:
                    st.dataframe(ultimate_mack_obj.to_frame(), width='stretch')
            except Exception as e:
                st.dataframe(st.session_state.mack_model.ultimate_.to_frame(), width='stretch')
                st.warning(f"Heatmap non disponible pour le triangle ultime : {str(e)}")
            
            st.subheader("Réserves IBNR")
            try:
                ibnr_mack_obj = st.session_state.mack_model.ibnr_
                heatmap_ibnr_mack = ibnr_mack_obj.heatmap()
                if hasattr(heatmap_ibnr_mack, 'savefig') or isinstance(heatmap_ibnr_mack, plt.Figure):
                    st.pyplot(heatmap_ibnr_mack)
                else:
                    st.dataframe(ibnr_mack_obj.to_frame(), width='stretch')
            except Exception as e:
                st.dataframe(st.session_state.mack_model.ibnr_.to_frame(), width='stretch')
                st.warning(f"Heatmap non disponible pour les réserves IBNR : {str(e)}")
            
            # Tableau récapitulatif avec erreurs standards
            st.subheader("Tableau récapitulatif avec erreurs standards")
            
            if hasattr(st.session_state.mack_model, 'summary_'):
                try:
                    # summary_ est un objet Triangle, il faut le convertir en DataFrame
                    summary_mack = st.session_state.mack_model.summary_
                    if hasattr(summary_mack, 'to_frame'):
                        summary_mack_df = summary_mack.to_frame()
                        st.dataframe(summary_mack_df, width='stretch')
                    else:
                        # Si ce n'est pas un Triangle, essayer directement
                        st.dataframe(summary_mack, width='stretch')
                except Exception as e:
                    # Fallback : construction manuelle du résumé
                    ultimate_mack_df = st.session_state.mack_model.ultimate_.to_frame()
                    ibnr_mack_df = st.session_state.mack_model.ibnr_.to_frame()
                    mack_std_err_df = st.session_state.mack_model.mack_std_err_.to_frame()
                    
                    resume_mack = pd.DataFrame({
                        'Année_origine': ultimate_mack_df.index,
                        'Ultimate': ultimate_mack_df.iloc[:, 0].values,
                        'IBNR': ibnr_mack_df.iloc[:, 0].values,
                        'Erreur_standard': mack_std_err_df.iloc[:, 0].values
                    })
                    
                    st.dataframe(resume_mack, width='stretch')
                    st.warning(f"Affichage du résumé manuel. Erreur avec summary_ : {str(e)}")
            else:
                # Construction manuelle du résumé si summary_ n'est pas disponible
                ultimate_mack_df = st.session_state.mack_model.ultimate_.to_frame()
                ibnr_mack_df = st.session_state.mack_model.ibnr_.to_frame()
                mack_std_err_df = st.session_state.mack_model.mack_std_err_.to_frame()
                
                resume_mack = pd.DataFrame({
                    'Année_origine': ultimate_mack_df.index,
                    'Ultimate': ultimate_mack_df.iloc[:, 0].values,
                    'IBNR': ibnr_mack_df.iloc[:, 0].values,
                    'Erreur_standard': mack_std_err_df.iloc[:, 0].values
                })
                
                st.dataframe(resume_mack, width='stretch')
            
            # Totaux avec erreur standard totale
            col_tot1, col_tot2, col_tot3 = st.columns(3)
            ultimate_mack_total = st.session_state.mack_model.ultimate_.sum().sum()
            ibnr_mack_total = st.session_state.mack_model.ibnr_.sum().sum()
            
            with col_tot1:
                st.metric("Total Ultimate", f"{ultimate_mack_total:,.2f}")
            with col_tot2:
                st.metric("Total IBNR", f"{ibnr_mack_total:,.2f}")
            with col_tot3:
                if hasattr(st.session_state.mack_model, 'total_mack_std_err_'):
                    try:
                        # total_mack_std_err_ peut être un DataFrame ou un Triangle, il faut extraire la valeur
                        total_std_err = st.session_state.mack_model.total_mack_std_err_
                        if hasattr(total_std_err, 'sum'):
                            # Si c'est un DataFrame ou Triangle, prendre la somme
                            total_std_err_value = total_std_err.sum().sum() if hasattr(total_std_err.sum(), 'sum') else total_std_err.sum()
                        elif hasattr(total_std_err, 'iloc'):
                            # Si c'est un DataFrame, prendre la première valeur
                            total_std_err_value = total_std_err.iloc[0, 0]
                        else:
                            # Sinon, essayer de convertir directement
                            total_std_err_value = float(total_std_err)
                        st.metric("Erreur standard totale", f"{total_std_err_value:,.2f}")
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
            st.dataframe(resultats_export, width='stretch')
            
            # Conversion en CSV
            csv = resultats_export.to_csv(index=False, encoding='utf-8-sig')
            
            st.download_button(
                label="📥 Télécharger les résultats en CSV",
                data=csv,
                file_name="resultats_chainladder.csv",
                mime="text/csv",
                help="Téléchargez les résultats des deux modèles au format CSV"
            )
        except Exception as e:
            st.error(f"Erreur lors de la préparation de l'export : {str(e)}")
    
    # Comparaison des modèles
    if st.session_state.cl_model is not None and st.session_state.mack_model is not None:
        with st.expander("🔍 Comparaison des modèles"):
            st.subheader("Comparaison des réserves IBNR")
            
            ibnr_cl_total = st.session_state.cl_model.ibnr_.sum().sum()
            ibnr_mack_total = st.session_state.mack_model.ibnr_.sum().sum()
            
            col_comp1, col_comp2 = st.columns(2)
            with col_comp1:
                st.metric("IBNR Chainladder", f"{ibnr_cl_total:,.2f}")
            with col_comp2:
                st.metric("IBNR Mack Chainladder", f"{ibnr_mack_total:,.2f}")
            
            difference = abs(ibnr_cl_total - ibnr_mack_total)
            st.info(f"Différence entre les deux modèles : {difference:,.2f}")

else:
    st.info("👈 Veuillez charger un fichier CSV et configurer les colonnes dans la barre latérale pour commencer.")

