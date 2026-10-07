
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Configuration de la page
st.set_page_config(
    page_title="Prédiction Risque Crédit",
    page_icon="💳",
    layout="wide"
)

# Titre
st.title("💳 Système de Prédiction du Risque de Crédit")
st.markdown("### Modèle : Random Forest avec interprétabilité SHAP")

# Charger le modèle et l'explainer
@st.cache_resource
def load_model():
    model = joblib.load('random_forest_model.pkl')
    explainer = joblib.load('shap_explainer.pkl')
    return model, explainer

model, explainer = load_model()

st.success("✅ Modèle chargé avec succès !")

# Informations sur le modèle
st.markdown("---")
st.markdown("### 📊 Performances du modèle")
col1, col2, col3, col4 = st.columns(4)
col1.metric("AUC-ROC", "0.802")
col2.metric("F1-score", "0.637")
col3.metric("Recall", "0.717")
col4.metric("Precision", "0.573")

st.markdown("---")
st.markdown("### 🔍 Prédiction d'un client")
st.markdown("*(Formulaire de saisie à venir — nous allons le construire ensemble)*")
