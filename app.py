
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from datetime import datetime

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Credit Risk Assessment | KIIT University",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .risk-high {
        background-color: #fee;
        border-left: 5px solid #e74c3c;
        padding: 1rem;
        border-radius: 6px;
    }
    .risk-low {
        background-color: #efe;
        border-left: 5px solid #27ae60;
        padding: 1rem;
        border-radius: 6px;
    }
    .footer {
        text-align: center;
        color: #999;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD ARTIFACTS
# ============================================
@st.cache_resource
def load_artifacts():
    model = joblib.load('random_forest_model.pkl')
    explainer = joblib.load('shap_explainer.pkl')
    columns_list = joblib.load('columns_list.pkl')
    model_results = joblib.load('model_results.pkl')
    shap_values_global = joblib.load('shap_values_global.pkl')
    X_test = joblib.load('X_test.pkl')
    return model, explainer, columns_list, model_results, shap_values_global, X_test

model, explainer, columns_list, model_results, shap_values_global, X_test = load_artifacts()

# ============================================
# HEADER
# ============================================
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.markdown("# 🏦")
with col_title:
    st.markdown('<div class="main-header">Credit Risk Assessment System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">M.Sc. Computer Science — KIIT University | Ensemble Models & SHAP Interpretability</div>', unsafe_allow_html=True)

# ============================================
# TABS
# ============================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Client Prediction",
    "📊 Global Analysis",
    "📈 Model Comparison",
    "📖 About & Methodology"
])

# ============================================
# TAB 1: CLIENT PREDICTION
# ============================================
with tab1:
    st.markdown("### Client Risk Assessment")
    st.markdown("Enter the client's information below to obtain a real-time risk prediction with SHAP explanation.")

    with st.form("client_form"):
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            duration = st.slider("Credit duration (months)", 4, 72, 24)
            amount = st.number_input("Credit amount (DM)", 250, 20000, 5000)
            age = st.slider("Age", 19, 75, 35)

        with col_b:
            installment_rate = st.slider("Installment rate (1-4)", 1, 4, 2)
            present_residence = st.slider("Present residence (1-4)", 1, 4, 2)
            existing_credits = st.slider("Existing credits", 1, 4, 1)

        with col_c:
            dependents = st.slider("Dependents", 1, 2, 1)
            status = st.selectbox("Checking account status", ["A11", "A12", "A13", "A14"])
            credit_history = st.selectbox("Credit history", ["A30", "A31", "A32", "A33", "A34"])

        submitted = st.form_submit_button("🔍 Assess Risk", use_container_width=True)

    if submitted:
        input_data = pd.DataFrame(np.zeros((1, len(columns_list))), columns=columns_list)

        if 'duration' in input_data.columns:
            input_data['duration'] = duration
        if 'amount' in input_data.columns:
            input_data['amount'] = amount
        if 'age' in input_data.columns:
            input_data['age'] = age
        if 'installment_rate' in input_data.columns:
            input_data['installment_rate'] = installment_rate
        if 'present_residence' in input_data.columns:
            input_data['present_residence'] = present_residence
        if 'existing_credits' in input_data.columns:
            input_data['existing_credits'] = existing_credits
        if 'dependents' in input_data.columns:
            input_data['dependents'] = dependents

        if f'status_{status}' in input_data.columns:
            input_data[f'status_{status}'] = 1
        if f'credit_history_{credit_history}' in input_data.columns:
            input_data[f'credit_history_{credit_history}'] = 1

        proba = model.predict_proba(input_data)[0][1]
        prediction = model.predict(input_data)[0]

        st.markdown("---")
        st.markdown("### 📋 Assessment Result")

        col_r1, col_r2 = st.columns([1, 2])
        with col_r1:
            st.metric("Default Probability", f"{proba:.1%}")
        with col_r2:
            if prediction == 1:
                st.markdown('<div class="risk-high"><strong>⚠️ HIGH RISK</strong><br/>The model classifies this client as likely to default.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="risk-low"><strong>✅ LOW RISK</strong><br/>The model classifies this client as reliable.</div>', unsafe_allow_html=True)

        st.markdown("### Risk Gauge")
        st.progress(float(proba))

        st.markdown("### 🧠 SHAP Decision Explanation")
        st.markdown("The waterfall plot below decomposes the prediction into individual feature contributions.")

        try:
            shap_values = explainer.shap_values(input_data)

            if isinstance(shap_values, list):
                shap_vals = shap_values[1][0]
                base_val = explainer.expected_value[1]
            elif len(shap_values.shape) == 3:
                shap_vals = shap_values[0, :, 1]
                base_val = explainer.expected_value[1]
            else:
                shap_vals = shap_values[0]
                base_val = explainer.expected_value

            fig, ax = plt.subplots(figsize=(10, 6))
            explanation = shap.Explanation(
                values=shap_vals,
                base_values=float(np.array(base_val).ravel()[0]) if hasattr(base_val, '__len__') else float(base_val),
                data=input_data.iloc[0].values,
                feature_names=input_data.columns.tolist()
            )
            shap.plots.waterfall(explanation, max_display=10, show=False)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not generate SHAP explanation: {e}")

# ============================================
# TAB 2: GLOBAL ANALYSIS
# ============================================
with tab2:
    st.markdown("### Global Feature Importance (SHAP)")
    st.markdown("This summary plot shows which features are most influential across all clients in the test set.")

    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values_global, X_test, show=False, max_display=15)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not generate summary plot: {e}")

    st.markdown("---")
    st.markdown("### Top 10 Most Influential Features")

    importance_df = pd.DataFrame({
        'Rank': range(1, 11),
        'Feature': ['status_A14', 'status_A11', 'duration', 'credit_history_A34',
                    'savings_A61', 'amount', 'status_A12', 'savings_A65',
                    'property_A121', 'age'],
        'SHAP Importance': [0.0718, 0.0386, 0.0306, 0.0215, 0.0207,
                            0.0159, 0.0128, 0.0125, 0.0124, 0.0123]
    })
    st.dataframe(importance_df, use_container_width=True, hide_index=True)

    st.markdown("### Interpretation")
    st.markdown("""
    - **Checking account status** (`status_A14`, `status_A11`) dominates the model's decisions, indicating that a client's current banking situation is the strongest predictor of default.
    - **Credit duration** (`duration`) and **credit history** (`credit_history_A34`) confirm their central role in credit risk assessment.
    - **Savings** (`savings_A61`, `savings_A65`) reflect the client's financial fragility.
    - **Age** appears in the top 10, raising ethical considerations regarding the use of protected attributes.
    """)

# ============================================
# TAB 3: MODEL COMPARISON
# ============================================
with tab3:
    st.markdown("### Comparative Analysis of Ensemble Models")

    df_results = pd.DataFrame(model_results).T
    df_results = df_results.round(4)
    df_results.columns = ['AUC-ROC', 'F1', 'Recall', 'Precision']

    st.markdown("#### Performance Table")
    st.dataframe(df_results, use_container_width=True)

    st.markdown("#### Best Model")
    st.success("**Random Forest** with `class_weight='balanced'` achieves the best AUC-ROC (0.802) and the highest Recall (0.717).")

    st.markdown("#### Visual Comparison")
    fig, ax = plt.subplots(figsize=(10, 5))
    df_results.plot(kind='bar', ax=ax, rot=0, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    st.pyplot(fig)

    st.markdown("#### Analysis")
    st.markdown("""
    - **XGBoost** and **LightGBM** (unweighted) achieve similar results, with Recall around 0.52.
    - **Random Forest** with class weighting significantly improves Recall (0.717), detecting nearly three-quarters of actual defaults.
    - The trade-off: Random Forest has slightly lower Precision (0.573 vs 0.608), but in credit risk, **Recall is prioritized** because missing a defaulter is far more costly than rejecting a good client.
    """)

# ============================================
# TAB 4: ABOUT & METHODOLOGY
# ============================================
with tab4:
    st.markdown("### About This Project")

    st.markdown("""
    **Title:** Credit Risk Prediction using Machine Learning: A Comparative Analysis of Ensemble Models and Interpretability via SHAP

    | | |
    |---|---|
    | **Author** | [Votre nom] |
    | **Supervisor** | Prof. Seema Gupta Bhol |
    | **Institution** | KIIT University |
    | **Program** | M.Sc. Computer Science |
    | **Year** | 2025–2026 |
    """)

    st.markdown("---")
    st.markdown("### Methodology")

    st.markdown("""
    **Dataset**
    - German Credit dataset (UCI Machine Learning Repository)
    - 1,000 instances, 20 explanatory features, 1 binary target
    - Class distribution: 70% good credit / 30% bad credit

    **Preprocessing**
    - Target recoding: 1 (good) → 0, 2 (bad) → 1
    - One-hot encoding of categorical variables → 61 features
    - Stratified 80/20 train/test split
    - Class imbalance handled via `class_weight='balanced'`

    **Models**
    - XGBoost (n_estimators=200, max_depth=4, learning_rate=0.1)
    - LightGBM (n_estimators=200, max_depth=4, learning_rate=0.1)
    - Random Forest (n_estimators=200, max_depth=6, class_weight='balanced')

    **Evaluation Metrics**
    - AUC-ROC, F1-score, Recall, Precision
    - Accuracy deliberately excluded due to class imbalance

    **Interpretability**
    - SHAP (SHapley Additive exPlanations)
    - Global: summary plot for feature importance
    - Local: waterfall plot for individual predictions

    **Deployment**
    - Streamlit web application
    - Deployed on Streamlit Community Cloud
    - Source code hosted on GitHub
    """)

# ============================================
# FOOTER
# ============================================
st.markdown(f"""
<div class="footer">
    © {datetime.now().year} — Noellie Anne Laurence Badjo KOKORA —KIIT University — Credit Risk Prediction using ML and SHAP<br/>
    Credit Risk Prediction using Ensemble Models and SHAP Interpretability
</div>
""", unsafe_allow_html=True)


