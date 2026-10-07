
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="💳",
    layout="wide"
)

# Load all artifacts
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

# Sidebar navigation
st.sidebar.title("💳 Navigation")
page = st.sidebar.radio(
    "Select a page",
    ["🔍 Client Prediction", "📊 Global Analysis", "📈 Model Comparison"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Info")
st.sidebar.info("**Random Forest** with SHAP interpretability")

# ============================================
# PAGE 1: CLIENT PREDICTION
# ============================================
if page == "🔍 Client Prediction":
    st.title("💳 Client Prediction")
    st.markdown("Fill in the client information below, then click **Predict**.")
    
    # Model performance
    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AUC-ROC", "0.802")
    col2.metric("F1-score", "0.637")
    col3.metric("Recall", "0.717")
    col4.metric("Precision", "0.573")
    
    st.markdown("---")
    
    # Input form
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
        
        submitted = st.form_submit_button("🔍 Predict Risk")
    
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
        st.markdown("### 📋 Prediction Result")
        
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.metric("Default probability", f"{proba:.1%}")
        with col_r2:
            if prediction == 1:
                st.error("⚠️ **BAD CREDIT** — High risk")
            else:
                st.success("✅ **GOOD CREDIT** — Low risk")
        
        st.markdown("### 🧠 Decision Explanation (SHAP)")
        st.markdown("The chart below shows how each feature influenced the decision.")
        
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
# PAGE 2: GLOBAL ANALYSIS
# ============================================
elif page == "📊 Global Analysis":
    st.title("📊 Global Analysis")
    st.markdown("### Feature Importance (SHAP Summary Plot)")
    st.markdown("This plot shows which features are most influential across all clients.")
    
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values_global, X_test, show=False, max_display=15)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not generate summary plot: {e}")
    
    st.markdown("---")
    st.markdown("### 📋 Top 10 Most Important Features")
    st.markdown("""
    | Rank | Feature | Importance |
    |------|---------|------------|
    | 1 | `status_A14` | 0.0718 |
    | 2 | `status_A11` | 0.0386 |
    | 3 | `duration` | 0.0306 |
    | 4 | `credit_history_A34` | 0.0215 |
    | 5 | `savings_A61` | 0.0207 |
    | 6 | `amount` | 0.0159 |
    | 7 | `status_A12` | 0.0128 |
    | 8 | `savings_A65` | 0.0125 |
    | 9 | `property_A121` | 0.0124 |
    | 10 | `age` | 0.0123 |
    """)
    
    st.markdown("---")
    st.markdown("### 🔍 Interpretation")
    st.markdown("""
    - **Checking account status** (`status_A14`, `status_A11`) dominates, indicating that a client's current banking situation is the strongest predictor of default.
    - **Credit duration** (`duration`) and **credit history** (`credit_history_A34`) confirm their central role.
    - **Savings** (`savings_A61`, `savings_A65`) reflect financial fragility.
    - **Age** appears in the top 10 — this raises ethical considerations regarding the use of protected attributes.
    """)

# ============================================
# PAGE 3: MODEL COMPARISON
# ============================================
elif page == "📈 Model Comparison":
    st.title("📈 Model Comparison")
    st.markdown("### Comparative Analysis of Three Ensemble Models")
    
    # Build DataFrame
    df_results = pd.DataFrame(model_results).T
    df_results = df_results.round(4)
    df_results.columns = ['AUC-ROC', 'F1', 'Recall', 'Precision']
    
    st.markdown("---")
    st.markdown("### 📊 Performance Table")
    st.dataframe(df_results, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 🏆 Best Model")
    st.success("**Random Forest** with `class_weight='balanced'` achieves the best AUC-ROC (0.802) and the highest Recall (0.717).")
    
    st.markdown("---")
    st.markdown("### 📈 Visual Comparison")
    
    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    df_results.plot(kind='bar', ax=ax, rot=0)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    st.pyplot(fig)
    
    st.markdown("---")
    st.markdown("### 🔍 Analysis")
    st.markdown("""
    - **XGBoost** and **LightGBM** (unweighted) achieve similar results, with Recall around 0.52.
    - **Random Forest** with `class_weight='balanced'` significantly improves Recall (0.717), detecting nearly three-quarters of actual defaults.
    - The trade-off: Random Forest has slightly lower Precision (0.573 vs 0.608), but in credit risk, **Recall is prioritized** because missing a defaulter is far more costly than rejecting a good client.
    """)

# Footer
st.markdown("---")
st.caption("Noellie Anne Laurence Badjo KOKORA — 25141020 — KIIT University — Credit Risk Prediction using ML and SHAP")
