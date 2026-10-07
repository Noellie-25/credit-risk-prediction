
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

# Title
st.title("💳 Credit Risk Prediction System")
st.markdown("### Model: Random Forest with SHAP Interpretability")

# Load model and explainer
@st.cache_resource
def load_model():
    model = joblib.load('random_forest_model.pkl')
    explainer = joblib.load('shap_explainer.pkl')
    return model, explainer

model, explainer = load_model()

# Model performance
st.markdown("---")
st.markdown("### 📊 Model Performance")
col1, col2, col3, col4 = st.columns(4)
col1.metric("AUC-ROC", "0.802")
col2.metric("F1-score", "0.637")
col3.metric("Recall", "0.717")
col4.metric("Precision", "0.573")

st.markdown("---")
st.markdown("### 🔍 Client Prediction")
st.markdown("Fill in the client information below, then click **Predict**.")

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

# Prediction processing
if submitted:
    # Build input DataFrame with the same columns as X_test
    input_data = pd.DataFrame(np.zeros((1, len(X_test.columns))), columns=X_test.columns)
    
    # Numeric fields
    input_data['duration'] = duration
    input_data['amount'] = amount
    input_data['age'] = age
    input_data['installment_rate'] = installment_rate
    input_data['present_residence'] = present_residence
    input_data['existing_credits'] = existing_credits
    input_data['dependents'] = dependents
    
    # Categorical fields (one-hot)
    if f'status_{status}' in input_data.columns:
        input_data[f'status_{status}'] = 1
    if f'credit_history_{credit_history}' in input_data.columns:
        input_data[f'credit_history_{credit_history}'] = 1
    
    # Prediction
    proba = model.predict_proba(input_data)[0][1]
    prediction = model.predict(input_data)[0]
    
    # Display result
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
    
    # Local SHAP explanation
    st.markdown("### 🧠 Decision Explanation (SHAP)")
    st.markdown("The chart below shows how each feature influenced the decision.")
    
    try:
        shap_values = explainer.shap_values(input_data)
        
        # Normalize structure
        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0]
            base_val = explainer.expected_value[1]
        elif len(shap_values.shape) == 3:
            shap_vals = shap_values[0, :, 1]
            base_val = explainer.expected_value[1]
        else:
            shap_vals = shap_values[0]
            base_val = explainer.expected_value
        
        # Waterfall plot
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
        st.info("The explanation will be available in the final version.")

# Footer
st.markdown("---")
st.caption("M.Sc. Computer Science Dissertation — KIIT University — Credit Risk Prediction using ML and SHAP")
