import streamlit as st
import pandas as pd
import pickle
import shap
from streamlit_shap import st_shap
import xgboost as xgb

st.set_page_config(page_title="Bankruptcy Prediction", layout="wide")

st.markdown(
    """
    <style>
    .main {background-color: #0e1117; color: white;}
    .sidebar .sidebar-content {background-color: #161a23;}
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
with open("final_xgb_model.pkl", "rb") as f:
    model = pickle.load(f)
st.title("💼 Bankruptcy Prediction System")
st.markdown("""
This interactive app uses a trained **XGBoost model** to predict the likelihood of bankruptcy based on financial ratios.

**Dataset Source:** Taiwan Bankruptcy Prediction Data  
**Features Used:** 94 financial indicators  
**Target:** Bankruptcy (1) or Not (0)
""")

st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📄 Upload CSV", "📊 Predict"])

if page == "🏠 Home":
    st.header("Welcome 👋")
    st.markdown("""
    - Use the **Upload CSV** tab to load your company data.
    - Go to **Predict** to test new inputs and view model explanations.
    - Need help formatting your CSV? 👉 [Download the 94-feature template below]
    """)
    with open("new_data.csv", "rb") as file:
        st.download_button("📥 Download CSV Template", file, file_name="new_data_template.csv")
elif page == "📄 Upload CSV":
    st.header("📄 Upload Data File")
    uploaded_file = st.file_uploader("Upload a CSV file with 94 financial features", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        if data.shape[1] != 94:
            st.error(f" Uploaded file has {data.shape[1]} columns. Expected 94.")
        else:
            st.success(" File uploaded successfully!")
            st.dataframe(data.head())
elif page == " Predict":
    st.header(" Predict Bankruptcy")
    uploaded_file = st.file_uploader("Upload a single-row CSV for prediction", type=["csv"], key="predict_upload")

    if uploaded_file:
        input_df = pd.read_csv(uploaded_file)
        if input_df.shape[1] != 94:
            st.error(f"❌ Expected 94 features, but got {input_df.shape[1]}. Please upload correct data.")
        else:
            prediction = model.predict(input_df)[0]
            pred_proba = model.predict_proba(input_df)[0][1]
            input_df["Prediction"] = prediction
            input_df["Probability"] = round(pred_proba, 4)
            result = "🔴 Likely to go Bankrupt" if prediction == 1 else "🟢 Not Likely to go Bankrupt"
            st.subheader("🧾 Prediction Result:")
            st.write(f"**{result}** with probability: **{pred_proba:.2%}**")
            st.dataframe(input_df)
            confidence_level = ""
            if pred_proba > 0.85:
                confidence_level = "🔴 Very High Confidence"
            elif pred_proba > 0.65:
                confidence_level = "🟠 High Confidence"
            elif pred_proba > 0.5:
                confidence_level = "🟡 Moderate Confidence"
            else:
                confidence_level = "🟢 Low Confidence"
            st.markdown("### 📈 Confidence Level")
            st.success(f"Prediction Confidence: **{pred_proba:.2%}** – {confidence_level}")
            st.markdown("### 🔍 Model Explanation (SHAP)")
            explainer = shap.Explainer(model)
            shap_values = explainer(input_df)
            st_shap(shap.plots.bar(shap_values), height=400)
