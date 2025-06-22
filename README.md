# bankruptcy-predictor
# 💼 Bankruptcy Prediction System

An interactive, AI-powered Streamlit web application that predicts whether a company is likely to go bankrupt based on 94 financial ratios.

## 🔍 Overview

This project uses a machine learning model trained on the **Taiwan Bankruptcy Prediction Dataset**. It integrates model interpretability (via SHAP values) and offers a clean, user-friendly interface for business users to upload financial data and get predictions.

---

## 🧠 Features

- 🔮 **Bankruptcy Prediction** using XGBoost
- 📈 **SHAP Explainability** to understand key features
- 📄 **CSV Upload** and Template Download
- 🔁 **Automated Prediction Pipeline** (every 3 days)
- 🧮 **Probability Scores** for prediction confidence
- 🎨 Clean & responsive Streamlit UI

---

## 🗂️ Repository Structure

📁 bankruptcy-predictor/
├── app.py # Streamlit web app
├── final_xgb_model.pkl # Trained XGBoost model
├── new_data.csv # Template file for uploading financial ratios
├── prediction.csv # Automatically generated predictions
├── bankruptcy_pipeline.py # Scheduled pipeline for auto predictions
├── requirements.txt # Python dependencies
└── README.md # Project documentation

---

## 📊 Dataset Used

- **Source**: Taiwan Bankruptcy Prediction (UCI Repository)
- **Instances**: 6819 companies
- **Features**: 94 financial ratios
- **Target**: `Bankrupt?` (1 = Yes, 0 = No)

---

## 🚀 How to Run Locally

### 1. Clone this repository:

```bash
git clone https://github.com/your-username/bankruptcy-predictor.git
cd bankruptcy-predictor
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run app.py
python bankruptcy_pipeline.py
