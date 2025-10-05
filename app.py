import streamlit as st
import pandas as pd
import numpy as np
import joblib
from streamlit_lottie import st_lottie
import requests

log_model = joblib.load("logistic_model.pkl")
rf_model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Loan Default Predictor", page_icon="💰", layout="wide")

st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
        font-family: 'Poppins', sans-serif;
    }
    .main {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 40px;
        margin: 20px auto;
        box-shadow: 0 4px 25px rgba(0,0,0,0.1);
        max-width: 900px;
    }
    .stButton>button {
        background-color: #2e7d32 !important;
        color: white !important;
        border-radius: 12px !important;
        font-size: 18px !important;
        padding: 10px 24px !important;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #43a047 !important;
        transform: scale(1.05);
    }
    h1, h2, h3, h4 {
        color: #2e7d32;
        font-weight: 600;
    }
    .result {
        font-size: 22px;
        font-weight: bold;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main'>", unsafe_allow_html=True)

st.title("💰 Loan Default Prediction Dashboard")
st.markdown(
    "Predict whether a borrower will **fully repay** their loan or **default**, using Machine Learning models."
)

def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_loan = load_lottie_url("https://assets8.lottiefiles.com/packages/lf20_jcikwtux.json")
st_lottie(lottie_loan, height=200, key="loan")


st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.radio("Select Model", ["Logistic Regression", "Random Forest"])


st.header("📋 Applicant Details")

col1, col2 = st.columns(2)

with col1:
    credit_policy = st.selectbox("Credit Policy (1 = Meets, 0 = Doesn’t)", [1, 0])
    int_rate = st.number_input("Interest Rate", value=0.12, format="%.4f")
    installment = st.number_input("Installment Amount", value=300.0)
    log_annual_inc = st.number_input("Log Annual Income", value=11.0)
    dti = st.number_input("Debt-to-Income Ratio", value=15.0)
    fico = st.number_input("FICO Score", value=700)

with col2:
    days_with_cr_line = st.number_input("Days with Credit Line", value=4000.0)
    revol_bal = st.number_input("Revolving Balance", value=5000)
    revol_util = st.number_input("Revolving Utilization (%)", value=45.0)
    inq_last_6mths = st.number_input("Inquiries in Last 6 Months", value=0)
    delinq_2yrs = st.number_input("Delinquencies (2 Years)", value=0)
    pub_rec = st.number_input("Public Records", value=0)

purpose = st.selectbox(
    "Purpose of Loan",
    [
        "credit_card",
        "debt_consolidation",
        "educational",
        "home_improvement",
        "major_purchase",
        "small_business",
    ],
)

purpose_dict = {
    "credit_card": [1, 0, 0, 0, 0, 0],
    "debt_consolidation": [0, 1, 0, 0, 0, 0],
    "educational": [0, 0, 1, 0, 0, 0],
    "home_improvement": [0, 0, 0, 1, 0, 0],
    "major_purchase": [0, 0, 0, 0, 1, 0],
    "small_business": [0, 0, 0, 0, 0, 1],
}

purpose_features = purpose_dict[purpose]


input_data = np.array([
    credit_policy, int_rate, installment, log_annual_inc, dti, fico,
    days_with_cr_line, revol_bal, revol_util, inq_last_6mths,
    delinq_2yrs, pub_rec
] + purpose_features).reshape(1, -1)

feature_names = [
    "credit.policy", "int.rate", "installment", "log.annual.inc", "dti", "fico",
    "days.with.cr.line", "revol.bal", "revol.util", "inq.last.6mths",
    "delinq.2yrs", "pub.rec",
    "purpose_credit_card", "purpose_debt_consolidation", "purpose_educational",
    "purpose_home_improvement", "purpose_major_purchase", "purpose_small_business"
]

input_df = pd.DataFrame(input_data, columns=feature_names)
input_scaled = scaler.transform(input_df)

st.markdown("---")
if st.button("🚀 Predict Loan Status"):
    if model_choice == "Logistic Regression":
        prob = log_model.predict_proba(input_scaled)[0][1]
    else:
        prob = rf_model.predict_proba(input_scaled)[0][1]

    result = "❌ High Risk of Default" if prob > 0.5 else "✅ Low Risk - Likely to Repay"
    color = "rgba(255, 99, 132, 0.8)" if prob > 0.5 else "rgba(75, 192, 192, 0.8)"

    st.markdown(
        f"<div class='result' style='background:{color}; color:white;'>{result}</div>",
        unsafe_allow_html=True,
    )

    st.progress(float(prob))
    st.metric(label="Probability of Default", value=f"{prob:.2%}")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    "<p style='text-align:center; margin-top:30px;'>Made with ❤️ by <b>Abhishek</b> using Streamlit</p>",
    unsafe_allow_html=True,
)
