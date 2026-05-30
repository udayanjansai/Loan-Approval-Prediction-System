import streamlit as st
import pandas as pd
import joblib


# =========================
# Streamlit Page Config
# =========================

st.set_page_config(
    page_title="Loan Risk Predictor",
    page_icon="💰",
    layout="wide",
)


# =========================
# Load Models
# =========================

@st.cache_resource
def load_artifacts():
    loan_model = joblib.load("loan_model.pkl")
    kmeans = joblib.load("kmeans.pkl")
    kmeans_scaler = joblib.load("kmeans_scaler.pkl")
    return loan_model, kmeans, kmeans_scaler


loan_model, kmeans, kmeans_scaler = load_artifacts()


# =========================
# Cluster Labels
# =========================

segment_map = {
    0: "Moderate-Risk Professionals",
    1: "Creditworthy Low-Income Customers",
    2: "Premium Low-Risk Customers",
    3: "High-Risk Borrowers",
    4: "Stable but Credit-Constrained Customers",
}


st.title("💰 Loan Approval Prediction System")
st.markdown("Predict loan default risk and customer segment")


# =========================
# Input Form
# =========================

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=70,
        value=35,
    )

    income = st.number_input(
        "Annual Income",
        min_value=10000,
        value=50000,
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=1000,
        value=100000,
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=650,
    )

    months_employed = st.number_input(
        "Months Employed",
        min_value=0,
        max_value=120,
        value=36,
    )

    num_credit_lines = st.selectbox(
        "Number of Credit Lines",
        [1, 2, 3, 4],
    )

    interest_rate = st.number_input(
        "Interest Rate (%)",
        min_value=0.0,
        max_value=30.0,
        value=10.0,
    )

with col2:
    loan_term = st.selectbox(
        "Loan Term (Months)",
        [12, 24, 36, 48, 60],
    )

    dti_ratio = st.slider(
        "DTI Ratio",
        0.0,
        0.9,
        0.3,
    )

    education = st.selectbox(
        "Education",
        ["Bachelor's", "Master's", "High School", "PhD"],
    )

    employment_type = st.selectbox(
        "Employment Type",
        ["Full-time", "Part-time", "Self-employed", "Unemployed"],
    )

    marital_status = st.selectbox(
        "Marital Status",
        ["Single", "Married", "Divorced"],
    )

    has_mortgage = st.selectbox(
        "Has Mortgage",
        ["Yes", "No"],
    )

    has_dependents = st.selectbox(
        "Has Dependents",
        ["Yes", "No"],
    )

    loan_purpose = st.selectbox(
        "Loan Purpose",
        ["Auto", "Business", "Education", "Home", "Other"],
    )

    has_cosigner = st.selectbox(
        "Has Co-Signer",
        ["Yes", "No"],
    )


# =========================
# Prediction Button
# =========================

if st.button("Predict Loan Status"):
    loan_income_ratio = loan_amount / income
    monthly_income = income / 12
    monthly_interest_rate = (interest_rate / 100) / 12

    if monthly_interest_rate == 0:
        emi = loan_amount / loan_term
    else:
        emi = (
            loan_amount
            * monthly_interest_rate
            * (1 + monthly_interest_rate) ** loan_term
            / ((1 + monthly_interest_rate) ** loan_term - 1)
        )

    emi_income_ratio = emi / monthly_income

    input_df = pd.DataFrame(
        {
            "Age": [age],
            "Income": [income],
            "LoanAmount": [loan_amount],
            "CreditScore": [credit_score],
            "MonthsEmployed": [months_employed],
            "NumCreditLines": [num_credit_lines],
            "InterestRate": [interest_rate],
            "LoanTerm": [loan_term],
            "DTIRatio": [dti_ratio],
            "Education": [education],
            "EmploymentType": [employment_type],
            "MaritalStatus": [marital_status],
            "HasMortgage": [has_mortgage],
            "HasDependents": [has_dependents],
            "LoanPurpose": [loan_purpose],
            "HasCoSigner": [has_cosigner],
            "LoanIncomeRatio": [loan_income_ratio],
            "EMIIncomeRatio": [emi_income_ratio],
        }
    )

    probability = loan_model.predict_proba(input_df)[0][1]
    prediction = 1 if probability >= 0.6 else 0

    if probability < 0.3:
        risk_category = "Low Risk"
    elif probability < 0.6:
        risk_category = "Medium Risk"
    else:
        risk_category = "High Risk"

    cluster_features = pd.DataFrame(
        [
            [
                age,
                income,
                credit_score,
                months_employed,
                num_credit_lines,
                dti_ratio,
            ]
        ],
        columns=[
            "Age",
            "Income",
            "CreditScore",
            "MonthsEmployed",
            "NumCreditLines",
            "DTIRatio",
        ],
    )

    cluster_scaled = kmeans_scaler.transform(cluster_features)
    cluster = kmeans.predict(cluster_scaled)[0]
    segment = segment_map.get(cluster, "Unknown Segment")

    if prediction == 0:
        loan_status = "Approved"
        status_color = "green"
    else:
        loan_status = "Rejected"
        status_color = "red"

    st.markdown("---")
    st.subheader("Prediction Result")

    st.markdown(
        f"<h2 style='color:{status_color};'>{loan_status}</h2>",
        unsafe_allow_html=True,
    )

    st.metric("Default Probability", f"{probability * 100:.2f}%")
    st.metric("Risk Category", risk_category)
    st.metric("Customer Segment", segment)
    st.progress(float(probability))
    st.info(f"Estimated default probability: {probability * 100:.2f}%")
