import streamlit as st
import pickle
import numpy as np

# Load files
model = pickle.load(open('loan_default_model.pkl', 'rb'))
threshold = pickle.load(open('threshold.pkl', 'rb'))

st.title("🏦 Loan Default Predictor")
st.write("Enter customer details to predict loan default risk")

col1, col2 = st.columns(2)

with col1:
    revolving = st.number_input("Credit Card Limit Used (0 to 1)", min_value=0.0, max_value=1.0, value=0.5)
    age = st.number_input("Age", min_value=18, max_value=100, value=18)
    late_30_59 = st.number_input("Times 30-59 Days Late", min_value=0, value=0)
    debt_ratio = st.number_input("Debt Ratio", min_value=0.0, value=0.3)
    monthly_income = st.number_input("Monthly Income ($)", min_value=0, value=5000)

with col2:
    open_credits = st.number_input("Total Open Loans/Credits", min_value=0, value=5)
    late_90 = st.number_input("Times 90+ Days Late", min_value=0, value=0)
    real_estate = st.number_input("Real Estate Loans", min_value=0, value=0)
    late_60_89 = st.number_input("Times 60-89 Days Late", min_value=0, value=0)
    dependents = st.number_input("Number of Dependents", min_value=0, value=0)

# Predict button
if st.button("Predict"):
    input_data = np.array([[revolving, age, late_30_59, debt_ratio,
                            monthly_income, open_credits, late_90,
                            real_estate, late_60_89, dependents]])

    probability = model.predict_proba(input_data)[:, 1][0]
    if probability >= threshold:
        st.error(f" High Risk! Probability of Default: {probability:.2%}")
    else:
        st.success(f" Low Risk! Probability of Default: {probability:.2%}")
