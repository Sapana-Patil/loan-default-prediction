import pandas as pd
import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap


# Load files
model = pickle.load(open('loan_default_model.pkl', 'rb'))
threshold = pickle.load(open('threshold.pkl', 'rb'))
features =pickle.load(open('feature_names.pkl', 'rb'))
target = pickle.load(open('target.pkl', 'rb'))
X=pickle.load(open('X.pkl', 'rb'))

st.title("Loan Default Predictor")
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
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(input_data)
    shap_values.feature_names = ['Credit Limit', 'Age', 'Late 30-59',
                                 'Debt Ratio', 'Income', 'Open Credits',
                                 'Late 90+', 'Real Estate', 'Late 60-89',
                                 'Dependents']
    fig, ax = plt.subplots()
    shap.plots.bar(shap_values[0], show=False)
    st.pyplot(fig)


#Feature Importance Chart
with st.container():
    st.header("Feature Importance Chart")
    fig_imp=pd.Series(model.feature_importances_,
                  index=['Credit Limit Used', 'Age', 'Late 30-59',
                            'Debt Ratio', 'Monthly Income', 'Open Credits',
                            'Late 90+', 'Real Estate', 'Late 60-89', 'Dependents'])
    fig , ax=plt.subplots()
    fig_imp.sort_values().plot(kind='barh', ax=ax)
    ax.set_title("Feature Importance")
    st.pyplot(fig)

    st.header(" Exploratory Data Analysis")
    st.subheader("Default vs Non-Default")
    fig,ax=plt.subplots()
    target.value_counts().plot(kind='bar', ax=ax)
    ax.set_xlabel("0 = No Default, 1 = Default")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Loan Default")
    st.pyplot(fig)

    st.subheader("Correlation Matrix")
    fig,ax=plt.subplots()
    X.columns = ['Credit Limit', 'Age', 'Late 30-59',
                 'Debt Ratio', 'Income', 'Open Credits',
                 'Late 90+', 'Real Estate', 'Late 60-89', 'Dependents']
    corr = X.corr()
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr,vmin=-1,vmax=1,cmap=cmap,ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig)







