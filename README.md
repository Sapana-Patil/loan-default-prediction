#  Loan Default Predictor

A machine learning web app that predicts whether a customer will default on a loan based on their financial history.

##  Live Demo
 [Click here to try the app](https://loan-default-prediction-45.streamlit.app/)

##  Features
- Predicts loan default risk based on 10 financial features
- Shows probability of default in percentage
- SHAP values explaining WHY the prediction was made
- Feature Importance Chart
- EDA — Default Distribution & Correlation Heatmap
- Clean 2-column input interface

## Tech Stack
- Python, Pandas, Scikit-learn, XGBoost, SHAP, Streamlit

## How it Works
- Trained XGBoost classifier on 150,000 loan records
- Handled class imbalance using SMOTE
- Optimized decision threshold to 0.2 for better recall
- Used SHAP values for model explainability
- Deployed as live web app using Streamlit Cloud

##  Model Performance
- Accuracy: 91%
- Recall (Defaulters): 53%
- F1 Score: 0.43
