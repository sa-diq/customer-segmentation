import streamlit as st
import pandas as pd
import joblib

# Load machine learning model
filename = 'randomforest_model.joblib'

rf_classifier = joblib.load(filename)

# Load data
df = pd.read_csv("Cluster_customer_data.csv")

# Set Streamlit options
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Customer Segmentation")

# Background
st.markdown('<style>body{background-color: Blue;}</style>', unsafe_allow_html=True)

# Create a form to take user inputs
with st.form("segmentation_form"):
    gender = st.selectbox("Select Gender", ["", "Male", "Female"])
    age = st.number_input("Age", min_value=18)
    annual_income = st.number_input("Annual Income", min_value=0)
    annual_income = int(annual_income / 1000) if annual_income >= 1000 else annual_income
    spending_score = st.number_input("Spending Score", min_value=0, max_value=99)

    submitted = st.form_submit_button("Predict")

if submitted:
    # Encode gender into numerical values
    gender_encoded = 0 if gender == "Female" else 1

    # Input data for prediction
    data = [[gender_encoded, age, annual_income, spending_score]]

    # Make predictions
    cluster = rf_classifier.predict(data)[0]

    st.write(f"The model predicts that the customer belongs to Cluster: {cluster}")

    # Filter data for the predicted cluster
    cluster_df = df[df['Cluster'] == cluster]
