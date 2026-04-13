import streamlit as st
import pandas as pd
import joblib

#load model
model = joblib.load("models/churn_model.pkl")

st.title("Customer Churn Prediction")

#inputs 
tenure = st.number_input("Tenure", 0, 100)
monthly_charges = st.number_input("Monthly Charges", 0, 10000)
total_charges = st.number_input("Total Charges", 0, 100000)

#prediction
if st.button("Predict"):
    input_data = pd.DataFrame({
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    
    model_columns = model.feature_names_in_
    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Customer will churn ❌")
    else:
        st.success("Customer will stay ✅")