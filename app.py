import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go


model = joblib.load('model.pkl')

st.set_page_config(
    page_title="Insurance AI",
    layout="wide"
)

st.title("Insurance Claim Auto Approval AI")

st.sidebar.header("Customer Details")


seniority = st.sidebar.slider(
    "Customer Seniority",
    0, 30, 5
)

premium = st.sidebar.number_input(
    "Premium",
    100.0,
    10000.0,
    500.0
)

claims_history = st.sidebar.slider(
    "Claims History",
    0, 20, 1
)

claim_ratio = st.sidebar.slider(
    "Claim Ratio",
    0.0, 5.0, 0.2
)

vehicle_value = st.sidebar.number_input(
    "Vehicle Value",
    1000.0,
    100000.0,
    10000.0
)

power = st.sidebar.slider(
    "Vehicle Power",
    50, 500, 100
)

driver_age = st.sidebar.slider(
    "Driver Age",
    18, 90, 35
)

experience = st.sidebar.slider(
    "Driving Experience",
    1, 60, 10
)

vehicle_age = st.sidebar.slider(
    "Vehicle Age",
    0, 30, 5
)

lapse = st.sidebar.selectbox(
    "Lapse",
    [0, 1]
)

policies = st.sidebar.slider(
    "Policies In Force",
    1, 10, 1
)


input_data = pd.DataFrame({
    'Seniority': [seniority],
    'Policies_in_force': [policies],
    'Premium': [premium],
    'N_claims_history': [claims_history],
    'R_Claims_history': [claim_ratio],
    'Lapse': [lapse],
    'Value_vehicle': [vehicle_value],
    'Power': [power],
    'Driver_Age': [driver_age],
    'Driving_Experience': [experience],
    'Vehicle_Age': [vehicle_age]
})

if st.button("Predict"):

    probability = model.predict_proba(
        input_data
    )[0][1]

    if probability >= 0.85:
        decision = "AUTO APPROVED"
        risk = "LOW RISK"
        color = "green"

    elif probability >= 0.60:
        decision = "MANUAL REVIEW"
        risk = "MEDIUM RISK"
        color = "orange"

    else:
        decision = "HIGH RISK"
        risk = "HIGH"
        color = "red"

    st.subheader(decision)

    st.metric(
        "Approval Probability",
        f"{probability*100:.2f}%"
    )

    st.metric(
        "Risk Level",
        risk
    )

    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': "Approval Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color}
        }
    ))

    st.plotly_chart(fig)