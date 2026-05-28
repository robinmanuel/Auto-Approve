import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

from image_model import predict_damage

model = joblib.load('model.pkl')

st.set_page_config(
    page_title="Insurance AI",
    layout="wide"
)

st.title("SmartClaim AI")

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

uploaded_image = st.file_uploader(
    "Upload Vehicle Damage Image",
    type=['jpg', 'jpeg', 'png']
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

customer_probability = None
damage_result = None
overall_probability = None

col1, col2, col3 = st.columns(3)

with col1:

    if st.button("Predict Customer Risk"):

        customer_probability = model.predict_proba(
            input_data
        )[0][1]

        st.session_state[
            'customer_probability'
        ] = customer_probability

        st.subheader(
            "Customer Risk Analysis"
        )

        st.metric(
            "Customer Approval Score",
            f"{customer_probability*100:.2f}%"
        )

        fig_customer = go.Figure(

            go.Indicator(

                mode="gauge+number",

                value=(
                    customer_probability
                    * 100
                ),

                title={
                    'text':
                    "Customer Approval Score"
                },

                gauge={

                    'axis': {
                        'range': [0, 100]
                    },

                    'bar': {
                        'color': "blue"
                    }
                }
            )
        )

        st.plotly_chart(
            fig_customer,
            use_container_width=True
        )

with col2:

    if st.button("Analyze Vehicle Damage"):

        if uploaded_image is not None:

            image_path = (
                "temp_uploaded_image.jpg"
            )

            with open(
                image_path,
                "wb"
            ) as f:

                f.write(
                    uploaded_image.getbuffer()
                )

            damage_result = predict_damage(
                image_path
            )

            st.session_state[
                'damage_result'
            ] = damage_result

            st.image(
                uploaded_image,
                caption="Uploaded Vehicle Image",
                use_container_width=True
            )

            st.subheader(
                "Damage Analysis"
            )

            st.write(
                f"Damage Type : "
                f"{damage_result['Damage_Type']}"
            )

            st.write(
                f"Confidence : "
                f"{damage_result['Confidence']}%"
            )

            st.write(
                f"Model Used : "
                f"{damage_result.get('Model_Used', 'DamageNet')}"
            )

            st.write(
                f"Estimated Repair Cost : "
                f"₹{damage_result['Estimated_Cost']:,.0f}"
            )

            damage_type = damage_result[
                'Damage_Type'
            ]

            image_score_map = {

                'no_damage': 0,
                'paint_scratch': 85,
                'dent': 75,
                'torn': 60,
                'broken_lamp': 50,
                'hole': 35,
                'lost_parts': 25,
                'broken_glass': 20
            }

            damage_score = image_score_map[
                damage_type
            ]

            fig_damage = go.Figure(

                go.Indicator(

                    mode="gauge+number",

                    value=damage_score,

                    title={
                        'text':
                        "Vehicle Claim Validity"
                    },

                    gauge={

                        'axis': {
                            'range': [0, 100]
                        },

                        'bar': {
                            'color': "purple"
                        }
                    }
                )
            )

            st.plotly_chart(
                fig_damage,
                use_container_width=True
            )

        else:

            st.warning(
                "Upload image first"
            )

with col3:

    if st.button("Overall Approval"):

        if (
            'customer_probability'
            in st.session_state
            and
            'damage_result'
            in st.session_state
        ):

            customer_probability = (
                st.session_state[
                    'customer_probability'
                ]
            )

            damage_result = (
                st.session_state[
                    'damage_result'
                ]
            )

            damage_type = damage_result[
                'Damage_Type'
            ]

            damage_penalty = {

                'no_damage': 1.00,
                'paint_scratch': 0.05,
                'dent': 0.10,
                'torn': 0.15,
                'broken_lamp': 0.20,
                'hole': 0.30,
                'lost_parts': 0.35,
                'broken_glass': 0.40
            }

            overall_probability = (

                customer_probability
                -
                damage_penalty[
                    damage_type
                ]

            )

            overall_probability = max(
                0,
                overall_probability
            )

            if damage_type == 'no_damage':
                if customer_probability > 0.80:
                    decision = (
                        "LIKELY FRAUDULENT CLAIM"
                    )
                    risk = "FRAUD ALERT"
                    color = "red"
                else:
                    decision = (
                        "CLAIM REJECTED - NO DAMAGE"
                    )
                    risk = "NO CLAIM"
                    color = "green"

            elif overall_probability >= 0.85:

                decision = (
                    "AUTO APPROVED"
                )

                risk = "LOW RISK"

                color = "green"

            elif overall_probability >= 0.60:

                decision = (
                    "MANUAL REVIEW"
                )

                risk = "MEDIUM RISK"

                color = "orange"

            else:

                decision = (
                    "SURVEYOR INSPECTION REQUIRED"
                )

                risk = "HIGH RISK"

                color = "red"

            st.subheader(
                decision
            )

            st.metric(
                "Overall Approval Score",
                f"{overall_probability*100:.2f}%"
            )

            st.metric(
                "Risk Level",
                risk
            )

            fig = go.Figure(

                go.Indicator(

                    mode="gauge+number",

                    value=(
                        overall_probability
                        * 100
                    ),

                    title={
                        'text':
                        "Overall Approval Score"
                    },

                    gauge={

                        'axis': {
                            'range': [0, 100]
                        },

                        'bar': {
                            'color': color
                        }
                    }
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.warning(
                "Run both predictions first"
            )