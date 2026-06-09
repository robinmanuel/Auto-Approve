import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import uuid

from pipeline import AutoApprovePipeline
from ocr import verify_document
from consistency_checker import checker
from approval_engine import ApprovalEngine

# -----------------------------
# LOAD MODELS
# -----------------------------
model = joblib.load("model.pkl")
pipeline = AutoApprovePipeline("parts_segmentation.pt")
engine = ApprovalEngine()

# -----------------------------
# APP CONFIG
# -----------------------------
st.set_page_config(page_title="SmartClaim AI", layout="wide")

st.title("SmartClaim AI")
st.subheader("AI-Powered Auto Insurance Claim Approval")

# -----------------------------
# SESSION STATE
# -----------------------------
defaults = {
    "customer_probability": None,
    "pipeline_result": None,
    "documents": {},
    "document_result": None,
    "consistency_result": None
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
st.sidebar.header("Customer Details")

seniority = st.sidebar.slider("Customer Seniority", 0, 30, 5)
premium = st.sidebar.number_input("Premium", 100.0, 10000.0, 500.0)
claims_history = st.sidebar.slider("Claims History", 0, 20, 1)

vehicle_value = st.sidebar.number_input("Vehicle Value", 1000.0, 1000000.0, 10000.0)
power = st.sidebar.slider("Vehicle Power", 50, 500, 100)
driver_age = st.sidebar.slider("Driver Age", 18, 90, 35)
experience = st.sidebar.slider("Driving Experience", 1, 60, 10)
vehicle_age = st.sidebar.slider("Vehicle Age", 0, 30, 5)
policies = st.sidebar.slider("Policies In Force", 1, 10, 1)

# -----------------------------
# INPUT DATA
# -----------------------------
input_data = pd.DataFrame({
    "Seniority": [seniority],
    "Policies_in_force": [policies],
    "Premium": [premium],
    "N_claims_history": [claims_history],
    "Value_vehicle": [vehicle_value],
    "Power": [power],
    "Driver_Age": [driver_age],
    "Driving_Experience": [experience],
    "Vehicle_Age": [vehicle_age]
})

col1, col2, col3 = st.columns(3)

# =====================================================
# 1. CUSTOMER RISK
# =====================================================
with col1:
    st.header("Customer Risk")

    if st.button("Predict Customer Risk"):
        prob = model.predict_proba(input_data)[0][1]
        st.session_state.customer_probability = prob

        st.metric("Approval Probability", f"{prob * 100:.2f}%")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={"text": "Customer Score"},
            gauge={"axis": {"range": [0, 100]}}
        ))
        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# 2. DAMAGE PIPELINE
# =====================================================
with col2:
    st.header("Damage Analysis")

    uploaded_image = st.file_uploader(
        "Upload Vehicle Damage Image",
        type=["jpg", "jpeg", "png"]
    )

    if st.button("Run Damage Detection"):

        if uploaded_image is None:
            st.warning("Upload image first")
        else:
            image_path = f"temp_{uuid.uuid4().hex}.jpg"

            with open(image_path, "wb") as f:
                f.write(uploaded_image.getbuffer())

            result = pipeline.analyze(image_path)
            st.session_state.pipeline_result = result

            st.success("Damage Analysis Complete")

            st.write("Total Cost:", result.get("total_estimated_cost", 0))
            st.write("Parts Found:", len(result.get("parts", [])))

            for p in result.get("parts", []):
                st.json(p)

# =====================================================
# 3. DOCUMENT VERIFICATION (SEPARATE BUTTON)
# =====================================================
with col3:
    st.header("Documents")

    document_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if st.button("Verify Documents"):

        if not document_files:
            st.warning("Upload documents first")
        else:
            st.session_state.documents = {}

            for file in document_files:
                temp_path = f"temp_{uuid.uuid4().hex}_{file.name}"

                with open(temp_path, "wb") as f:
                    f.write(file.getbuffer())

                doc_result = verify_document(temp_path)

                doc_type = doc_result["document_type"]
                doc_type = doc_type.strip().upper().replace(" ", "_")

                st.session_state.documents[doc_type] = doc_result

            st.success("Documents Verified")

            for k, v in st.session_state.documents.items():
                with st.expander(k):
                    st.json(v["fields"])

            required = ["POLICY", "RC", "DRIVING_LICENSE", "INVOICE"]
            missing = [d for d in required if d not in st.session_state.documents]

            doc_score = max(0, 100 - len(missing) * 25)

            st.session_state.document_result = {
                "score": doc_score,
                "missing": missing
            }

            st.write("Document Score:", doc_score)

# =====================================================
# 4. FINAL APPROVAL ENGINE (NEW BUTTON)
# =====================================================
st.markdown("---")
st.header("Final Approval")

if st.button("Run Final Approval Engine"):

    if st.session_state.customer_probability is None:
        st.warning("Run Customer Risk first")

    elif st.session_state.pipeline_result is None:
        st.warning("Run Damage Detection first")

    elif st.session_state.document_result is None:
        st.warning("Verify Documents first")

    else:

        consistency_result = checker.evaluate(
            st.session_state.documents,
            vehicle_value
        )

        st.session_state.consistency_result = consistency_result

        final_result = engine.evaluate(
            st.session_state.customer_probability,
            st.session_state.pipeline_result,
            st.session_state.document_result,
            consistency_result
        )

        st.subheader("Final Decision")

        st.write(final_result["decision"])

        colA, colB = st.columns(2)

        with colA:
            st.metric("Approval Score", f"{final_result.get('approval_score', 0)}%")
            st.metric("Risk Level", final_result["risk_level"])

        with colB:
            st.metric("Consistency Score", consistency_result.get("score", 0))

        st.subheader("Reasons")

        for r in final_result.get("reasons", []):
            st.write("•", r)