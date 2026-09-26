import streamlit as st
import pandas as pd
import joblib


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)


# ==========================================
# LOAD SAVED MODEL
# ==========================================

model = joblib.load("fraud_detection_model.pkl")
features = joblib.load("fraud_detection_features.pkl")
threshold = joblib.load("fraud_detection_threshold.pkl")


# ==========================================
# TITLE
# ==========================================

st.title("💳 Credit Card Fraud Detection")

st.write(
    "Enter the transaction details below to check "
    "whether the transaction is likely to be fraudulent."
)


# ==========================================
# INPUTS
# ==========================================

st.subheader("Transaction Details")


amount = st.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=100.0,
    step=1.0
)


hour = st.number_input(
    "Transaction Hour (0-23)",
    min_value=0,
    max_value=23,
    value=12,
    step=1
)


distance_from_home_km = st.number_input(
    "Distance From Home (km)",
    min_value=0.0,
    value=10.0,
    step=1.0
)


transactions_last_24h = st.number_input(
    "Transactions In Last 24 Hours",
    min_value=0,
    value=1,
    step=1
)


card_present = st.selectbox(
    "Card Present?",
    ["No", "Yes"]
)


international_transaction = st.selectbox(
    "International Transaction?",
    ["No", "Yes"]
)


new_device = st.selectbox(
    "New Device?",
    ["No", "Yes"]
)


failed_attempts = st.number_input(
    "Failed Attempts",
    min_value=0,
    value=0,
    step=1
)


# ==========================================
# CONVERT YES/NO TO 0/1
# ==========================================

card_present_value = 1 if card_present == "Yes" else 0

international_transaction_value = (
    1 if international_transaction == "Yes" else 0
)

new_device_value = 1 if new_device == "Yes" else 0


# ==========================================
# CREATE TRANSACTION DATA
# ==========================================

transaction = pd.DataFrame({
    "amount": [amount],
    "hour": [hour],
    "distance_from_home_km": [distance_from_home_km],
    "transactions_last_24h": [transactions_last_24h],
    "card_present": [card_present_value],
    "international_transaction": [international_transaction_value],
    "new_device": [new_device_value],
    "failed_attempts": [failed_attempts]
})


# Keep same feature order as training
transaction = transaction[features]


# ==========================================
# PREDICTION BUTTON
# ==========================================

if st.button("🔍 Predict Transaction"):

    fraud_probability = model.predict_proba(
        transaction
    )[:, 1][0]

    prediction = (
        fraud_probability >= threshold
    )


    # ==========================================
    # DISPLAY PROBABILITY
    # ==========================================

    st.subheader("Prediction Result")

    st.metric(
        "Fraud Probability",
        f"{fraud_probability * 100:.2f}%"
    )


    # ==========================================
    # DISPLAY RESULT
    # ==========================================

    if prediction:

        st.error("🚨 FRAUD TRANSACTION")

        st.write(
            "The model has classified this transaction "
            "as potentially fraudulent."
        )

    else:

        st.success("✅ NORMAL TRANSACTION")

        st.write(
            "The model has classified this transaction "
            "as normal."
        )


    # ==========================================
    # SHOW THRESHOLD
    # ==========================================

    st.info(
        f"Decision Threshold: {threshold:.2f}"
    )