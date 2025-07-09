
import streamlit as st
import pandas as pd
import joblib

# Load saved model and tools
model = joblib.load("polymer_rf_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

st.title("🔬 Polymer Type Classifier")

uploaded_file = st.file_uploader("Upload your polymer CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Input Data")
    st.write(df.head())

    # Preprocess uploaded data
    X_new = df.drop(columns=["Unnamed: 0", "label", "smiles"])
    X_scaled = scaler.transform(X_new)

    # Make predictions
    y_pred = model.predict(X_scaled)
    y_label = y_pred  # No need for inverse_transform if already string

    st.subheader("🔬 Predicted Polymer Types")
    st.write(pd.DataFrame(y_label, columns=["Predicted Type"]))
