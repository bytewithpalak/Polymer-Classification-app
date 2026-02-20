import streamlit as st
import pandas as pd
import joblib

# Load trained objects
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

st.set_page_config(page_title="Polymer Classifier", layout="centered")
st.title("🔬 Polymer Type Classifier")
st.markdown("Upload a CSV file with 2048 fingerprint features.")

uploaded_file = st.file_uploader("Upload fingerprint dataset (.csv)", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data Preview")
    st.write(df.head())

    # Remove unwanted columns if present
    df = df.drop(columns=["Unnamed: 0", "label", "smiles"], errors="ignore")

    try:
        X_scaled = scaler.transform(df)
        predictions = model.predict(X_scaled)
        labels = encoder.inverse_transform(predictions)

        result_df = pd.DataFrame({
            "Prediction": labels
        })

        st.subheader("🧪 Predictions")
        st.write(result_df)

    except Exception as e:
        st.error("Error processing file. Make sure it contains the correct 2048 fingerprint features.")
