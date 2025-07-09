
import streamlit as st
import pandas as pd
import joblib

# Load saved model and tools
model = joblib.load("polymer_rf_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

# Set page layout
st.set_page_config(page_title="Polymer Classifier", page_icon="🧪", layout="centered")

# Header
st.markdown("<h1 style='text-align:center; color:#6a0dad;'>🧪 Polymer Type Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Upload your polymer data & let AI predict its type 💡</p>", unsafe_allow_html=True)
st.write("---")

# Sidebar for upload
st.sidebar.header("📂 Upload Your File")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# Main logic
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("📄 Uploaded Data")
        st.dataframe(df.head())

        # Drop unnecessary columns (ignore if not present)
        X_new = df.drop(columns=["Unnamed: 0", "label", "smiles"], errors="ignore")
        X_scaled = scaler.transform(X_new)
        y_pred = model.predict(X_scaled)
        y_label = y_pred  # already strings

        st.subheader("🔍 Predicted Polymer Types")
        st.success("✅ Successfully predicted!")
        st.dataframe(pd.DataFrame(y_label, columns=["Predicted Type"]))

    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.warning("⬅️ Please upload a CSV file to get started.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center; font-size:13px;'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)

