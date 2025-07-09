import streamlit as st
import pandas as pd
import joblib


model = joblib.load("polymer_rf_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

df = pd.read_csv("polymer_dataset.csv") 


st.set_page_config(page_title="Polymer Classifier", page_icon="🧪", layout="centered")

st.markdown("<h1 style='text-align:center; color:#6a0dad;'>🧪 Polymer Type Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Enter polymer properties to predict its type 💡</p>", unsafe_allow_html=True)
st.write("---")

st.subheader("🔬 Enter Polymer Properties")


temp = st.number_input("Glass Transition Temperature", min_value=0.0, max_value=500.0, value=100.0)
index = st.number_input("Refractive Index", min_value=0.0, max_value=3.0, value=1.5)
strength = st.number_input("Tensile Strength", min_value=0.0, max_value=1000.0, value=300.0)


user_input = pd.DataFrame({
    'Glass Transition Temperature': [temp],
    'Refractive Index': [index],
    'Tensile Strength': [strength]
})

# Predict button
if st.button("🔍 Predict Polymer Type"):
    try:
        X_scaled = scaler.transform(user_input)
        y_pred = model.predict(X_scaled)
        st.success(f"🧪 Predicted Polymer Type: {y_pred[0]}")
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center; font-size:13px;'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)
