import streamlit as st
import joblib
from src.data_preprocessing import preprocess_single_input

# Load model and tools
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

# UI
st.title("🎓 Student Dropout Prediction")

sex = st.selectbox("Sex", ["F", "M"])
age = st.slider("Age", 15, 22, 17)
study_time = st.slider("Study time", 1, 4, 2)
absences = st.slider("Absences", 0, 100, 4)
G1 = st.slider("Grade 1", 0, 20, 10)
G2 = st.slider("Grade 2", 0, 20, 10)
internet = st.selectbox("Internet access", ["yes", "no"])

input_data = {
    "sex": sex,
    "age": age,
    "studytime": study_time,
    "absences": absences,
    "G1": G1,
    "G2": G2,
    "internet": internet
}

if st.button("Predict"):
    X_input = preprocess_single_input(input_data, feature_columns, scaler)
    prediction = model.predict(X_input)[0]

    if prediction == 1:
        st.error("❌ At Risk of Dropping Out")
    else:
        st.success("✅ Likely to Continue")
