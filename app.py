import streamlit as st
import joblib
import numpy as np

# 1. Load model and scaler
model = joblib.load('heart_rf_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")
st.title('Heart Disease Prediction System')
st.write('Enter patient data and the model will predict the risk')
st.markdown("---")

# 2. Inputs - all 13 features
col1, col2 = st.columns(2)

with col1:
    age = st.number_input('Age', 20, 100, 50)
    sex = st.selectbox('Sex', [0, 1], format_func=lambda x: 'Female 0' if x==0 else 'Male 1')
    cp = st.selectbox('Chest Pain Type CP', [0,1,2,3], format_func=lambda x: ['Typical Angina','Atypical Angina','Non-anginal Pain','Asymptomatic'][x])
    trestbps = st.number_input('Resting BP', 80, 200, 120)
    chol = st.number_input('Cholesterol', 100, 600, 200)
    fbs = st.selectbox('Fasting Blood Sugar > 120', [0,1], format_func=lambda x: 'No 0' if x==0 else 'Yes 1')
    restecg = st.selectbox('Rest ECG', [0,1,2], format_func=lambda x: ['Normal','ST-T abnormality','LV hypertrophy'][x])

with col2:
    thalach = st.number_input('Max Heart Rate', 60, 220, 150)
    exang = st.selectbox('Exercise Induced Angina', [0,1], format_func=lambda x: 'No 0' if x==0 else 'Yes 1')
    oldpeak = st.number_input('ST Depression', 0.0, 6.0, 1.0, 0.1)
    slope = st.selectbox('ST Slope', [0,1,2], format_func=lambda x: ['Upsloping','Flat','Downsloping'][x])
    ca = st.selectbox('Major Vessels CA', [0,1,2,3,4])
    thal = st.selectbox('Thal', [0,1,2,3], format_func=lambda x: ['Normal','Fixed defect','Reversable defect','Unknown'][x])

# 3. Predict Button
if st.button('Predict Heart Disease Risk'):
	input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
	input_scaled = scaler.transform(input_data)
	prediction = model.predict(input_scaled)
	prediction_proba = model.predict_proba(input_scaled)
if prediction[0] == 1:
	st.error(f'⚠️ High Risk of Heart Disease - {prediction_proba[0][1]*100:.2f}%')
else:
	st.success(f'✅ No Heart Disease Detected - {prediction_proba[0][0]*100:.2f}%')