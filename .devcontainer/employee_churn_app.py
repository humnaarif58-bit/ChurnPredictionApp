import streamlit as st
import pandas as pd
from joblib import load
import dill

with open('pipeline.pkl', 'rb') as file:
    model = dill.load(file)

my_feature_dict = load('my_feature_dict.pkl')

def predict_churn(data):
    prediction = model.predict(data)
    return prediction

st.set_page_config(page_title="Employee Churn Prediction", page_icon="🧑", layout="wide")

st.markdown(
    """
    <h1 style='text-align:center; color:#2E8B57; font-size:32px;'>🧑 Employee Churn Prediction App</h1>
    <p style='text-align:center; font-size:16px;'>Predict whether an Employee is likely to churn based on input features.</p>
    """,
    unsafe_allow_html=True
)

st.sidebar.header("🧩 Input Features")
st.sidebar.markdown("<p style='font-size:14px; color:grey;'>Select feature values below:</p>", unsafe_allow_html=True)

st.sidebar.subheader("Categorical Features")
categorical_input = my_feature_dict.get('CATEGORICAL')
categorical_input_vals = {}

for i, col in enumerate(categorical_input.get('Column Name').values()):
    categorical_input_vals[col] = st.sidebar.selectbox(col, categorical_input.get('Members')[i], key=col)

st.sidebar.subheader("Numerical Features")
numerical_input = my_feature_dict.get('NUMERICAL')
numerical_input_vals = {}

for col in numerical_input.get('Column Name'):
    numerical_input_vals[col] = st.sidebar.number_input(col, key=col)

input_data = dict(list(categorical_input_vals.items()) + list(numerical_input_vals.items()))
input_data = pd.DataFrame.from_dict(input_data, orient='index').T

st.markdown("---")
st.subheader("Developed by Humna Arif")

if st.button("Predict"):
    prediction = predict_churn(input_data)[0]
    translation_dict = {"Yes": "Expected", "No": "Not Expected"}
    prediction_translate = translation_dict.get(prediction)
    color = "#FF4B4B" if prediction == "Yes" else "#00C851"
    st.markdown(
        f"""
        <div style="background-color:{color}; padding:20px; border-radius:10px; text-align:center; color:white;">
            <h2 style='font-size:26px;'>Prediction: {prediction}</h2>
            <h4 style='font-size:18px;'>Employee is <b>{prediction_translate}</b> to churn.</h4>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:grey; font-size:14px;'>Based on Employee Dataset</p>",
    unsafe_allow_html=True
)
