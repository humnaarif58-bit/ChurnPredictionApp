import streamlit as st
import pandas as pd
from joblib import load
import dill
import matplotlib.pyplot as plt
from sklearn import tree
import numpy as np

# Load model and features
with open('pipeline.pkl', 'rb') as file:
    model = dill.load(file)
my_feature_dict = load('my_feature_dict.pkl')

def predict_churn(data):
    prediction = model.predict(data)
    return prediction

# Page config
st.set_page_config(page_title="Employee Churn Prediction", page_icon="🧑", layout="wide")

# Header
st.markdown(
    """
    <h1 style='text-align:center; color:#2E8B57; font-size:30px; margin-bottom:0;'>🧑 Employee Churn Prediction</h1>
    <p style='text-align:center; color:grey; font-size:14px; margin-top:2px;'>Predict if an employee is likely to leave based on inputs</p>
    """,
    unsafe_allow_html=True
)

# Adjusted layout: 30% left, 70% right
col1, col2 = st.columns([0.85, 2])

# ---------------- LEFT COLUMN ----------------
with col1:
    st.markdown("<h4 style='color:#2E8B57; margin-bottom:4px;'>🧩 Input Features</h4>", unsafe_allow_html=True)
    st.markdown("<p style='color:grey; font-size:12px; margin-top:0;'>Please fill in all fields below:</p>", unsafe_allow_html=True)

    categorical_input = my_feature_dict.get('CATEGORICAL')
    categorical_input_vals = {}

    for i, col in enumerate(categorical_input.get('Column Name').values()):
        categorical_input_vals[col] = st.selectbox(
            f"{col}:", categorical_input.get('Members')[i], key=col
        )

    numerical_input = my_feature_dict.get('NUMERICAL')
    numerical_input_vals = {}
    for col in numerical_input.get('Column Name'):
        numerical_input_vals[col] = st.number_input(f"{col}:", key=col)

    input_data = dict(list(categorical_input_vals.items()) + list(numerical_input_vals.items()))
    input_df = pd.DataFrame.from_dict(input_data, orient='index').T

# ---------------- RIGHT COLUMN ----------------
with col2:
    # Predict button (centered)
    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
    predict_btn = st.button("✨ Predict Churn", use_container_width=False)
    st.markdown("</div>", unsafe_allow_html=True)

    if predict_btn:
        prediction = predict_churn(input_df)[0]
        translation_dict = {"Yes": "Expected", "No": "Not Expected"}
        prediction_translate = translation_dict.get(prediction)
        color = "#FF4B4B" if prediction == "Yes" else "#00C851"

        # Draw Decision Tree (only one small tree)
        clf = model.named_steps['classifier'] if hasattr(model, 'named_steps') else model
        if hasattr(clf, 'estimators_'):
            estimator = clf.estimators_[0]
        else:
            estimator = clf

        if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
            preprocessor = model.named_steps['preprocessor']
            X_transformed = preprocessor.transform(input_df)
        else:
            X_transformed = input_df.select_dtypes(include=[np.number])

        plt.figure(figsize=(8, 3))
        tree.plot_tree(
            estimator,
            feature_names=[f"F{i}" for i in range(X_transformed.shape[1])],
            class_names=["No", "Yes"],
            filled=True,
            rounded=True,
            proportion=True
        )
        st.pyplot(plt.gcf())

        # Smaller Prediction Result Box
        st.markdown(
            f"""
            <div style="background-color:{color}; padding:12px; border-radius:8px; text-align:center; color:white; margin-top:12px; max-width:500px; margin-left:auto; margin-right:auto;">
                <h3 style='font-size:24px; margin:0;'>Prediction: {prediction}</h3>
                <p style='font-size:16px; margin-top:4px;'>Employee is <b>{prediction_translate}</b> to churn.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Footer
    st.markdown("<hr style='margin-top:25px;'>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center; font-size:15px; color:grey;'>Developed by <b>Humna Arif</b></p>",
        unsafe_allow_html=True
    )
