import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import image
import pickle
import os
import pickle


# Set up the Streamlit app
st.set_page_config(page_title="Heart Disease Predictor", page_icon=":heart:")
st.title(":red[_Heart Disease Predictor_]")

# absolute path to this file
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# absolute path to this file's root directory
PARENT_DIR = os.path.join(FILE_DIR, os.pardir)
# absolute path of directory_of_interest
dir_of_interest = os.path.join(PARENT_DIR, "resources")
model_of_interest = os.path.join(PARENT_DIR, "resources", "models")

IMAGE_PATH = os.path.join(dir_of_interest, "image", "heart_disease.jpg")
DATA_PATH = os.path.join(dir_of_interest, "data", "heart-disease.csv")
MODEL_Path = os.path.join(model_of_interest,'nb_model.pkl')
with open(MODEL_Path, 'rb') as file:
    model = pickle.load(file)

# st.title("Heart Disease Prediction")

img = image.imread(IMAGE_PATH)
st.image(img, caption=None, width=400, use_column_width=200, clamp=False, channels="RGB", output_format="auto")



df = pd.read_csv(DATA_PATH)

# Define a function to make predictions
def predict_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    inputs = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(inputs)[0]
    if prediction == 1:
        return "The patient is likely to have heart disease."
    else:
        return "The patient is unlikely to have heart disease."

# Add some explanatory text
st.markdown("This app predicts whether a patient is likely to have heart disease based on their medical information.")

# Add sliders and dropdowns for the input features
age = st.slider("Age", min_value=1, max_value=120, value=30)
sex = st.selectbox("Sex", options=["Male", "Female"])
cp = st.selectbox("Chest Pain Type", options=[1, 2, 3, 4])
trestbps = st.slider("Resting Blood Pressure (mm Hg)", min_value=1, max_value=300, value=120)
chol = st.slider("Serum Cholesterol (mg/dL)", min_value=1, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", options=["Yes", "No"])
restecg = st.selectbox("Resting Electrocardiographic Results", options=[0, 1, 2])
thalach = st.slider("Maximum Heart Rate Achieved", min_value=1, max_value=300, value=150)
exang = st.selectbox("Exercise Induced Angina", options=["Yes", "No"])
oldpeak = st.slider("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[1, 2, 3])
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3])
thal = st.selectbox("Thalassemia", options=[1, 2, 3, "Unknown"])

# Convert the input features to the correct data types
sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0
if thal == "Unknown":
    thal = np.nan

# Make a prediction and display the result
if st.button("Predict"):
    result = predict_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
    st.write(result)
