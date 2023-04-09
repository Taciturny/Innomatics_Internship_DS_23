import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import joblib
import json

RESOURCES_PATH = os.path.join(os.getcwd(), 'resource')
dictPath = os.path.join(RESOURCES_PATH,'laptopdict.json')
modelPath = os.path.join(RESOURCES_PATH,'model.pkl')


# Load the saved model
model = joblib.load(modelPath)

# Load the dictionary containing the laptop specifications
with open(dictPath, 'r') as f:
    laptopdict = json.load(f)

# Set up the Streamlit app
st.title('Estimating Laptop Price')
st.text("""Please enter the specifications of the laptop you're interested in, \n
and we'll provide an estimated price.""")

# Create the form for user input
with st.form('loan_form', clear_on_submit=True):
    ram_size = st.selectbox('Input Ram Size', laptopdict[0].keys())
    ram_size = laptopdict[0][ram_size]
    
    ram_type = st.selectbox('Input Ram Type', laptopdict[1].keys())
    ram_type = laptopdict[1][ram_type]
    
    processor = st.selectbox('Processor', laptopdict[2].keys())
    processor = laptopdict[2][processor]
    
    storage = st.selectbox('Storage', laptopdict[3].keys())
    storage = laptopdict[3][storage]
    
    storage_type = st.selectbox('Storage Type', laptopdict[4].keys())
    storage_type = laptopdict[4][storage_type]
    
    os_type = st.selectbox('OS', laptopdict[5].values())
    laptopdict[5] = {value: key for key, value in laptopdict[5].items()}
    os_type = laptopdict[5][os_type]
    
    features = np.array([ram_size, ram_type, processor, storage, storage_type, os_type]).reshape(1,-1)
    
    # Add the submit button
    submitted = st.form_submit_button(label='Submit')
    
    if submitted:
        prediction = round(np.exp(model.predict(features))[0])
        st.markdown(f'The Estimated Price of the Laptop ₹{prediction}')
        st.snow()
