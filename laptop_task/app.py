import os
import json
import joblib
import pandas as pd
import streamlit as st
import numpy as np


st.set_page_config(
    page_title="Estimating Laptop Price",
    page_icon=":computer:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "This is a Laptop Price Estimation App!"}
)

# Set the page configuration
# st.set_page_config(page_title='Estimating Laptop Price', page_icon=':computer:', layout='wide', menu_items={'About': 'This is a Laptop Price Estimation App!'})
# st.set_page_config(page_title="My Streamlit App", page_icon=":computer:", layout="wide", initial_sidebar_state="expanded", backgroundColor="#F8F8FF")


RESOURCES_PATH = os.path.join(os.getcwd(), 'resource')
dictPath = os.path.join(RESOURCES_PATH,'laptopdict.json')
modelPath = os.path.join(RESOURCES_PATH,'model.pkl')


# Load the saved model
model = joblib.load(modelPath)

# Load the dictionary containing the laptop specifications
with open(dictPath, 'r') as f:
    laptopdict = json.load(f)

# Set up the Streamlit app
st.title(':blue[Estimating Laptop Price]')
st.text("""Please enter the specifications of the laptop you're interested in, and we'll provide an estimated price.""")

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

    display = st.selectbox('Display', laptopdict[6].values())
    laptopdict[6] = {value: key for key, value in laptopdict[6].items()}
    display = laptopdict[6][display]
    
    features = [ram_size, ram_type, processor, storage, storage_type, os_type, display]

    # Create a pandas dataframe with the laptop specifications
    features_df = pd.DataFrame([features], columns=['ram_size', 'ram_type', 'processor', 'storage', 'storage_type', 'os_type', 'display'])

    features_df = pd.DataFrame({
        'ram_size': [ram_size],
        'ram_type': [ram_type],
        'processor': [processor],
        'storage': [storage],
        'storage_type': [storage_type],
        'os_type' : [os_type],
        'display' : [display]
}, dtype=int)
  
  # Add the submit button
    submitted = st.form_submit_button(label='Submit')
    
    if submitted:
        prediction = round(np.exp(model.predict(features_df))[0])
        st.markdown(f'The Estimated Price of the Laptop ₹{prediction}')
        st.snow()
