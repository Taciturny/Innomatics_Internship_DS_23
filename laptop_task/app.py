import streamlit as st
# import numpy as np
import pandas as pd
import pickle
import os


RESOURCES_PATH = os.path.join(os.getcwd(), 'resources')
# Load the trained model
# model = pickle.load(open('best_model.pkl', 'rb'))
modelPath = os.path.join(RESOURCES_PATH,'best_model.pkl')
model = pickle.load(open(modelPath, 'rb'))




# Define the input form
def input_form():
    ram_size = st.number_input('RAM Size', min_value=1, max_value=2147483647)
    ram_type = st.number_input('RAM Type', min_value=1, max_value=2147483647)
    processor = st.number_input('Processor', min_value=1, max_value=2147483647)
    storage = st.number_input('Storage', min_value=1, max_value=2147483647)
    os = st.number_input('Operating System', min_value=1, max_value=2147483647)
    display = st.number_input('Display Size', min_value=1.0, max_value=100.0, step=0.1)
    storage_type = st.number_input('Storage Type', min_value=1, max_value=2147483647)

    # Store the user input as a dictionary
    user_input = {'ram_size': ram_size,
                  'ram_type': ram_type,
                  'processor': processor,
                  'storage': storage,
                  'os': os,
                  'display': display,
                  'storage_type': storage_type}

    return user_input

# Define the main function
def main():
    # Create a title for the app
    st.title('Laptop Price Predictor')

    # Create a description for the app
    st.write('This app predicts the price of a laptop based on its features.')

    # Get user input
    user_input = input_form()

    # Convert user input to DataFrame
    input_df = pd.DataFrame([user_input])
    
    pp = np.array(input_df). reshape (1,-1)

    # Make a prediction using the loaded model
    prediction = model.predict(pp)

    # Display the predicted price to the user
    st.write('The predicted price of the laptop is:', prediction[0])

if __name__ == '__main__':
    main()
