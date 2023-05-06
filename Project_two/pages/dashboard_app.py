import streamlit as st
from matplotlib import image
from PIL import Image
import pandas as pd
import plotly.express as px
import os
import numpy as np

st.set_page_config(page_title="Heart Disease Dashboard",
                   page_icon=":bar_chart:",
                   layout="wide")

# absolute path to this file
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# absolute path to this file's root directory
PARENT_DIR = os.path.join(FILE_DIR, os.pardir)
# absolute path of directory_of_interest
dir_of_interest = os.path.join(PARENT_DIR, "resources")

IMAGE_PATH = os.path.join(dir_of_interest, "image", "heart_disease.jpg")
DATA_PATH = os.path.join(dir_of_interest, "data", "heart-disease.csv")

st.title("Heart Disease Dashboard")

img = image.imread(IMAGE_PATH)
st.image(img, caption=None, width=300, use_column_width=200, clamp=False, channels="RGB", output_format="auto")

data = pd.read_csv(DATA_PATH)
st.dataframe(data.head(10))
st.text('The table shows the dataset sample')

# Sidebar
st.sidebar.title("Heart Disease Dashboard")
selected_columns = st.sidebar.multiselect("Select variables to plot", 
                                           ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
                                            "thalach", "exang", "oldpeak", "slope", "ca", "thal"])

# Main content
st.title("Heart Disease Dashboard")
st.write("Data source: [Heart Disease UCI](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)")

# Plots
if len(selected_columns) > 0:
    st.subheader("Plots")
    for col in selected_columns:
        fig = None
        if col in ["trestbps", "chol", "thalach", "oldpeak"]:
            fig = px.box(data, y=col, color="sex", points="all")
        elif col == "age":
            fig = px.histogram(data, x=col, nbins=10, color="sex")
        else:
            fig = px.violin(data, y=col, color="sex", box=True, points="all")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.write("Please select at least one variable to plot.")
