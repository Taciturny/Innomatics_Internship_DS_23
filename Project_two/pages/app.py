import streamlit as st
from matplotlib import image
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

IMAGE_PATH = os.path.join(dir_of_interest, "images", "heart_disease.jpg")
DATA_PATH = os.path.join(dir_of_interest, "data", "heart-disease.csv")

st.title("Heart Disease Dashboard")

img = image.imread(IMAGE_PATH)
st.image(img, caption=None, width=200, use_column_width=100, clamp=False, channels="RGB", output_format="auto")



df = pd.read_csv(DATA_PATH)
st.dataframe(df.head(10))
st.text('The table shows the dataset sample')


st.sidebar.header("Please Select Here:")
gender = st.sidebar.selectbox(
    "Select the Gender:",
    options=df["sex"].unique()
)
st.sidebar.info("Men=1 and Women=0")

disease = st.sidebar.selectbox(
    "Select Heart Disease:",
    options=df["target"].unique()
)
st.sidebar.info("Disease=1 and No Disease=0")

col1, col2, col3, col4  = st.columns(4)

fig_1 = px.box(df[df['sex'] == gender], x="age")
col1.plotly_chart(fig_1, use_container_width=True)

fig_2 = px.histogram(df[df['sex'] == gender], x="chol")
col2.plotly_chart(fig_2, use_container_width=True)

fig_3 = px.bar(df[df['target'] == disease], x="thalach")
col2.plotly_chart(fig_3, use_container_width=True)

fig_4 = px.histogram(df[df['target'] == disease], x="cp")
col2.plotly_chart(fig_4, use_container_width=True)


    


