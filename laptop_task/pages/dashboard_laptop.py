import os
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from resources.laptop_details import df
import plotly.express as px
import altair as alt


# absolute path to this file
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# absolute path to this file's root directory
PARENT_DIR = os.path.join(FILE_DIR, os.pardir)
# absolute path of directory_of_interest
dir_of_interest = os.path.join(PARENT_DIR, "resources")
DATA_PATH = os.path.join(dir_of_interest, "df.csv")

df = pd.read_csv(DATA_PATH)

# Define univariate chart
plt1 = sns.displot(data=df, x="price", kde=True,height=5,aspect=2)
plt2 = sns.catplot(data=df, x="ram_size", kind="count",height=5,aspect=2)
plt3 = sns.catplot(data=df, x="storage", kind="count",height=5,aspect=2)
os_counts = df.groupby('os')['os'].count().reset_index(name='count')
fig = px.pie(os_counts, names='os', title='OS Distribution')

univariatetext = """
The Above plots display univariate distributions of the dataset\n
* The first plot illustrates the distribution of the target variable (price) and highlights the left skewness,
 indicating the need for data transformation.\n
* The other plots show the count of various columns in the dataset, including Ram Size, Storage, and Operating System."""

# Define multivariate chart
fig1 = px.box(df, x='ram_size', y='price', title='Ram Size vs Price')
fig2 = px.box(df, x='storage', y='price', title='Storage vs Price')
fig3 = px.box(df, x='storage_type', y='price', title='StorageType vs Price')
fig4 = px.box(df, x='os', y='price', title='OS vs Price')
chart = alt.Chart(df).mark_boxplot(ticks=False).encode(
    x='ram_type:O',
    y='price:Q'
).properties(title="Ram Type vs Price")
chart = chart.configure_scale(
    bandPaddingInner=0.1
)

multivariatetext = """
The above box plots depict the relationships between the target variable, price, and various features in the dataset. 
Specifically:\n
* The First plot show the relationship between the Ram Size and price
and shows the price seems to increase with increase in ram size
* The Second Plot shows the relationship between Storage and price
The price also seems to increase with increase in storage size
* Also the other plots shows relationship between StorageType and OS
"""
st.title('UniVariate Plots')
st.pyplot(plt1)
st.pyplot(plt2)
st.pyplot(plt3)
st.plotly_chart(fig)
st.markdown(univariatetext)

st.title('MultiVariate Plots')
st.plotly_chart(fig1, theme=None, use_container_width=True)
st.plotly_chart(fig2, theme=None, use_container_width=True)
st.plotly_chart(fig3, theme=None, use_container_width=True)
st.plotly_chart(fig4, theme=None, use_container_width=True)
st.altair_chart(chart, use_container_width=True)


st.markdown(multivariatetext)
