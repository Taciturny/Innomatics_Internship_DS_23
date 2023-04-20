import streamlit as st
import pandas as pd
import os
from PIL import Image, ImageEnhance
import streamlit_folium as sf
# from folium.plugins import MarkerCluster
from streamlit_folium.plugins import MarkerCluster
from sklearn.neighbors import BallTree
from streamlit_folium import st_folium_static as folium_static
# from streamlit_folium import folium_static


st.set_page_config(page_title="Pub Finder App",
                   page_icon=":🍺:",
                   layout="centered")

# absolute path to this file
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# absolute path to this file's root directory
PARENT_DIR = os.path.join(FILE_DIR, os.pardir)
# absolute path of directory_of_interest
dir_of_interest = os.path.join(PARENT_DIR, "resources")

HOME_IMAGE_PATH = os.path.join(dir_of_interest, "image", "pub_home.jpg")
LOCATIONS_IMAGE_PATH = os.path.join(dir_of_interest, "image", "pub_location.jpg")
NEAREST_IMAGE_PATH = os.path.join(dir_of_interest, "image", "pub_nearest.jpg")
RANDOM_IMAGE_PATH = os.path.join(dir_of_interest, "image", "pub_random.jpg")
DATA_PATH = os.path.join(dir_of_interest, "data", "pub_data.csv")

st.markdown("<h1 style='color: red;'>Pub Finder App</h1>", unsafe_allow_html=True)

# Load home page image
home_img = Image.open(HOME_IMAGE_PATH)
enhancer = ImageEnhance.Contrast(home_img)
home_img = enhancer.enhance(1.5)

# Load locations page image
locations_img = Image.open(LOCATIONS_IMAGE_PATH)
enhancer = ImageEnhance.Contrast(locations_img)
locations_img = enhancer.enhance(1.5)

# Load nearest pub page image
nearest_img = Image.open(NEAREST_IMAGE_PATH)
enhancer = ImageEnhance.Contrast(nearest_img)
nearest_img = enhancer.enhance(1.5)

# Load random pub page image
random_img = Image.open(RANDOM_IMAGE_PATH)
enhancer = ImageEnhance.Contrast(random_img)
random_img = enhancer.enhance(1.5)

# Load data
df = pd.read_csv(DATA_PATH)

# Page 1 - Home Page
def home():
    st.image(home_img, caption=None, width=500, use_column_width=200, clamp=False, channels="RGB", output_format="auto")
    st.write("This app allows you to find pubs in the United Kingdom (UK) and discover their locations.")
    st.write("The dataset used in this app contains information about over 50,000 pubs in the UK.")
    st.write("You can navigate to the other pages using the menu on the left.")

    # Show basic information and statistics about the dataset
    st.write("Here are some basic statistics about the dataset:")
    st.write("- Total number of pubs:", df.shape[0])
    st.write("- Number of unique local authorities:", df["local_authority"].nunique())
    st.write("- Number of unique postal codes:", df["postcode"].nunique())

# Page 2 - Pub Locations
def pub_locations():
    st.image(locations_img, caption=None, width=500, use_column_width=200, clamp=False, channels="RGB", output_format="auto")
    st.title("Pub Locations")

    # Select filter type
    filter_type = st.selectbox("Select filter type", ["Postal Code", "Local Authority"])

    # Filter by postal code or local authority
    if filter_type == "Postal Code":
        selected_code = st.text_input("Enter Postal Code")
        pubs = df[df["postcode"] == selected_code]
    else:
        selected_authority = st.selectbox("Select Local Authority", df["local_authority"].unique())
        pubs = df[df["local_authority"] == selected_authority]

    # Check if pubs DataFrame is empty
    if pubs.empty:
        st.write("No pubs found for selected filter.")
    else:
        # Show pubs on a map
        st.write("Map of selected pubs:")
        m = sf.folium_static(sf.Map(location=[pubs["latitude"].mean(), pubs["longitude"].mean()], zoom_start=13))

        marker_cluster = MarkerCluster().add_to(m)

        for _, row in pubs.iterrows():
            sf.Marker([row["latitude"], row["longitude"]], popup=row["name"]).add_to(marker_cluster)

        folium_static(m)


# # Page 3 - Find the nearest Pub
def find_nearest_pub():
    st.image(nearest_img, caption=None, width=500, use_column_width=200, clamp=False, channels="RGB", output_format="auto")
    st.title("Find the Nearest Pub")

    # Get user's location
    user_lat = st.text_input("Enter your Latitude")
    user_lon = st.text_input("Enter your Longitude")

    # Convert to float and filter by distance
    if user_lat and user_lon:
        user_location = [[float(user_lat), float(user_lon)]]
        pubs = df[["latitude", "longitude"]].dropna()
        ball_tree = BallTree(pubs, metric="haversine")
        dist, ind = ball_tree.query(user_location, k=5)
        nearest_pubs = df.iloc[ind[0]]

        # Show pubs on a map
        st.write("Map of nearest pubs:")
        m = folium.Map(location=[user_lat, user_lon], zoom_start=15)
        marker_cluster = MarkerCluster().add_to(m)
        for _, row in nearest_pubs.iterrows():
            popup_message = f"<b>{row['name']}</b><br>{row['address']}<br>"
            folium.Marker([row["latitude"], row["longitude"]], popup=popup_message).add_to(marker_cluster)
        folium_static(m)



# Page 4 - Random Pub Selector
def random_pub():
    st.image(random_img, caption=None, width=400, use_column_width=100, clamp=False, channels="RGB", output_format="auto")
    st.title("Random Pub Selector")

    # Select a random pub from the dataset
    random_row = df.sample(n=1)

    # Show the pub on a map
    st.write("Map of selected pub:")
    m = folium.Map(location=[random_row["latitude"].iloc[0], random_row["longitude"].iloc[0]], zoom_start=13)
    folium.Marker([random_row["latitude"].iloc[0], random_row["longitude"].iloc[0]], popup=random_row["name"].iloc[0]).add_to(m)
    folium_static(m)


menu = ["Home", "Pub Locations", "Find the Nearest Pub", "Random Pub Selector"]
choice = st.sidebar.selectbox("Select a page", menu)

if choice == "Home":
    home()
elif choice == "Pub Locations":
    pub_locations()
elif choice == "Find the Nearest Pub":
    find_nearest_pub()
elif choice == "Random Pub Selector":
    random_pub()
