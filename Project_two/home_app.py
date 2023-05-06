import streamlit as st

# Set page title and favicon
st.set_page_config(page_title="Home", page_icon=":heart:", layout="wide")

st.title("Innomatics Data App")
st.snow()

def app():
    # Add an "About Me" section
    st.write("# About Me")
    st.write("Hi there! I'm **Uzoagulu, Promise Chinecherem**, a data science enthusiast and aspiring machine learning engineer. I created this heart disease predictor as a personal project to showcase my skills in machine learning and web development.")

    # What is heart disease?
    st.write("# About Heart Disease")
    st.write("Heart disease refers to various conditions that affect the heart. The most common form of heart disease is coronary artery disease, which occurs when the arteries that supply blood to the heart become narrow or blocked, leading to chest pain, shortness of breath, and other symptoms.")
    st.write("Other types of heart disease include heart failure, arrhythmias, and heart valve problems.")
    st.write("If you are concerned about your risk of heart disease, you should talk to your doctor about getting a heart health check. This may involve having your blood pressure, cholesterol, and other factors checked.")
    
    # Add a brief introduction to the heart disease predictor
    st.write("# Heart Disease Predictor")
    st.write("This app predicts the likelihood of a person having heart disease based on their health metrics, such as age, gender, blood pressure, and cholesterol levels. To get started, enter your health metrics in the sidebar called _app_ and click the 'Predict' button.")

    # Add links to your LinkedIn and GitHub accounts
    st.write("# Connect with Me")
    st.write("If you'd like to learn more about me or see more of my projects, feel free to connect with me on LinkedIn or check out my GitHub.")
    st.write("[LinkedIn](https://www.linkedin.com/in/promiseecheremuzoagulu)")
    st.write("[GitHub](https://github.com/Taciturny)")

if __name__ == '__main__':
    app()
