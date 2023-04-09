from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump,load
import streamlit as st
from PIL import Image

# Set the app configuration and banner image
st.set_page_config(
    page_title="Tweets classifier",
    page_icon="📜",
    layout="wide"
)

# Add a banner image
st.markdown("<h1 style='text-align: center;'>Arabic and Yemeni Tweets Sentiment Analysis Classifier </h1>", unsafe_allow_html=True)

# Hide the menu and footer
hide_menu_style = """
    <style>
        #MainMenu {visibility : hidden}
        footer {visibility : hidden}
    </style>
    """
st.markdown(hide_menu_style, unsafe_allow_html=True)

# Load the trained model
loaded_model = load('random_forest_model.joblib')

# Create the form to input tweets
with st.form(key='values'):
    st.write("📰 Input tweets to classify")
    new_data = st.text_area("📰 Input tweets to classify", label_visibility = "collapsed")
    
    submitted_data = st.form_submit_button(label = 'Classify')

    # Classify the new data and show the result
    if submitted_data:
        new_data_prediction = loaded_model.predict([new_data])

        if new_data_prediction == -1:
            st.error("Negative")
        elif new_data_prediction == 0:
            st.success("Neutral")
        else:
            st.success("Positive")
