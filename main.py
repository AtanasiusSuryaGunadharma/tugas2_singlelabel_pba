import streamlit as st
from utils.data_loader import load_data
import seaborn as sns
from sklearn.model_selection import train_test_split
from models.multi_label_classifiers import get_multilabel_classifier, create_vectorizer, evaluate_multilabel_model, create_multilabel_target
from utils.visualization import plot_multilabel_confusion_matrix
from streamlit_extras.let_it_rain import rain
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from datetime import datetime, timedelta
import time
import base64
from pathlib import Path
from st_social_media_links import SocialMediaIcons

# Initialize session state variables to store the model and vectorizer
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'model_name' not in st.session_state:
    st.session_state.model_name = None
if 'label_column' not in st.session_state:
    st.session_state.label_column = None

# Load data
if 'df' not in st.session_state:
    st.session_state.df = load_data()

# Set page configuration
st.set_page_config(
    page_title="Single-label Text Classification",
    layout="wide"
)

# Add title and description
st.title("Modul 2 - Single-label Text Classification")
st.markdown(
    "Single label klasifikasi teks menggunakan model Random Forest, SVM, dan Multinomial Naive Bayes."
)

# Main page content
st.write("""
## Welcome to the Single-label Text Classification App
         
This application demonstrates text classification using various machine learning models.

### Available Pages:

1. **Dataset Explorer** - Explore and understand the dataset
2. **Model Training** - Train and evaluate different classification models
3. **Prediction** - Make predictions on new text inputs

Use the sidebar to navigate between pages.
""")

# Show dataset overview
st.subheader("Dataset Overview")
df = st.session_state.df
st.write(f"Number of samples: {df.shape[0]}")
st.write(f"Number of features: {df.shape[1]}")
st.dataframe(df.head(5))

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()
    
def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Menambahkan audio autoplay menggunakan HTML
try:
    with open(r"lagu_picapica.mp3", "rb") as audio_file:
        audio_base64 = base64.b64encode(audio_file.read()).decode()

    audio_html = f"""
    <audio autoplay loop>
        <source src="data:audio/mpeg;base64,{audio_base64}" type="audio/mpeg">
        Your browser does not support the audio element.
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)
except FileNotFoundError:
    st.error("File audio tidak ditemukan. Pastikan 'natal_lagu3.mp3' sudah ada di direktori project.")
    
# Change Background Streamlit
set_background(r"background_music2.gif")
