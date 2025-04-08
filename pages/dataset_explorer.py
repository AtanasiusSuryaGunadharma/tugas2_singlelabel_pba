import streamlit as st
import pandas as pd
from utils.visualization import plot_label_distribution
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

st.set_page_config(page_title="Dataset Explorer", layout="wide")
st.title("Dataset Explorer")

# Access data from session state
df = st.session_state.df

# Show basic dataset info
st.subheader("Dataset Overview")
st.write(f"Number of samples: {df.shape[0]}")
st.write(f"Number of features: {df.shape[1]}")

# Display sample data
st.subheader("Data")
st.dataframe(df.head(20))

# Choose which label to explore
label_to_explore = st.selectbox(
    "Choose label to explore:",
    ["fuel", "machine", "part", "others", "price", "service"]
)

# Show label distribution
st.subheader(f"{label_to_explore.capitalize()} Label Distribution")
fig = plot_label_distribution(df, label_to_explore)
st.pyplot(fig)

# Show some insights
st.subheader("Label Distribution")
col1, col2 = st.columns(2)

with col1:
    st.write("Distribution by sentiment")
    sentiment_counts = df[label_to_explore].value_counts()
    st.dataframe(sentiment_counts)

with col2:
    st.write("Percentage distribution")
    sentiment_percent = df[label_to_explore].value_counts(normalize=True) * 100
    st.dataframe(sentiment_percent.round(2).astype(str) + '%')

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
