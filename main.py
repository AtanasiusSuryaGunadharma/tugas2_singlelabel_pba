import streamlit as st
from utils.data_loader import load_data

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
