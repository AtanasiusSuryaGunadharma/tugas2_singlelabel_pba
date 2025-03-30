import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from models.classifiers import get_classifier, create_vectorizer, evaluate_model
from utils.visualization import plot_confusion_matrix

st.set_page_config(page_title="Model Training", layout="wide")
st.title("Model Training")

# Access data from session state
df = st.session_state.df

# Choose which label to train on
label_column = st.selectbox(
    "Choose label to train:",
    ["fuel", "machine", "part", "others", "price", "service"]
)

# Model selection
model_option = st.selectbox(
    "Select Model", ["Random Forest", "SVM", "Multinomial Naive Bayes"])

# Vectorization parameters
st.subheader("Text Vectorization Parameters")
max_features = st.slider(
    "Max Features", min_value=1000, max_value=10000, value=5000, step=1000)

# Training parameters
test_size = st.slider("Test Size", min_value=0.1,
                      max_value=0.5, value=0.2, step=0.05)

# Model-specific parameters
model_params = {}
if model_option == "Random Forest":
    model_params['n_estimators'] = st.slider(
        "Number of Trees", min_value=10, max_value=200, value=100, step=10)
elif model_option == "SVM":
    model_params['C'] = st.slider("Regularization Parameter (C)",
                                  min_value=0.01, max_value=10.0, value=1.0, step=0.01)
elif model_option == "Multinomial Naive Bayes":
    model_params['alpha'] = st.slider("Smoothing Parameter (alpha)",
                                      min_value=0.01, max_value=1.0, value=1.0, step=0.01)

# Processing
if st.button("Train Model"):
    st.info("Training in progress...")

    # Split data
    X = df['sentence']
    y = df[label_column]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)

    # Vectorize text
    vectorizer = create_vectorizer(max_features)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Get and train model
    model = get_classifier(model_option, **model_params)
    model.fit(X_train_tfidf, y_train)

    # Save model, vectorizer, and parameters to session state
    st.session_state.trained_model = model
    st.session_state.vectorizer = vectorizer
    st.session_state.model_name = model_option
    st.session_state.label_column = label_column

    # Evaluate model
    accuracy, report_df, comparison_df, y_pred = evaluate_model(
        model, X_test_tfidf, y_test)

    # Display results
    st.success("Training complete!")
    st.subheader("Model Performance")

    # Display accuracy
    st.write(f"Overall Accuracy: {accuracy:.4f}")

    # Side by side comparison
    st.subheader("Side-by-Side Comparison")
    comparison_df['Text'] = X_test.reset_index(drop=True)
    st.dataframe(comparison_df.head(10))

    # Confusion matrix
    st.subheader("Confusion Matrix")
    fig = plot_confusion_matrix(y_test, y_pred)
    st.pyplot(fig)

    # Classification report
    st.subheader("Classification Report")
    st.dataframe(report_df)
