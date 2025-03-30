import streamlit as st
import pandas as pd
from utils.preprocessing import preprocess_text
from utils.visualization import plot_prediction_probabilities
from models.classifiers import get_classifier, create_vectorizer
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Prediction", layout="wide")
st.title("Make Predictions")

# Access data from session state
df = st.session_state.df

# Input text
user_input = st.text_area("Tulis text bebas:",
                          "Avanza bahan bakar nya boros banget")

# Check if we have trained model stored in session state
if st.session_state.trained_model is None:
    st.warning(
        "No trained model found. Please train a model in the 'Model Training' page first.")
    model_status = st.empty()
else:
    st.success(
        f"Using trained {st.session_state.model_name} model for {st.session_state.label_column} classification")

if st.button("Predict"):
    st.info("Predicting...")

    # Preprocess input
    preprocessed_input = preprocess_text(user_input)

    # Check if we have trained model stored
    if st.session_state.trained_model is not None:
        # Use the trained model from session state
        model = st.session_state.trained_model
        vectorizer = st.session_state.vectorizer
        label_column = st.session_state.label_column

        # Transform input using stored vectorizer
        input_tfidf = vectorizer.transform([preprocessed_input])

        # Make prediction
        prediction = model.predict(input_tfidf)[0]

        # Show the input text
        st.subheader("Input Text")
        st.write(user_input)

        st.subheader("Preprocessed Text")
        st.write(preprocessed_input)

        # Show the prediction with nice formatting
        st.subheader(f"Predicted {label_column.capitalize()} Sentiment")

        # Set color based on sentiment
        color = "gray"
        if prediction == "positive":
            color = "green"
        elif prediction == "negative":
            color = "red"

        st.markdown(
            f"<h3 style='color:{color};'>{prediction}</h3>", unsafe_allow_html=True)

        # Show probabilities if available
        fig, prob_df = plot_prediction_probabilities(model, input_tfidf)
        if fig is not None:
            st.subheader("Prediction Probabilities")
            st.pyplot(fig)

    else:
        # Train a default model if none is available
        model_status.info(
            "No trained model found. Training a default model Random Forest....")

        # Define label column
        label_column = "fuel"  # Default to fuel sentiment

        # Create training data
        y = df[label_column]

        # Split data
        X_train, _, y_train, _ = train_test_split(
            df['sentence'], y, test_size=0.2, random_state=42)

        # Vectorize properly
        vectorizer = create_vectorizer(max_features=5000)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        input_tfidf = vectorizer.transform([preprocessed_input])

        # Train model
        model = get_classifier("Random Forest", n_estimators=100)
        model.fit(X_train_tfidf, y_train)

        # Save to session state for future use
        st.session_state.trained_model = model
        st.session_state.vectorizer = vectorizer
        st.session_state.model_name = "Random Forest (default)"
        st.session_state.label_column = label_column

        # Make prediction
        prediction = model.predict(input_tfidf)[0]

        # Display results
        st.success("Default model trained and prediction complete!")

        # Show the input text
        st.subheader("Input Text")
        st.write(user_input)

        st.subheader("Preprocessed Text")
        st.write(preprocessed_input)

        # Show the prediction with nice formatting
        st.subheader(f"Predicted {label_column.capitalize()} Sentiment")

        # Set color based on sentiment
        color = "gray"
        if prediction == "positive":
            color = "green"
        elif prediction == "negative":
            color = "red"

        st.markdown(
            f"<h3 style='color:{color};'>{prediction}</h3>", unsafe_allow_html=True)

        # Show probabilities if available
        fig, prob_df = plot_prediction_probabilities(model, input_tfidf)
        if fig is not None:
            st.subheader("Prediction Probabilities")
            st.pyplot(fig)
