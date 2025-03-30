import streamlit as st
import pandas as pd
from utils.visualization import plot_label_distribution

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
