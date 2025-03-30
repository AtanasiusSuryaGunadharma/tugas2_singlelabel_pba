import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st


def plot_label_distribution(df, label_column):
    """
    Create a bar plot showing the distribution of labels
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    df[label_column].value_counts().plot(kind='bar', ax=ax)
    plt.title(f"Distribution of {label_column}")
    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_test, y_pred):
    """
    Plot a confusion matrix
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = sorted(set(y_test))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    return fig


def plot_prediction_probabilities(model, input_tfidf):
    """
    Plot the prediction probabilities as a bar chart
    """
    if not hasattr(model, "predict_proba"):
        return None, None

    probabilities = model.predict_proba(input_tfidf)[0]
    class_labels = model.classes_
    prob_df = pd.DataFrame({
        'Class': class_labels,
        'Probability': probabilities
    })
    prob_df = prob_df.sort_values('Probability', ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(prob_df['Class'], prob_df['Probability'],
                  color=['green' if c == 'positive' else 'red' if c == 'negative' else 'gray' for c in prob_df['Class']])

    # Add percentage labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2%}',
                ha='center', va='bottom', rotation=0)

    plt.title("Prediction Probabilities")
    plt.ylim(0, 1.0)
    plt.ylabel("Probability")
    plt.tight_layout()

    return fig, prob_df
