from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd


def get_classifier(model_name, **params):
    """
    Return a classifier instance based on model name and parameters
    """
    if model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=params.get('n_estimators', 100),
            random_state=42
        )
    elif model_name == "SVM":
        return SVC(
            C=params.get('C', 1.0),
            probability=True,
            random_state=42
        )
    elif model_name == "Multinomial Naive Bayes":
        return MultinomialNB(
            alpha=params.get('alpha', 1.0)
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def create_vectorizer(max_features=5000):
    """
    Create a TF-IDF vectorizer
    """
    return TfidfVectorizer(max_features=max_features)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a model and return performance metrics
    """
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    # Use zero_division=0 to avoid warnings
    report = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()

    # Create comparison DataFrame
    comparison_df = pd.DataFrame()
    comparison_df['Actual'] = y_test.reset_index(drop=True)
    comparison_df['Predicted'] = y_pred
    comparison_df['Match'] = comparison_df['Actual'] == comparison_df['Predicted']

    return accuracy, report_df, comparison_df, y_pred
