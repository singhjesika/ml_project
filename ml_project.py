import os
import re
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# NLP & sklearn
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# UI
import streamlit as st

# Ensure NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Paths for model files
MODEL_PATH = Path('model.joblib')
VECT_PATH = Path('vectorizer.joblib')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess_text(text: str) -> str:
    """Clean, tokenize, remove stopwords and lemmatize."""
    if not isinstance(text, str):
        return ''
    # basic cleaning
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)  # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)  # keep letters and spaces
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)


def load_dataset(csv_path='reviews.csv') -> pd.DataFrame:
    """Load dataset and run preprocessing. Expect columns 'review' and 'label'."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset '{csv_path}' not found. Put a CSV with columns 'review' and 'label' in the folder.")
    df = pd.read_csv(csv_path)
    if 'review' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain 'review' and 'label' columns.")
    df = df[['review', 'label']].dropna().reset_index(drop=True)
    df['cleaned'] = df['review'].astype(str).apply(preprocess_text)
    return df


def train_and_save(df: pd.DataFrame, vect_path=VECT_PATH, model_path=MODEL_PATH):
    """Train TF-IDF + MultinomialNB and persist vectorizer+model."""
    st.info('Training model — this may take a few seconds depending on dataset size.')
    X = df['cleaned'].values
    y = df['label'].astype(int).values

    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42, stratify=y)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    # Save
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vect_path)

    return model, vectorizer, acc, report, cm


@st.cache_data
def load_model_and_vectorizer(model_path=MODEL_PATH, vect_path=VECT_PATH):
    """Load saved model and vectorizer if they exist, else return (None,None)."""
    if model_path.exists() and vect_path.exists():
        model = joblib.load(model_path)
        vectorizer = joblib.load(vect_path)
        return model, vectorizer
    return None, None


# ------------------- Streamlit UI -------------------

def app():
    st.set_page_config(page_title='Fake Product Review Monitor', layout='centered')
    st.title('🔍 Fake Product Review Monitoring System')

    st.markdown(
        """
        This app trains a simple TF-IDF + Naive Bayes model to detect fake reviews.
        Upload a `reviews.csv` (columns `review`, `label`) or place it in the same folder as this script.
        """
    )

    # show model status
    model, vectorizer = load_model_and_vectorizer()

    col1, col2 = st.columns(2)
    with col1:
        if model is not None:
            st.success('Model and vectorizer loaded from disk.')
        else:
            st.warning('No saved model found. You will need to train one.')

    with col2:
        if st.button('Train / Retrain model from reviews.csv'):
            try:
                df = load_dataset('reviews.csv')
            except Exception as e:
                st.exception(e)
                st.stop()

            with st.spinner('Training...'):
                model, vectorizer, acc, report, cm = train_and_save(df)

            st.success(f'Training finished — test accuracy: {acc:.4f}')
            st.text('Classification report:')
            st.text(report)
            st.text('Confusion matrix:')
            st.write(cm.tolist())

    st.markdown('---')

    # If model missing, allow upload and train
    if model is None:
        st.info('If you have a CSV file, upload it now to train the model without restarting.')
        uploaded = st.file_uploader('Upload reviews.csv', type=['csv'])
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                df = df[['review','label']].dropna().reset_index(drop=True)
                df['cleaned'] = df['review'].astype(str).apply(preprocess_text)
            except Exception as e:
                st.exception(e)
                st.stop()

            if st.button('Train on uploaded CSV'):
                with st.spinner('Training...'):
                    model, vectorizer, acc, report, cm = train_and_save(df)
                st.success(f'Training finished — test accuracy: {acc:.4f}')
                st.text(report)

    st.markdown('---')
    st.header('Try the model')

    review_input = st.text_area('Enter a product review to classify', height=150)
    if st.button('Check Review'):
        if not review_input.strip():
            st.warning('Please enter some review text.')
        else:
            # ensure model exists (maybe just trained)
            model, vectorizer = load_model_and_vectorizer()
            if model is None or vectorizer is None:
                st.error('No trained model available. Please train the model first (use the button above).')
            else:
                cleaned = preprocess_text(review_input)
                vec = vectorizer.transform([cleaned])
                pred = model.predict(vec)[0]
                probs = model.predict_proba(vec) if hasattr(model, 'predict_proba') else None

                if pred == 1:
                    st.error('Prediction: Fake Review 🚨')
                else:
                    st.success('Prediction: Genuine Review ✅')

                if probs is not None:
                    st.write(f'Probabilities: {probs.tolist()[0]}')

    st.markdown('---')
    st.header('Model & Dataset Quick Info')
    if Path('reviews.csv').exists():
        try:
            df_local = pd.read_csv('reviews.csv')
            st.write('Local reviews.csv found — sample:')
            st.dataframe(df_local.head(5))
            st.write(f'Total rows: {len(df_local)}')
        except Exception:
            st.write('Found reviews.csv but failed to read it.')
    else:
        st.info('No reviews.csv in the folder. You can upload a CSV to train the model.')

    st.markdown('---')
    st.caption('Tip: For improved accuracy, use balanced labeled data, add reviewer metadata features, or use transformer models (BERT).')


if __name__ == '__main__':
    app()
