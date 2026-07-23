from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import MAX_FEATURES


def create_vectorizer():
    return TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=(1, 2)
    )


def fit_transform(vectorizer, text):
    return vectorizer.fit_transform(text)


def transform(vectorizer, text):
    return vectorizer.transform(text)