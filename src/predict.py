import joblib

from src.config import (
    BEST_MODEL_FILE,
    LOGISTIC_MODEL_PATH,
    KNN_MODEL_PATH,
    RANDOM_FOREST_MODEL_PATH,
    MLP_MODEL_PATH,
    VECTORIZER_PATH,
)
from src.preprocess import clean_text


MODEL_PATHS = {
    "Logistic Regression": LOGISTIC_MODEL_PATH,
    "KNN": KNN_MODEL_PATH,
    "Random Forest": RANDOM_FOREST_MODEL_PATH,
    "MLP": MLP_MODEL_PATH,
}


def load_best_model():
    with open(BEST_MODEL_FILE, "r", encoding="utf-8") as f:
        model_name = f.read().strip()

    if model_name not in MODEL_PATHS:
        raise ValueError(f"Unknown model: {model_name}")

    model = joblib.load(MODEL_PATHS[model_name])
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


def predict_news(text):
    model, vectorizer = load_best_model()

    cleaned_text = clean_text(text)
    vector = vectorizer.transform([cleaned_text])

    prediction = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]
    confidence = max(probabilities) * 100

    return {
        "label": "Real" if prediction == 1 else "Fake",
        "confidence": round(confidence, 2),
    }