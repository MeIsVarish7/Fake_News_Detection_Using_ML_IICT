import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from src.config import (
    LOGISTIC_MODEL_PATH,
    KNN_MODEL_PATH,
    RANDOM_FOREST_MODEL_PATH,
    MLP_MODEL_PATH,
    VECTORIZER_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.data_loader import load_data
from src.feature_engineering import create_vectorizer, fit_transform, transform
from src.preprocess import preprocess_dataframe
from src.utils import ensure_dir, timer, elapsed


def train_models():
    data = load_data()
    data = preprocess_dataframe(data)

    X = data["text"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    vectorizer = create_vectorizer()

    X_train_vec = fit_transform(vectorizer, X_train)
    X_test_vec = transform(vectorizer, X_test)

    ensure_dir("models")

    joblib.dump(vectorizer, VECTORIZER_PATH)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=300,
            random_state=RANDOM_STATE,
        ),
    }

    model_paths = {
        "Logistic Regression": LOGISTIC_MODEL_PATH,
        "KNN": KNN_MODEL_PATH,
        "Random Forest": RANDOM_FOREST_MODEL_PATH,
        "MLP": MLP_MODEL_PATH,
    }

    training_times = {}

    for name, model in models.items():
        start = timer()
        model.fit(X_train_vec, y_train)
        training_times[name] = elapsed(start)

        joblib.dump(model, model_paths[name])

    return (
        models,
        vectorizer,
        X_test_vec,
        y_test,
        training_times,
    )