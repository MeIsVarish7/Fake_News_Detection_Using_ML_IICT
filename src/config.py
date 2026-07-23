import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "train.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_news.csv")

MODEL_DIR = os.path.join(BASE_DIR, "models")

LOGISTIC_MODEL_PATH = os.path.join(MODEL_DIR, "logistic.pkl")
KNN_MODEL_PATH = os.path.join(MODEL_DIR, "knn.pkl")
RANDOM_FOREST_MODEL_PATH = os.path.join(MODEL_DIR, "random_forest.pkl")
MLP_MODEL_PATH = os.path.join(MODEL_DIR, "mlp.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")

RESULT_DIR = os.path.join(BASE_DIR, "results")

MODEL_COMPARISON = os.path.join(RESULT_DIR, "model_comparison.csv")
CLASSIFICATION_REPORT = os.path.join(RESULT_DIR, "classification_report.txt")
BEST_MODEL_FILE = os.path.join(RESULT_DIR, "best_model.txt")
PREDICTIONS_FILE = os.path.join(RESULT_DIR, "predictions.csv")

GRAPH_DIR = os.path.join(RESULT_DIR, "graphs")

ACCURACY_GRAPH = os.path.join(GRAPH_DIR, "accuracy.png")
PRECISION_GRAPH = os.path.join(GRAPH_DIR, "precision.png")
RECALL_GRAPH = os.path.join(GRAPH_DIR, "recall.png")
F1_GRAPH = os.path.join(GRAPH_DIR, "f1_score.png")
TRAINING_TIME_GRAPH = os.path.join(GRAPH_DIR, "training_time.png")
MODEL_COMPARISON_GRAPH = os.path.join(GRAPH_DIR, "model_comparison.png")

CM_DIR = os.path.join(RESULT_DIR, "confusion_matrices")

LOGISTIC_CM = os.path.join(CM_DIR, "logistic_cm.png")
KNN_CM = os.path.join(CM_DIR, "knn_cm.png")
RANDOM_FOREST_CM = os.path.join(CM_DIR, "random_forest_cm.png")
MLP_CM = os.path.join(CM_DIR, "mlp_cm.png")

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_FEATURES = 5000