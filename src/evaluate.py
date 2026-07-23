import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
)

from src.config import (
    MODEL_COMPARISON,
    CLASSIFICATION_REPORT,
    BEST_MODEL_FILE,
    PREDICTIONS_FILE,
    GRAPH_DIR,
    CM_DIR,
)
from src.train import train_models
from src.utils import ensure_dir


def evaluate_models():
    ensure_dir("results", GRAPH_DIR, CM_DIR)

    models, vectorizer, X_test, y_test, training_times = train_models()

    results = []
    reports = []

    cm_files = {
        "Logistic Regression": "logistic_cm.png",
        "KNN": "knn_cm.png",
        "Random Forest": "random_forest_cm.png",
        "MLP": "mlp_cm.png",
    }

    for name, model in models.items():
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Training Time (s)": training_times[name],
        })

        reports.append(f"\n{name}\n")
        reports.append(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()

        plt.savefig(os.path.join(CM_DIR, cm_files[name]))
        plt.close()

    results_df = pd.DataFrame(results)

    results_df.to_csv(MODEL_COMPARISON, index=False)

    with open(CLASSIFICATION_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(reports))

    best_model = results_df.sort_values(
        "Accuracy",
        ascending=False
    ).iloc[0]

    with open(BEST_MODEL_FILE, "w", encoding="utf-8") as f:
        f.write(best_model["Model"])

    predictions = pd.DataFrame({
        "Actual": y_test,
        "Prediction": list(models[best_model["Model"]].predict(X_test)),
    })

    predictions.to_csv(PREDICTIONS_FILE, index=False)

    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

    for metric in metrics:
        plt.figure(figsize=(8, 5))
        plt.bar(results_df["Model"], results_df[metric])
        plt.ylabel(metric)
        plt.title(f"{metric} Comparison")
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(os.path.join(GRAPH_DIR, f"{metric.lower().replace(' ', '_')}.png"))
        plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(results_df["Model"], results_df["Training Time (s)"])
    plt.ylabel("Seconds")
    plt.title("Training Time Comparison")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, "training_time.png"))
    plt.close()

    return results_df