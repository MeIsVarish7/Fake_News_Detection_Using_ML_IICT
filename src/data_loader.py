import pandas as pd

from src.config import PROCESSED_DATA_PATH


def load_data():
    data = pd.read_csv(PROCESSED_DATA_PATH)

    required_columns = ["title", "full_text", "label"]
    missing = [col for col in required_columns if col not in data.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data["title"] = data["title"].fillna("").astype(str).str.strip()
    data["full_text"] = data["full_text"].fillna("").astype(str).str.strip()

    data["text"] = (data["title"] + " " + data["full_text"]).str.strip()

    data = data[data["text"] != ""]

    data["label"] = pd.to_numeric(data["label"], errors="coerce")
    data = data.dropna(subset=["label"])
    data["label"] = data["label"].astype(int)

    return data[["text", "label"]].reset_index(drop=True)