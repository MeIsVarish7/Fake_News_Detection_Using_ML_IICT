import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FAKE_PATH = os.path.join(BASE_DIR, "data", "raw", "Fake.csv")
TRUE_PATH = os.path.join(BASE_DIR, "data", "raw", "True.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_news.csv")


def clean_dataset():
    print("Loading datasets...")

    fake_df = pd.read_csv(FAKE_PATH)
    true_df = pd.read_csv(TRUE_PATH)

    print(f"Fake news articles : {len(fake_df)}")
    print(f"True news articles : {len(true_df)}")

    fake_df["label"] = 0
    true_df["label"] = 1

    fake_df.rename(columns={"text": "full_text"}, inplace=True)
    true_df.rename(columns={"text": "full_text"}, inplace=True)

    required_cols = ["title", "full_text", "label"]

    fake_df = fake_df[required_cols]
    true_df = true_df[required_cols]

    df = pd.concat([fake_df, true_df], ignore_index=True)

    df.dropna(inplace=True)

    df["title"] = (
        df["title"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    df["full_text"] = (
        df["full_text"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    df.drop_duplicates(inplace=True)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print("\nDataset created successfully.")
    print(f"Total rows : {len(df)}")
    print(f"Unique titles : {df['title'].nunique()}")
    print(f"Unique articles : {df['full_text'].nunique()}")
    print(f"Saved to : {OUTPUT_PATH}")


if __name__ == "__main__":
    clean_dataset()