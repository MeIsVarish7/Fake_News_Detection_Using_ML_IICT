# AI-Powered Fake News Detection Using Text Classification

## Overview

This project is an AI-powered Fake News Detection system developed using Machine Learning and Natural Language Processing (NLP). It classifies news articles as **Real** or **Fake** based on their textual content. The project includes data preprocessing, TF-IDF feature extraction, multiple machine learning models, model evaluation, and a Flask web application for predictions.

---

## Features

- Text preprocessing using NLTK
- TF-IDF feature extraction
- Multiple Machine Learning models
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Random Forest
  - Multi-Layer Perceptron (MLP)
- Performance evaluation
- Confusion matrices
- Accuracy comparison graphs
- Classification reports
- Best model selection
- Flask web interface for prediction

---

## Project Structure

```
Fake-News-Detection/
│
├── data/
│   ├── raw/
│   │   └── train.csv
│   └── processed/
│       └── cleaned_news.csv
│
├── models/
│
├── results/
│   ├── confusion_matrices/
│   ├── graphs/
│   ├── best_model.txt
│   ├── classification_report.txt
│   ├── model_comparison.csv
│   └── predictions.csv
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── save_models.py
│   └── utils.py
│
├── static/
│   └── style.css
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── app.py
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib
- Seaborn
- Flask
- Joblib

---

## Machine Learning Pipeline

1. Load dataset
2. Clean and preprocess text
3. Remove stopwords
4. Lemmatize words
5. Convert text into TF-IDF vectors
6. Train multiple classifiers
7. Evaluate each model
8. Save trained models
9. Predict news using the best-performing model

---

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## Dataset

Place the dataset in:

```
data/raw/train.csv
```

After preprocessing, the cleaned dataset should be stored as:

```
data/processed/cleaned_news.csv
```

The processed dataset must contain the following columns:

- text
- label

where:

- 0 = Fake News
- 1 = Real News

---

## Run the Project

### Train and Evaluate Models

```bash
python main.py
```

This will:

- preprocess the dataset
- train all models
- evaluate performance
- generate graphs
- save confusion matrices
- save trained models

---

### Run the Web Application

```bash
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

Paste a news article and click **Predict**.

---

## Output

The project generates:

- Trained models
- Best model information
- Classification report
- Model comparison CSV
- Confusion matrices
- Performance graphs

---

## Future Improvements

- Deep Learning models (LSTM/BERT)
- Real-time news API integration
- Larger datasets
- Model deployment on cloud platforms
- Multilingual fake news detection

---

## Author

**IICT Summer Internship Project**

AI-Powered Fake News Detection Using Text Classification