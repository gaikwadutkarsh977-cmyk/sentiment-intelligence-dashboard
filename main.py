import pandas as pd
import numpy as np
import re
import string
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Loading dataset...")

# Load Dataset
df = pd.read_csv("dataset.csv", encoding='latin-1', header=None)

# Rename columns
df.columns = ["target", "id", "date", "flag", "user", "tweet"]

# Keep only required columns
df = df[["target", "tweet"]]

# Convert 4 → 1 (Positive)
df["target"] = df["target"].replace(4, 1)

print("Dataset Loaded Successfully!")
print("Total Rows:", len(df))

# OPTIONAL: reduce size if system slow
df = df.sample(10000, random_state=42)

print("Using Sample Size:", len(df))

# Text Cleaning Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    return text

print("Cleaning tweets...")
df["tweet"] = df["tweet"].apply(clean_text)

# Features & Labels
X = df["tweet"]
y = df["target"]

print("Converting text to numbers (TF-IDF)...")
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X)

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Model Training Completed!")

# Evaluation
y_pred = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Save Model & Vectorizer
pickle.dump(model, open("sentiment_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\nModel Saved Successfully!")

# Live Prediction
while True:
    user_input = input("\nEnter tweet (type 'exit' to quit): ")

    if user_input.lower() == "exit":
        break

    cleaned = clean_text(user_input)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)

    if prediction[0] == 1:
        print("Sentiment: Positive 😊")
    else:
        print("Sentiment: Negative 😡")