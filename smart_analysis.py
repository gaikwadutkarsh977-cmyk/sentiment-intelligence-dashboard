import pandas as pd
import pickle
import re
import string
import sys

# ==============================
# Load trained model & vectorizer
# ==============================

try:
    model = pickle.load(open("sentiment_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    print("Model Loaded Successfully!\n")
except:
    print("Error: Model files not found.")
    sys.exit()

# ==============================
# Text Cleaning Function
# ==============================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    return text

# ==============================
# Load Dataset
# ==============================

file_name = input("Enter CSV file name (with .csv): ")

try:
    df = pd.read_csv(file_name)
except:
    print("Error: File not found.")
    sys.exit()

print("\nAvailable Columns:", list(df.columns))

column_name = input("Enter column name containing text: ")

if column_name not in df.columns:
    print("Column not found. Run again.")
    sys.exit()

# ==============================
# Prediction
# ==============================

df["cleaned_text"] = df[column_name].astype(str).apply(clean_text)
X = vectorizer.transform(df["cleaned_text"])
predictions = model.predict(X)

df["predicted_sentiment"] = predictions
df["predicted_sentiment"] = df["predicted_sentiment"].map({1: "Positive", 0: "Negative"})

# ==============================
# Business Summary
# ==============================

positive = (df["predicted_sentiment"] == "Positive").sum()
negative = (df["predicted_sentiment"] == "Negative").sum()

print("\n===== BUSINESS REPORT =====")
print("Total Reviews:", len(df))
print("Positive Reviews:", positive)
print("Negative Reviews:", negative)

# ==============================
# Professional Menu
# ==============================

while True:
    print("\nOptions:")
    print("1. Read All Reviews")
    print("2. Read Only Positive Reviews")
    print("3. Read Only Negative Reviews")
    print("4. Exit")

    choice = input("Enter your choice (1-4): ")

    if choice == "1":
        print("\nAll Reviews:\n")
        print(df[[column_name, "predicted_sentiment"]])

    elif choice == "2":
        print("\nPositive Reviews:\n")
        print(df[df["predicted_sentiment"] == "Positive"][[column_name, "predicted_sentiment"]])

    elif choice == "3":
        print("\nNegative Reviews:\n")
        print(df[df["predicted_sentiment"] == "Negative"][[column_name, "predicted_sentiment"]])

    elif choice == "4":
        print("Exiting... Thank You!")
        break

    else:
        print("Invalid choice. Try again.")

# Save file
df.to_csv("output_with_sentiment.csv", index=False)
print("\nFile saved as output_with_sentiment.csv")