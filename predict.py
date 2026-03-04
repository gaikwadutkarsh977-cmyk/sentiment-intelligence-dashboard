import pickle
import re
import string

# Load saved model
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    return text

print("Model Loaded Successfully!")

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