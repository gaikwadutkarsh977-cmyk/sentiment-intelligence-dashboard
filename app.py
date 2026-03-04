import streamlit as st
import pandas as pd
import re
import joblib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Sentiment Intelligence Dashboard", layout="wide")

# ---------------------------
# Text Cleaning Function
# ---------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------------------------
# Train Model (First Run Only)
# ---------------------------
@st.cache_resource
def train_model():
    data = pd.read_csv("dataset.csv")  # your training dataset
    data["text"] = data["text"].apply(clean_text)

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(data["text"])
    y = data["sentiment"]

    model = LogisticRegression()
    model.fit(X, y)

    return model, vectorizer

model, vectorizer = train_model()

# ---------------------------
# UI HEADER
# ---------------------------
st.title("📊 Sentiment Intelligence Dashboard")
st.markdown("Upload review data and analyze customer sentiment instantly.")

st.divider()

# ---------------------------
# FILE UPLOAD
# ---------------------------
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    column = st.selectbox("Select Review Column", df.columns)

    if st.button("Run Analysis"):

        df["cleaned"] = df[column].astype(str).apply(clean_text)
        X_input = vectorizer.transform(df["cleaned"])
        df["Predicted_Sentiment"] = model.predict(X_input)

        st.success("Analysis Complete ✅")

        # ---------------------------
        # FILTER SECTION (FIXED)
        # ---------------------------
        st.subheader("Filter Reviews")

        filter_option = st.radio(
            "Select Sentiment",
            ["All", "Positive", "Negative"],
            horizontal=True,
            key="sentiment_filter_unique"
        )

        if filter_option == "Positive":
            filtered_df = df[df["Predicted_Sentiment"] == "positive"]
        elif filter_option == "Negative":
            filtered_df = df[df["Predicted_Sentiment"] == "negative"]
        else:
            filtered_df = df

        # ---------------------------
        # KPI SECTION
        # ---------------------------
        total = len(df)
        positive = len(df[df["Predicted_Sentiment"] == "positive"])
        negative = len(df[df["Predicted_Sentiment"] == "negative"])

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Reviews", total)
        col2.metric("Positive Reviews", positive)
        col3.metric("Negative Reviews", negative)

        st.divider()

        # ---------------------------
        # SMALL PROFESSIONAL BAR CHART
        # ---------------------------
        st.subheader("Sentiment Distribution")

        fig, ax = plt.subplots(figsize=(4,3))

        ax.bar(["Positive", "Negative"], [positive, negative])
        ax.set_ylabel("Count")
        ax.set_title("Review Sentiment")

        st.pyplot(fig)

        # ---------------------------
        # SHOW FILTERED REVIEWS
        # ---------------------------
        st.subheader("Filtered Reviews")
        st.dataframe(filtered_df[[column, "Predicted_Sentiment"]])

        # ---------------------------
        # DOWNLOAD OPTION
        # ---------------------------
        csv = filtered_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Filtered CSV",
            data=csv,
            file_name="filtered_reviews.csv",
            mime="text/csv"
        )