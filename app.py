import streamlit as st
import pandas as pd
import re
import plotly.express as px
from textblob import TextBlob

st.set_page_config(
    page_title="Sentiment Intelligence Dashboard",
    layout="wide",
)

# -------------------------
# Clean Text
# -------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -------------------------
# Sentiment Function
# -------------------------
def get_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity

    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# -------------------------
# UI HEADER
# -------------------------
st.title("📊 Sentiment Intelligence Dashboard")
st.markdown("Analyze customer reviews and generate business insights instantly.")

st.divider()

# -------------------------
# File Upload
# -------------------------
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    column = st.selectbox("Select Review Column", df.columns)

    if st.button("Run Analysis"):

        df["Cleaned_Text"] = df[column].astype(str).apply(clean_text)
        df["Sentiment"] = df["Cleaned_Text"].apply(get_sentiment)

        st.success("Analysis Completed Successfully")

        # -------------------------
        # KPIs
        # -------------------------
        total = len(df)
        positive = len(df[df["Sentiment"] == "Positive"])
        negative = len(df[df["Sentiment"] == "Negative"])
        neutral = len(df[df["Sentiment"] == "Neutral"])

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Reviews", total)
        col2.metric("Positive", positive)
        col3.metric("Negative", negative)
        col4.metric("Neutral", neutral)

        st.divider()

        # -------------------------
        # Filter
        # -------------------------
        filter_option = st.radio(
            "Filter Reviews",
            ["All", "Positive", "Negative", "Neutral"],
            horizontal=True,
            key="filter_unique"
        )

        if filter_option != "All":
            filtered_df = df[df["Sentiment"] == filter_option]
        else:
            filtered_df = df

        # -------------------------
        # SMALL PROFESSIONAL CHART
        # -------------------------
        st.subheader("Sentiment Distribution")

        sentiment_counts = df["Sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ["Sentiment", "Count"]

        fig = px.bar(
            sentiment_counts,
            x="Sentiment",
            y="Count",
            height=350,
            text_auto=True
        )

        fig.update_layout(
            margin=dict(l=20, r=20, t=30, b=20)
        )

        st.plotly_chart(fig, use_container_width=False)

        # -------------------------
        # Show Filtered Reviews
        # -------------------------
        st.subheader("Filtered Reviews")
        st.dataframe(
            filtered_df[[column, "Sentiment"]],
            use_container_width=True
        )

        # -------------------------
        # Download Button
        # -------------------------
        csv = filtered_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Filtered CSV",
            csv,
            "filtered_reviews.csv",
            "text/csv"
        )