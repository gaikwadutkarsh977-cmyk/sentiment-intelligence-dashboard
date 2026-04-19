import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob
import pdfplumber
from datetime import datetime
import pytz

st.set_page_config(page_title="Sentiment Intelligence Dashboard", layout="wide")

# -------------------------
# INDIA TIME GREETING
# -------------------------
india = pytz.timezone("Asia/Kolkata")
hour = datetime.now(india).hour
if hour < 12:
    greet = "Good Morning ☀"
elif hour < 17:
    greet = "Good Afternoon 🌤"
else:
    greet = "Good Evening 🌙"

# -------------------------
# UI STYLE
# -------------------------
st.markdown("""
<style>

.stApp{
background: linear-gradient(135deg,#4facfe,#43e97b);
}

.metric{
background:white;
padding:20px;
border-radius:12px;
text-align:center;
box-shadow:0px 3px 8px rgba(0,0,0,0.2);
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# HEADER
# -------------------------
st.title("📊 Sentiment Intelligence Dashboard")
st.write(greet + " — Let's start analyzing your data")

st.divider()

# -------------------------
# FILE UPLOAD
# -------------------------
file = st.file_uploader("Upload CSV or PDF file", type=["csv","pdf"])

if file:

    # -------------------------
    # READ CSV
    # -------------------------
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)

    # -------------------------
    # READ PDF
    # -------------------------
    if file.name.endswith(".pdf"):

        text_list = []

        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()

                if text:
                    lines = text.split("\n")
                    text_list.extend(lines)

        df = pd.DataFrame(text_list, columns=["review"])

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -------------------------
    # FIND SENTIMENT COLUMN
    # -------------------------
    sentiment_col = None

    for col in df.columns:
        if "sentiment" in col.lower():
            sentiment_col = col

    # -------------------------
    # FIND TEXT COLUMN
    # -------------------------
    text_col = None

    for col in df.columns:
        if "review" in col.lower() or "text" in col.lower() or "comment" in col.lower():
            text_col = col

    if text_col is None:
        text_col = st.selectbox("Select Review Column", df.columns)

    # -------------------------
    # SENTIMENT FUNCTION
    # -------------------------
    def get_sentiment(text):

        polarity = TextBlob(str(text)).sentiment.polarity

        if polarity > 0:
            return "Positive"
        elif polarity < 0:
            return "Negative"
        else:
            return "Neutral"

    # -------------------------
    # APPLY SENTIMENT
    # -------------------------
    if sentiment_col:
        df["Sentiment"] = df[sentiment_col]
    else:
        df["Sentiment"] = df[text_col].apply(get_sentiment)

    # -------------------------
    # METRICS
    # -------------------------
    total = len(df)
    positive = (df["Sentiment"]=="Positive").sum()
    negative = (df["Sentiment"]=="Negative").sum()
    neutral = (df["Sentiment"]=="Neutral").sum()

    st.divider()

    c1,c2,c3,c4 = st.columns(4)

    with c1:
        st.markdown(f'<div class="metric"><h3>Total Reviews</h3><h2>{total}</h2></div>',unsafe_allow_html=True)

    with c2:
        st.markdown(f'<div class="metric"><h3>Positive</h3><h2>{positive}</h2></div>',unsafe_allow_html=True)

    with c3:
        st.markdown(f'<div class="metric"><h3>Negative</h3><h2>{negative}</h2></div>',unsafe_allow_html=True)

    with c4:
        st.markdown(f'<div class="metric"><h3>Neutral</h3><h2>{neutral}</h2></div>',unsafe_allow_html=True)

    st.divider()

    # -------------------------
    # FILTER
    # -------------------------
    st.sidebar.title("Filter Reviews")

    option = st.sidebar.radio(
        "Choose Sentiment",
        ["All","Positive","Negative","Neutral"]
    )

    if option=="All":
        filtered = df
    else:
        filtered = df[df["Sentiment"]==option]

    # -------------------------
    # GRAPH
    # -------------------------
    st.subheader("Sentiment Distribution")

    fig = px.histogram(
        df,
        x="Sentiment",
        color="Sentiment",
        height=350
    )

    st.plotly_chart(fig,use_container_width=True)

    st.divider()

    # -------------------------
    # TABLE
    # -------------------------
    st.subheader("Filtered Reviews")

    st.dataframe(filtered[[text_col,"Sentiment"]])

    # -------------------------
    # DOWNLOAD
    # -------------------------
    csv = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Filtered CSV",
        csv,
        "sentiment_results.csv",
        "text/csv"
    )