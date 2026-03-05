import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob
from datetime import datetime
import pdfplumber

st.set_page_config(page_title="Sentiment Intelligence Dashboard", layout="wide")

# -------------------------
# Greeting system
# -------------------------

hour = datetime.now().hour

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
background: linear-gradient(120deg,#e8f2ff,#e6fff4);
}

.header{
font-size:38px;
font-weight:700;
color:#222;
}

.metric-card{
background:white;
padding:25px;
border-radius:14px;
text-align:center;
box-shadow:0 4px 15px rgba(0,0,0,0.1);
transition:0.3s;
}

.metric-card:hover{
transform:scale(1.04);
}

.metric-title{
font-size:18px;
color:#666;
}

.metric-value{
font-size:32px;
font-weight:bold;
color:#111;
}

.filter-box{
background:white;
padding:15px;
border-radius:12px;
box-shadow:0 3px 10px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# HEADER
# -------------------------

st.markdown('<div class="header">📊 Sentiment Intelligence Dashboard</div>', unsafe_allow_html=True)

st.write(f"**{greet} — Let's start analyzing customer feedback**")

st.divider()

# -------------------------
# FILE UPLOAD
# -------------------------

file = st.file_uploader("Upload CSV or PDF File", type=["csv","pdf"])

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
    positive = (df["Sentiment"] == "Positive").sum()
    negative = (df["Sentiment"] == "Negative").sum()
    neutral = (df["Sentiment"] == "Neutral").sum()

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
        <div class="metric-title">Total Reviews</div>
        <div class="metric-value">{total}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
        <div class="metric-title">Positive</div>
        <div class="metric-value">{positive}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
        <div class="metric-title">Negative</div>
        <div class="metric-value">{negative}</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card">
        <div class="metric-title">Neutral</div>
        <div class="metric-value">{neutral}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # -------------------------
    # FILTER
    # -------------------------

    st.sidebar.markdown("### Filter Reviews")

    option = st.sidebar.radio(
        "Choose Sentiment",
        ["All", "Positive", "Negative", "Neutral"]
    )

    if option == "All":
        filtered = df

    else:
        filtered = df[df["Sentiment"] == option]

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

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # -------------------------
    # TABLE
    # -------------------------

    st.subheader("Filtered Reviews")

    st.dataframe(filtered[[text_col, "Sentiment"]])

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