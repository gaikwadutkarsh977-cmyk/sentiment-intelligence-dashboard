import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob
from datetime import datetime
import pytz

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Sentiment Dashboard", layout="wide")

# -------------------------
# TIME (FIXED ISSUE)
# -------------------------
tz = pytz.timezone("Asia/Kolkata")
current_time = datetime.now(tz)
hour = current_time.hour

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

.stApp {
    background: linear-gradient(135deg, #1f4037, #99f2c8);
}

.metric-box {
    background: white;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
    transition: 0.3s;
}

.metric-box:hover {
    transform: scale(1.05);
}

h1, h2, h3 {
    color: white;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# HEADER
# -------------------------
st.title("📊 Sentiment Intelligence Dashboard")
st.write(f"{greet} 👋 | Time: {current_time.strftime('%I:%M %p')}")

st.divider()

# -------------------------
# FILE UPLOAD
# -------------------------
file = st.file_uploader("Upload CSV File", type=["csv"])

if file:

    df = pd.read_csv(file)

    st.subheader("📂 Dataset Preview")
    st.dataframe(df.head())

    # -------------------------
    # DETECT TEXT COLUMN
    # -------------------------
    text_col = None

    for col in df.columns:
        if "review" in col.lower() or "text" in col.lower() or "comment" in col.lower():
            text_col = col

    if text_col is None:
        text_col = st.selectbox("Select Text Column", df.columns)

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
        st.markdown(f'<div class="metric-box"><h3>Total</h3><h2>{total}</h2></div>', unsafe_allow_html=True)

    with c2:
        st.markdown(f'<div class="metric-box"><h3>Positive</h3><h2>{positive}</h2></div>', unsafe_allow_html=True)

    with c3:
        st.markdown(f'<div class="metric-box"><h3>Negative</h3><h2>{negative}</h2></div>', unsafe_allow_html=True)

    with c4:
        st.markdown(f'<div class="metric-box"><h3>Neutral</h3><h2>{neutral}</h2></div>', unsafe_allow_html=True)

    st.divider()

    # -------------------------
    # FILTER
    # -------------------------
    st.sidebar.title("🔎 Filter")

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
    st.subheader("📊 Sentiment Distribution")

    fig = px.histogram(
        df,
        x="Sentiment",
        color="Sentiment",
        height=350
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # -------------------------
    # TABLE
    # -------------------------
    st.subheader("📋 Filtered Reviews")
    st.dataframe(filtered[[text_col, "Sentiment"]])

    # -------------------------
    # DOWNLOAD
    # -------------------------
    csv = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇ Download Filtered Data",
        csv,
        "sentiment_results.csv",
        "text/csv"
    )

else:
    st.info("👆 Please upload a CSV file to start analysis")