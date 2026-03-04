import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import pytz

st.set_page_config(page_title="Sentiment Intelligence Dashboard", layout="wide")

# -------------------------
# FIXED TIMEZONE (India)
# -------------------------
india = pytz.timezone("Asia/Kolkata")
now = datetime.now(india)
current_hour = now.hour
current_time = now.strftime("%I:%M %p")

if current_hour < 12:
    greeting = "Good Morning"
elif current_hour < 17:
    greeting = "Good Afternoon"
else:
    greeting = "Good Evening"

# -------------------------
# PROFESSIONAL UI
# -------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #141E30, #243B55);
    color: white;
}

/* Sidebar White Clean */
section[data-testid="stSidebar"] {
    background: white;
}
section[data-testid="stSidebar"] * {
    color: black !important;
}

/* Fix Browse Button Text */
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] span,
button {
    color: black !important;
}

/* Download Button Fix */
.stDownloadButton button {
    background-color: #243B55;
    color: white !important;
    border-radius: 8px;
}

/* Metric Card */
.metric-card {
    background: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 15px;
    text-align:center;
    transition: 0.3s;
}
.metric-card:hover {
    transform: translateY(-6px);
    background: rgba(255,255,255,0.15);
}

/* Title */
.main-title {
    font-size: 38px;
    font-weight: 700;
}

/* Subtitle */
.sub-title {
    font-size: 15px;
    color: #cde7ff;
}

/* Logo */
.logo-circle {
    width:90px;
    height:90px;
    border-radius:50%;
    background: linear-gradient(135deg,#56ccf2,#2f80ed);
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:34px;
    font-weight:bold;
    color:white;
    box-shadow: 0 0 20px rgba(0,123,255,0.4);
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# HEADER
# -------------------------
col1, col2 = st.columns([1,5])

with col1:
    st.markdown('<div class="logo-circle">SI</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="main-title">Sentiment Intelligence Dashboard</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-title">{greeting} 👋 | {current_time}</div>', unsafe_allow_html=True)

st.divider()

# -------------------------
# FILE UPLOAD
# -------------------------
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    # Detect sentiment column
    sentiment_column = None
    for col in df.columns:
        if "sentiment" in col.lower():
            sentiment_column = col
            break

    if sentiment_column is None:
        st.error("No sentiment column found.")
        st.stop()

    df["Sentiment"] = df[sentiment_column].astype(str).str.strip().str.lower()

    # Detect text column
    text_column = None
    for col in df.columns:
        if "text" in col.lower():
            text_column = col
            break

    if text_column is None:
        text_column = st.selectbox("Select Review Column", df.columns)

    # -------------------------
    # METRICS
    # -------------------------
    total = len(df)
    positive = (df["Sentiment"] == "positive").sum()
    negative = (df["Sentiment"] == "negative").sum()
    neutral = (df["Sentiment"] == "neutral").sum()

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f'<div class="metric-card"><h4>Total Reviews</h4><h1>{total}</h1></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><h4>Positive</h4><h1>{positive}</h1></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><h4>Negative</h4><h1>{negative}</h1></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><h4>Neutral</h4><h1>{neutral}</h1></div>', unsafe_allow_html=True)

    st.divider()

    # -------------------------
    # SIDEBAR FILTER
    # -------------------------
    st.sidebar.header("Filter Reviews")

    filter_choice = st.sidebar.radio(
        "Select Sentiment",
        ["All", "Positive", "Negative", "Neutral"]
    )

    if filter_choice == "All":
        filtered_df = df
    else:
        filtered_df = df[df["Sentiment"] == filter_choice.lower()]

    # -------------------------
    # ANIMATED GRAPH
    # -------------------------
    st.subheader("Sentiment Distribution")

    summary_df = df["Sentiment"].value_counts().reset_index()
    summary_df.columns = ["Sentiment", "Count"]

    fig = px.bar(
        summary_df,
        x="Sentiment",
        y="Count",
        text_auto=True,
        color="Sentiment",
        height=350
    )

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        transition_duration=800
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # -------------------------
    # TABLE
    # -------------------------
    st.subheader("Filtered Reviews")

    st.dataframe(
        filtered_df[[text_column, "Sentiment"]],
        use_container_width=True
    )

    # -------------------------
    # DOWNLOAD BUTTON
    # -------------------------
    csv = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Filtered CSV",
        csv,
        "filtered_reviews.csv",
        "text/csv"
    )