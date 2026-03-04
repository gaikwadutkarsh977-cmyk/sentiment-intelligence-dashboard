import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import time

st.set_page_config(page_title="Sentiment Intelligence Dashboard", layout="wide")

# -------------------------
# Dynamic Greeting
# -------------------------
current_hour = datetime.now().hour

if current_hour < 12:
    greeting = "Good Morning"
elif current_hour < 17:
    greeting = "Good Afternoon"
else:
    greeting = "Good Evening"

current_time = datetime.now().strftime("%I:%M %p")

# -------------------------
# PROFESSIONAL CORPORATE THEME
# -------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

/* Sidebar White */
section[data-testid="stSidebar"] {
    background: white;
    color: black;
}
section[data-testid="stSidebar"] * {
    color: black !important;
}

/* Animated Metric Card */
.metric-card {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 15px;
    text-align:center;
    transition: 0.3s;
}
.metric-card:hover {
    transform: translateY(-5px);
    background: rgba(255,255,255,0.15);
}

/* Title */
.main-title {
    font-size: 36px;
    font-weight: 700;
}

/* Sub */
.sub-title {
    font-size: 14px;
    color: #dcefff;
}

/* Logo */
.logo-circle {
    width:85px;
    height:85px;
    border-radius:50%;
    background: linear-gradient(135deg,#1c92d2,#f2fcfe);
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:32px;
    font-weight:bold;
    color:#0f2027;
    box-shadow: 0 0 20px rgba(255,255,255,0.3);
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Header
# -------------------------
col1, col2 = st.columns([1,5])

with col1:
    st.markdown('<div class="logo-circle">SI</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="main-title">Sentiment Intelligence Dashboard</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-title">{greeting} 👋 | {current_time}</div>', unsafe_allow_html=True)

st.divider()

# -------------------------
# Upload Section
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
    # METRICS WITH ANIMATION
    # -------------------------
    total = len(df)
    positive = (df["Sentiment"] == "positive").sum()
    negative = (df["Sentiment"] == "negative").sum()
    neutral = (df["Sentiment"] == "neutral").sum()

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f'<div class="metric-card"><h3>Total Reviews</h3><h1>{total}</h1></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><h3>Positive</h3><h1>{positive}</h1></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><h3>Negative</h3><h1>{negative}</h1></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><h3>Neutral</h3><h1>{neutral}</h1></div>', unsafe_allow_html=True)

    st.divider()

    # -------------------------
    # FILTER (WHITE SIDEBAR)
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
        height=350,
        color="Sentiment"
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
    # FILTERED TABLE
    # -------------------------
    st.subheader("Filtered Reviews")

    st.dataframe(
        filtered_df[[text_column, "Sentiment"]],
        use_container_width=True
    )

    # -------------------------
    # DOWNLOAD CSV (ADDED BACK)
    # -------------------------
    csv = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Filtered CSV",
        csv,
        "filtered_reviews.csv",
        "text/csv"
    )