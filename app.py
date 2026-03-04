import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="Sentiment Intelligence Dashboard", layout="wide")

# -------------------------
# Greeting
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
# LIGHT PROFESSIONAL GRADIENT
# -------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #1fa2ff, #12d8fa, #a6ffcb);
    color: white;
}

/* Fix upload box white issue */
[data-testid="stFileUploader"] {
    background-color: rgba(255,255,255,0.08);
    padding: 10px;
    border-radius: 10px;
}

/* Fix sidebar white issue */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
}

/* Remove white boxes */
.css-1d391kg, .css-1v0mbdj {
    background: transparent !important;
}

.main-title {
    font-size: 34px;
    font-weight: 700;
}

.sub-title {
    font-size: 14px;
    color: #eafdfc;
}

.logo-circle {
    width:85px;
    height:85px;
    border-radius:50%;
    background: linear-gradient(135deg,#12d8fa,#1fa2ff);
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:34px;
    font-weight:bold;
    color:white;
    box-shadow: 0 0 20px rgba(0,255,200,0.4);
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Header (SAME STRUCTURE)
# -------------------------
col1, col2 = st.columns([1,5])

with col1:
    st.markdown('<div class="logo-circle">SI</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="main-title">Sentiment Intelligence Dashboard</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-title">{greeting} 👋 | Current Time: {current_time}</div>', unsafe_allow_html=True)

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

    # Metrics
    total = len(df)
    positive = (df["Sentiment"] == "positive").sum()
    negative = (df["Sentiment"] == "negative").sum()
    neutral = (df["Sentiment"] == "neutral").sum()

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Reviews", total)
    col2.metric("Positive", positive)
    col3.metric("Negative", negative)
    col4.metric("Neutral", neutral)

    st.divider()

    # Sidebar Filter
    st.sidebar.header("Filter Reviews")

    filter_choice = st.sidebar.radio(
        "Select Sentiment",
        ["All", "Positive", "Negative", "Neutral"]
    )

    if filter_choice == "All":
        filtered_df = df
    else:
        filtered_df = df[df["Sentiment"] == filter_choice.lower()]

    # Chart
    st.subheader("Sentiment Distribution")

    summary_df = df["Sentiment"].value_counts().reset_index()
    summary_df.columns = ["Sentiment", "Count"]

    fig = px.bar(
        summary_df,
        x="Sentiment",
        y="Count",
        height=300,
        text_auto=True
    )

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        margin=dict(l=10, r=10, t=30, b=10)
    )

    st.plotly_chart(fig, use_container_width=False)

    st.divider()

    # Table
    st.subheader("Filtered Reviews")

    st.dataframe(
        filtered_df[[text_column, "Sentiment"]],
        use_container_width=True
    )