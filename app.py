import streamlit as st
import pandas as pd
import pickle
import re
import plotly.express as px
from io import StringIO

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="Sentiment Intelligence Dashboard",
    layout="wide"
)

# ------------------------------------------------
# MINIMAL PROFESSIONAL CSS
# ------------------------------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.kpi-card {
    background-color: #f8f9fc;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0px 2px 6px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------
@st.cache_resource
def load_model():
    with open("sentiment_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

# ------------------------------------------------
# TEXT CLEANING
# ------------------------------------------------
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z ]", "", text)
    return text.lower()

# ------------------------------------------------
# HEADER WITH LOGO
# ------------------------------------------------
col_logo, col_title = st.columns([1,6])

with col_logo:
    st.image("https://cdn-icons-png.flaticon.com/512/9068/9068756.png", width=70)

with col_title:
    st.title("Sentiment Intelligence Dashboard")
    st.caption("Enterprise Review Sentiment Analytics Platform")

st.divider()

# ------------------------------------------------
# FILE UPLOAD
# ------------------------------------------------
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    try:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        df = pd.read_csv(stringio)
    except:
        st.error("Please upload a valid UTF-8 CSV file.")
        st.stop()

    st.success("File Loaded Successfully")

    column_name = st.selectbox("Select Text Column", df.columns)

    if st.button("Run Sentiment Analysis", type="primary"):

        df[column_name] = df[column_name].astype(str).apply(clean_text)

        vectors = vectorizer.transform(df[column_name])
        predictions = model.predict(vectors)

        df["Sentiment"] = predictions

        # Handle numeric output (0/1)
        if set(df["Sentiment"].unique()).issubset({0,1}):
            df["Sentiment"] = df["Sentiment"].map({1: "Positive", 0: "Negative"})

        positive = int((df["Sentiment"] == "Positive").sum())
        negative = int((df["Sentiment"] == "Negative").sum())
        total = int(len(df))

        positive_pct = round((positive/total)*100, 1) if total > 0 else 0
        negative_pct = round((negative/total)*100, 1) if total > 0 else 0

        # ---------------- KPI SECTION ----------------
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Reviews", total)

        with col2:
            st.metric("Positive Reviews", f"{positive} ({positive_pct}%)")

        with col3:
            st.metric("Negative Reviews", f"{negative} ({negative_pct}%)")

        st.divider()

        # ---------------- CHART TOGGLE ----------------
        chart_type = st.radio(
            "Select Chart Type",
            ["Bar Chart", "Donut Chart"],
            horizontal=True
        )

        chart_data = pd.DataFrame({
            "Sentiment": ["Positive", "Negative"],
            "Count": [positive, negative]
        })

        st.subheader("Sentiment Distribution")

        if total > 0:

            if chart_type == "Bar Chart":

                fig = px.bar(
                    chart_data,
                    x="Sentiment",
                    y="Count",
                    text="Count",
                    height=320
                )

                fig.update_layout(
                    showlegend=False,
                    transition_duration=500,
                    margin=dict(l=40, r=40, t=40, b=40)
                )

            else:

                fig = px.pie(
                    chart_data,
                    names="Sentiment",
                    values="Count",
                    hole=0.6,
                    height=320
                )

                fig.update_layout(
                    transition_duration=500,
                    margin=dict(l=40, r=40, t=40, b=40)
                )

            col_c1, col_c2, col_c3 = st.columns([1,2,1])
            with col_c2:
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("No predictions to display.")

        st.divider()

        # ---------------- FILTER SECTION ----------------
        st.subheader("Review Explorer")

        filter_option = st.selectbox(
            "Filter by Sentiment",
            ["All", "Positive", "Negative"]
        )

        if filter_option != "All":
            filtered_df = df[df["Sentiment"] == filter_option]
        else:
            filtered_df = df

        st.dataframe(filtered_df.head(100), use_container_width=True)

        # ---------------- DOWNLOAD ----------------
        csv = filtered_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Results CSV",
            data=csv,
            file_name="sentiment_results.csv",
            mime="text/csv"
        )