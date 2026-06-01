import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.title {
    text-align:center;
    color:#1E3A8A;
    font-size:40px;
    font-weight:bold;
}

.subtitle {
    text-align:center;
    color:#6B7280;
    font-size:20px;
    margin-bottom:30px;
}

.stButton>button {
    width:100%;
    background-color:#2563EB;
    color:white;
    font-size:18px;
    border-radius:10px;
    height:50px;
}

.result-box {
    padding:20px;
    border-radius:15px;
    background:#E0F2FE;
    border:2px solid #38BDF8;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="title">🎬 Movie Review Sentiment Analysis System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Deep Learning Based Sentiment Classification</div>',
    unsafe_allow_html=True
)

# ==========================================
# LOAD MODELS
# ==========================================

@st.cache_resource
def load_models():
    try:
        rnn = tf.keras.models.load_model("simple_rnn_model.keras")
        lstm = tf.keras.models.load_model("lstm_model.keras")
        gru = tf.keras.models.load_model("gru_model.keras")

        return rnn, lstm, gru

    except Exception as e:
        st.error(str(e))
        raise e

# ==========================================
# WORD INDEX
# ==========================================

word_index = imdb.get_word_index()

MAX_LEN = 200
VOCAB_SIZE = 10000

# ==========================================
# PREPROCESS FUNCTION
# ==========================================

def preprocess_text(text):

    text = text.lower()

    text = re.sub(
        r'[^a-zA-Z\s]',
        '',
        text
    )

    sequence = []

    for word in text.split():

        if word in word_index:

            idx = word_index[word] + 3

            if idx < VOCAB_SIZE:
                sequence.append(idx)

    padded = pad_sequences(
        [sequence],
        maxlen=MAX_LEN,
        padding='post'
    )

    return padded

# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_review(review, model):

    processed = preprocess_text(review)

    probability = model.predict(
        processed,
        verbose=0
    )[0][0]

    sentiment = (
        "Positive"
        if probability >= 0.5
        else "Negative"
    )

    confidence = (
        probability
        if probability >= 0.5
        else 1 - probability
    )

    return sentiment, confidence, probability

# ==========================================
# INPUT AREA
# ==========================================

review = st.text_area(
    "Enter your movie review here...",
    height=180
)

selected_model = st.radio(
    "Select Model",
    [
        "SimpleRNN",
        "LSTM",
        "GRU"
    ]
)

# ==========================================
# PREDICT BUTTON
# ==========================================

if st.button("Analyze Review"):

    if review.strip() == "":

        st.warning("Please enter a review.")

    else:

        if selected_model == "SimpleRNN":
            model = model_rnn

        elif selected_model == "LSTM":
            model = model_lstm

        else:
            model = model_gru

        sentiment, confidence, prob = predict_review(
            review,
            model
        )

        st.markdown(
            "<div class='result-box'>",
            unsafe_allow_html=True
        )

        st.success(
            f"Sentiment: {sentiment}"
        )

        st.info(
            f"Confidence: {confidence*100:.2f}%"
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

        # ======================================
        # PROBABILITIES
        # ======================================

        positive_prob = prob
        negative_prob = 1 - prob

        st.subheader(
            "Prediction Probabilities"
        )

        prob_df = pd.DataFrame({

            "Class": [
                "Positive",
                "Negative"
            ],

            "Probability": [
                positive_prob,
                negative_prob
            ]

        })

        st.bar_chart(
            prob_df.set_index("Class")
        )

        # ======================================
        # CONFIDENCE CHART
        # ======================================

        st.subheader(
            "Confidence Chart"
        )

        fig, ax = plt.subplots(
            figsize=(6,4)
        )

        ax.bar(
            ["Confidence"],
            [confidence*100]
        )

        ax.set_ylim(0,100)

        ax.set_ylabel(
            "Confidence (%)"
        )

        st.pyplot(fig)

        # ======================================
        # COMPARE ALL MODELS
        # ======================================

        st.subheader(
            "Model Comparison"
        )

        models = {

            "SimpleRNN": model_rnn,
            "LSTM": model_lstm,
            "GRU": model_gru

        }

        results = []

        for name, mdl in models.items():

            sent, conf, _ = predict_review(
                review,
                mdl
            )

            results.append([

                name,
                sent,
                round(conf*100,2)

            ])

        comparison_df = pd.DataFrame(

            results,

            columns=[
                "Model",
                "Sentiment",
                "Confidence (%)"
            ]

        )

        st.dataframe(
            comparison_df,
            use_container_width=True
        )
