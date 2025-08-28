import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import streamlit as st

# --- Config ---
max_features = 10000   # vocab size used during training
maxlen = 200           # sequence length used during training

# Load IMDB word index (maps words -> integers)
word_index = imdb.get_word_index()

# Load trained model
model = load_model("Simple_RNN_.h5")

# --- Prediction helper ---
def predict_sentiments(review):
    # Convert text -> word indices
    words = review.lower().split()
    seq = []
    for w in words:
        idx = word_index.get(w, 2)  # 2 = <OOV> token in IMDB
        if idx >= max_features:     # clamp to OOV if out of vocab
            idx = 2
        seq.append(idx)

    # Pad to same length as training
    seq = pad_sequences([seq], maxlen=maxlen)

    # Predict
    score = model.predict(seq, verbose=0)[0][0]
    sentiment = "Positive" if score >= 0.5 else "Negative"

    return sentiment, float(score)

# --- Streamlit UI ---
st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify whether it is **Positive** or **Negative**.")

user_input = st.text_area("Movie Review")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review first!")
    else:
        sentiment, score = predict_sentiments(user_input)
        st.success(f"**Sentiment:** {sentiment}")
        st.write(f"**Prediction Score:** {score:.4f}")
else:
    st.info("👆 Please enter a movie review and click *Classify*.")
