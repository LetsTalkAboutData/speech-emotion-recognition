import streamlit as st
import numpy as np
import joblib
from utils import record_audio, extract_mfcc

# Load model
model = joblib.load("model.pkl")
emotion_labels = {0: "angry", 1: "calm", 2: "happy", 3: "sad"}

st.set_page_config(page_title="Real-Time Emotion Detector", layout="centered")
st.title("🎤 Real-Time Speech Emotion Recognition")

st.write("Press the button below to record your voice and detect emotion in real time.")

if st.button("🎙️ Record and Detect"):
    with st.spinner("Recording..."):
        audio, sr = record_audio(duration=3)
    st.success("Recording Complete!")

    with st.spinner("Extracting Features..."):
        mfcc = extract_mfcc(audio, sr)

    with st.spinner("Predicting Emotion..."):
        prediction = model.predict(mfcc)[0]
        label = emotion_labels[int(prediction)]
        st.subheader(f"🧠 Detected Emotion: **{label}**")
