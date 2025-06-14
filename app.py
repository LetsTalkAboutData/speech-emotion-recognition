import streamlit as st
import numpy as np
import joblib
import librosa
import soundfile as sf
from utils import extract_mfcc

# Load model
model = joblib.load("model.pkl")
emotion_labels = {0: "angry", 1: "calm", 2: "happy", 3: "sad"}

st.set_page_config(page_title="Speech Emotion Detector", layout="centered")
st.title("🎤 Speech Emotion Recognition App")

st.write("Upload a `.wav` audio file to detect the emotion in the speech.")

uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    try:
        # Read and process audio
        audio_data, sample_rate = sf.read(uploaded_file)
        mfcc = extract_mfcc(audio_data, sample_rate)

        # Predict emotion
        prediction = model.predict(mfcc)[0]
        emotion = emotion_labels[int(prediction)]
        st.success(f"🧠 Detected Emotion: **{emotion}**")

    except Exception as e:
        st.error(f"Error processing the audio file: {str(e)}")
