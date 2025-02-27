import streamlit as st
import numpy as np
import librosa
import librosa.display
import io
import pandas as pd
import matplotlib.pyplot as plt
import random

# Set Streamlit layout to wide screen
st.set_page_config(layout="wide")

# Load built-in librosa example tracks for a larger dataset
@st.cache_data
def load_sample_data():
    dataset = []
    example_tracks = [
        "trumpet", "brahms", "nutcracker", "choice", "vibeace"
    ]  # Only valid librosa example tracks
    for track in example_tracks:
        try:
            y, sr = librosa.load(librosa.ex(track), sr=22050)
            mfcc_features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1).tolist()
            dataset.append({"Title": track.capitalize(), "Artist": "Unknown", "Genre": "Instrumental", "MFCC_Features": mfcc_features})
        except Exception as e:
            print(f"Error loading track {track}: {e}")
    return dataset

dataset = load_sample_data()

def extract_mfcc(audio_bytes, sr=22050, n_mfcc=13):
    y, _ = librosa.load(io.BytesIO(audio_bytes), sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc

def find_closest_match(query_mfcc, dataset, threshold):
    query_mfcc_mean = np.mean(query_mfcc, axis=1)
    for song in dataset:
        distance = np.linalg.norm(np.array(song["MFCC_Features"]) - query_mfcc_mean)
        if distance <= threshold:
            return song
    return None

# Streamlit UI
st.title("🎵 Tuneify: Music Recognition & Recommendation")
st.markdown("""
### 🔍 How to Use Tuneify:
1. **Upload an audio file** (MP3 or WAV format, max 50MB).
2. **View the MFCC visualization** of the uploaded track.
3. **Adjust the Euclidean distance slider** for better recommendations.
4. **Click 'Find Recommendation'** to check if a match is found.
""")

uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])

distance_threshold = st.slider("Set Euclidean Distance Threshold", min_value=0.0, max_value=100.0, value=20.0, step=0.5)

if uploaded_file:
    if uploaded_file.size > 50 * 1024 * 1024:
        st.error("❌ File too large! Please upload a file smaller than 50MB.")
    else:
        st.success("✅ File uploaded successfully!")
        st.audio(uploaded_file, format="audio/wav")
        audio_bytes = uploaded_file.read()
        query_mfcc = extract_mfcc(audio_bytes)
        
        # Display MFCC
        st.subheader("📊 MFCC Features")
        fig, ax = plt.subplots()
        librosa.display.specshow(query_mfcc, x_axis='time')
        ax.set_title("MFCC Features of Uploaded Audio")
        st.pyplot(fig)
        
        if st.button("Find Recommendation"):
            matched_song = find_closest_match(query_mfcc, dataset, distance_threshold)
            
            if matched_song is not None:
                st.subheader("🎶 Identified Song")
                st.write(f"**Title:** {matched_song['Title']}")
                st.write(f"**Artist:** {matched_song['Artist']}")
                st.write(f"**Genre:** {matched_song['Genre']}")
            else:
                st.write("❌ No recommendations found. Try adjusting the threshold.")
