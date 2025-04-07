import streamlit as st
import numpy as np
import librosa
import librosa.display
import io
import pandas as pd
import matplotlib.pyplot as plt
import requests
from pydub import AudioSegment
import random

# Set Streamlit layout to wide screen
st.set_page_config(layout="wide")

# Load built-in librosa example tracks for a larger dataset
@st.cache_data
def load_sample_data():
    dataset = []
    example_tracks = ["trumpet", "brahms", "nutcracker", "choice", "vibeace"]
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
    best_match = None
    min_distance = float('inf')
    for song in dataset:
        distance = np.linalg.norm(np.array(song["MFCC_Features"]) - query_mfcc_mean)
        if distance < min_distance:
            min_distance = distance
            best_match = song
    if min_distance <= threshold and best_match is not None:
        return best_match
    return None

def recognize_song_and_get_lyrics(audio_bytes, api_token):
    # Convert audio to a format compatible with AudD (e.g., MP3)
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    audio.export("temp_audio.mp3", format="mp3")
    
    # AudD API request
    url = "https://api.audd.io/"
    files = {'file': open("temp_audio.mp3", "rb")}
    data = {
        'api_token': api_token,
        'return': 'lyrics'  # Request lyrics along with song info
    }
    response = requests.post(url, files=files, data=data)
    
    result = response.json()
    if result['status'] == 'success' and result['result'] is not None:
        song_info = {
            "Title": result['result'].get('title', 'Unknown'),
            "Artist": result['result'].get('artist', 'Unknown'),
            "Lyrics": result['result'].get('lyrics', 'Lyrics not available')
        }
        return song_info
    return None

# Streamlit UI
st.title("🎵 Tuneify: Music Recognition & Recommendation")
st.markdown("""
### 🔍 How to Use Tuneify:
1. **Upload an audio file** (MP3 or WAV format, max 50MB).
2. **View the MFCC visualization** of the uploaded track.
3. **Adjust the Euclidean distance slider** for MFCC-based recommendations.
4. **Click 'Find Recommendation'** for MFCC matching or 'Recognize Song & Lyrics' for high-quality recognition with lyrics.
""")

uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])
distance_threshold = st.slider("Set Euclidean Distance Threshold", min_value=0.0, max_value=100.0, value=20.0, step=0.5)
api_token = st.text_input("Enter your AudD API Token", type="password")

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

        # MFCC-based recommendation
        if st.button("Find Recommendation"):
            matched_song = find_closest_match(query_mfcc, dataset, distance_threshold)
            if matched_song is not None:
                st.subheader("🎶 Identified Song (MFCC-Based)")
                st.write(f"**Title:** {matched_song['Title']}")
                st.write(f"**Artist:** {matched_song['Artist']}")
                st.write(f"**Genre:** {matched_song['Genre']}")
            else:
                st.write("❌ No recommendations found. Try adjusting the threshold.")

        # High-quality recognition with lyrics
        if st.button("Recognize Song & Lyrics") and api_token:
            song_info = recognize_song_and_get_lyrics(audio_bytes, api_token)
            if song_info:
                st.subheader("🎶 Recognized Song (High-Quality)")
                st.write(f"**Title:** {song_info['Title']}")
                st.write(f"**Artist:** {song_info['Artist']}")
                st.write("**Lyrics:**")
                st.text(song_info['Lyrics'])
            else:
                st.write("❌ Could not recognize the song or fetch lyrics. Check your API token or audio quality.")
        elif not api_token:
            st.warning("⚠️ Please enter a valid AudD API token to use this feature.")
