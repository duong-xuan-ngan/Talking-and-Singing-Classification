import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Apply seaborn style for better aesthetics
sns.set_theme(style="whitegrid")

# Define the file path
folder_path = "extracted_vocal"
file_name = "audio.wav"
audio_path = os.path.join(folder_path, file_name)

# Check if the file exists
if not os.path.exists(audio_path):
    raise FileNotFoundError(f"The file {audio_path} does not exist.")

# Load the audio file
y, sr = librosa.load(audio_path, sr=None)

# Check original audio length
print("Original Audio Length:", len(y))

# Trimming leading/lagging silence with adjusted top_db
y_trimmed, index = librosa.effects.trim(y, top_db=30)

# Check trimmed audio length
print("Trimmed Audio Length:", len(y_trimmed))

# Ensure that trimming hasn't removed all audio
if len(y_trimmed) == 0:
    raise ValueError("Trimming removed all audio. Consider reducing the 'top_db' parameter.")

# Extract features
mfcc = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=13)
chroma = librosa.feature.chroma_stft(y=y_trimmed, sr=sr)
spectral_contrast = librosa.feature.spectral_contrast(y=y_trimmed, sr=sr)
zero_crossing_rate = librosa.feature.zero_crossing_rate(y_trimmed)
spectral_centroid = librosa.feature.spectral_centroid(y=y_trimmed, sr=sr)
spectral_rolloff = librosa.feature.spectral_rolloff(y=y_trimmed, sr=sr)

# Function to plot spectrogram-like features
def plot_spectrogram_feature(feature, feature_name, y_axis, sr, cmap='coolwarm'):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(feature, x_axis='time', y_axis=y_axis, sr=sr, cmap=cmap)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"{feature_name}", fontsize=16)
    plt.tight_layout()
    plt.show()

# Function to plot time-series features
def plot_time_series_feature(feature, feature_name, color='b'):
    plt.figure(figsize=(10, 4))
    plt.plot(feature, color=color)
    plt.title(f"{feature_name}", fontsize=16)
    plt.xlabel("Frames", fontsize=12)
    plt.ylabel(feature_name, fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 1. MFCC
plot_spectrogram_feature(mfcc, "MFCC", y_axis='mel', sr=sr)

# 2. Chroma
plot_spectrogram_feature(chroma, "Chroma", y_axis='chroma', sr=sr)

# 3. Spectral Contrast
plot_spectrogram_feature(spectral_contrast, "Spectral Contrast", y_axis='linear', sr=sr)

# 4. Spectral Centroid
plot_time_series_feature(spectral_centroid[0], "Spectral Centroid", color='r')

# 5. Spectral Roll-Off
plot_time_series_feature(spectral_rolloff[0], "Spectral Roll-Off", color='g')

# 6. Zero-Crossing Rate
plot_time_series_feature(zero_crossing_rate[0], "Zero-Crossing Rate", color='purple')
