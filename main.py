import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from glob import glob
import librosa
import librosa.display
import IPython.display as ipd
from itertools import cycle

# setting up the visulization theme
sns.set_theme(style="white", palette=None) # set the style of the seaborn plots to be white 
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"] # store the color palette
color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

# loading audio files
audio_files = glob('Talking-and-Signing-Classification\data\singing\data1.wav') # get the file
ipd.Audio(audio_files[0]) #allow to listen to the music directly

# load the audio data
y, sr = librosa.load(audio_files[0]) # load the audio data into a waveform (y) and sample rate (sr)
print(f'y: {y[:10]}') # print the first 10 values of y
print(f'shape y: {y.shape}')
print(f'sr: {sr}') # sample rate

# plot the raw audio
pd.Series(y).plot(figsize=(10, 5), #using panda series to plot
                  lw=1,  #line width
                  title='Raw Audio Example',
                 color=color_pal[0])
plt.show()

# Trimming leading/lagging silence
y_trimmed, _ = librosa.effects.trim(y, top_db=20) #any y below 20db get trimmed

# plot the modified audio
pd.Series(y_trimmed).plot(figsize=(10, 5),
                  lw=1,
                  title='Raw Audio Trimmed Example',
                 color=color_pal[1])
plt.show()

# zoom in on a segment and plot
pd.Series(y[30000:30500]).plot(figsize=(10, 5),
                  lw=1,
                  title='Raw Audio Zoomed In Example',
                 color=color_pal[2])
plt.show()

# Short-Time Fourier Transform (STFT)
D = librosa.stft(y) # compute stft
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max) # convert to decibels
S_db.shape #print the shape of spectrogram array

# Plot the spectrogram
fig, ax = plt.subplots(figsize=(10, 5))
img = librosa.display.specshow(S_db,
                              x_axis='time',
                              y_axis='log',
                              ax=ax)
ax.set_title('Spectogram Example', fontsize=20)
fig.colorbar(img, ax=ax, format=f'%0.2f')
plt.show()

# compute the melspectrogram
S = librosa.feature.melspectrogram(y=y,
                                   sr=sr,
                                   n_mels=128 * 2,)
S_db_mel = librosa.amplitude_to_db(S, ref=np.max) # converts to db

fig, ax = plt.subplots(figsize=(10, 5))
# Plot the mel spectogram
img = librosa.display.specshow(S_db_mel,
                              x_axis='time',
                              y_axis='log',
                              ax=ax)
ax.set_title('Mel Spectogram Example', fontsize=20)
fig.colorbar(img, ax=ax, format=f'%0.2f')
plt.show()
