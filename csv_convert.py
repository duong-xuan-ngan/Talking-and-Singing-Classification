import os
import pandas as pd
import numpy as np
import librosa
from glob import glob
import logging
from sklearn.preprocessing import MinMaxScaler
import joblib
from tqdm import tqdm

# Configure logging to capture all events and errors
logging.basicConfig(
    filename='feature_extraction_final.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

def get_wav_duration(filepath):
    """
    Returns the duration of a WAV file in seconds.
    """
    try:
        y, sr = librosa.load(filepath, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        return duration
    except Exception as e:
        logging.error(f"Error reading {filepath}: {e}")
        return None

def extract_features(file_path, top_db=20):
    """
    Extracts audio features from a WAV file.

    Parameters:
    - file_path (str): Path to the WAV file.
    - top_db (int): The threshold (in decibels) below reference to consider as silence for trimming.

    Returns:
    - features (dict): A dictionary of extracted features and the file name.
    """
    try:
        # Load the audio file with a duration of 30 seconds
        y, sr = librosa.load(file_path, sr=None, duration=30)

        # Trim silence from the beginning and end
        y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)

        # Initialize a dictionary to store features
        features = {}

        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=13)
        for i in range(1, 14):
            mfcc_mean = np.mean(mfcc[i-1])
            mfcc_std = np.std(mfcc[i-1])
            features[f'mfcc_mean_{i}'] = mfcc_mean
            features[f'mfcc_std_{i}'] = mfcc_std

        # Extract Chroma features
        chroma = librosa.feature.chroma_stft(y=y_trimmed, sr=sr)
        for i in range(1, 13):
            chroma_mean = np.mean(chroma[i-1])
            chroma_std = np.std(chroma[i-1])
            features[f'chroma_mean_{i}'] = chroma_mean
            features[f'chroma_std_{i}'] = chroma_std

        # Extract Spectral Contrast
        spec_contrast = librosa.feature.spectral_contrast(y=y_trimmed, sr=sr)
        for i in range(1, 8):
            spec_contrast_mean = np.mean(spec_contrast[i-1])
            spec_contrast_std = np.std(spec_contrast[i-1])
            features[f'spec_contrast_mean_{i}'] = spec_contrast_mean
            features[f'spec_contrast_std_{i}'] = spec_contrast_std

        # Extract Zero-Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y=y_trimmed)
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)

        # Extract Spectral Centroid
        spec_centroid = librosa.feature.spectral_centroid(y=y_trimmed, sr=sr)
        features['spec_centroid_mean'] = np.mean(spec_centroid)
        features['spec_centroid_std'] = np.std(spec_centroid)

        # Extract Spectral Roll-off
        spec_rolloff = librosa.feature.spectral_rolloff(y=y_trimmed, sr=sr)
        features['spec_rolloff_mean'] = np.mean(spec_rolloff)
        features['spec_rolloff_std'] = np.std(spec_rolloff)

        # Add file name as a separate feature
        features['file_name'] = os.path.basename(file_path)

    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        return None

    return features

def process_final_folder(final_folder='final', output_csv='data.csv'):
    """
    Processes each WAV file in the final folder to extract features and save them to a CSV file.

    Parameters:
    - final_folder (str): Path to the 'final' directory containing WAV files.
    - output_csv (str): Path to the output CSV file.
    """
    # Verify that the final folder exists
    if not os.path.isdir(final_folder):
        logging.error(f"The directory '{final_folder}' does not exist.")
        print(f"The directory '{final_folder}' does not exist. Exiting the script.")
        return

    # Find all WAV files in the final folder
    wav_files = glob(os.path.join(final_folder, '*.wav'))

    if not wav_files:
        logging.warning(f"No WAV files found in '{final_folder}'.")
        print(f"No WAV files found in '{final_folder}'. Nothing to process.")
        return

    logging.info(f"Found {len(wav_files)} WAV file(s) in '{final_folder}'. Starting feature extraction.")
    print(f"Found {len(wav_files)} WAV file(s) in '{final_folder}'. Starting feature extraction.")

    features_list = []

    # Iterate over each WAV file with a progress bar
    for file in tqdm(wav_files, desc="Extracting Features", unit="file"):
        features = extract_features(file)
        if features:
            features_list.append(features)
        else:
            logging.warning(f"Feature extraction failed for {file}")

    if not features_list:
        logging.error("Feature extraction failed for all files. Exiting the script.")
        print("Feature extraction failed for all files. Please check the log for details.")
        return

    # Create a DataFrame from the features
    df = pd.DataFrame(features_list)

    # Handle missing values if any
    if df.isnull().values.any():
        logging.info("Missing values detected. Filling missing values with 0.")
        df.fillna(0, inplace=True)
        print("Missing values detected. Filled missing values with 0.")

    # Separate features from file names
    feature_columns = [col for col in df.columns if col != 'file_name']
    X = df[feature_columns]
    file_names = df['file_name']

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Scale the features
    X_scaled = scaler.fit_transform(X)

    # Create a scaled DataFrame
    df_scaled = pd.DataFrame(X_scaled, columns=feature_columns)
    df_scaled['file_name'] = file_names

    # Reorder columns to have 'file_name' last
    cols = [col for col in df_scaled.columns if col != 'file_name'] + ['file_name']
    df_scaled = df_scaled[cols]

    # Save the scaled features to CSV
    df_scaled.to_csv(output_csv, index=False)
    logging.info(f"Feature extraction complete. Data saved to '{output_csv}'.")
    print(f"Feature extraction complete. Data saved to '{output_csv}'.")

    # Save the scaler for future use
    scaler_path = 'minmax_scaler_final.save'
    joblib.dump(scaler, scaler_path)
    logging.info(f"Scaler saved to '{scaler_path}'.")
    print(f"Scaler saved to '{scaler_path}'.")

def main():
    """
    Main function to execute the feature extraction process.
    """
    # Define the path to the 'final' folder in the current directory
    current_directory = os.getcwd()
    final_folder = os.path.join(current_directory, 'final')

    # Define the output CSV path
    output_csv = os.path.join(current_directory, 'data.csv')

    # Start processing
    process_final_folder(final_folder=final_folder, output_csv=output_csv)

if __name__ == "__main__":
    main()
