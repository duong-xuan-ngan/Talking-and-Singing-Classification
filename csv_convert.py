import os
import pandas as pd
import numpy as np
import librosa
import logging
from sklearn.preprocessing import MinMaxScaler
import json

# Configure logging to capture all events and errors
logging.basicConfig(
    filename='feature_extraction_final.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

def extract_features(file_path, top_db=20):
    """
    Extracts audio features from a WAV file.

    Parameters:
    - file_path (str): Path to the WAV file.
    - top_db (int): The threshold (in decibels) below reference to consider as silence for trimming.

    Returns:
    - features (dict): A dictionary of extracted features.
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

    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        return None

    return features

def save_scaler_parameters(scaler, file_path, feature_names):
    """
    Saves the MinMaxScaler parameters and feature names to a JSON file.

    Parameters:
    - scaler (MinMaxScaler): The fitted scaler.
    - file_path (str): Path to the JSON file where parameters will be saved.
    - feature_names (list): List of feature names used in scaling.
    """
    scaler_params = {
        'feature_names': feature_names,
        'data_min_': scaler.data_min_.tolist(),
        'data_max_': scaler.data_max_.tolist(),
        'data_range_': scaler.data_range_.tolist(),
        'scale_': scaler.scale_.tolist(),
        'min_': scaler.min_.tolist()
    }
    with open(file_path, 'w') as f:
        json.dump(scaler_params, f)
    logging.info(f"Scaler parameters saved to '{file_path}'.")
    print(f"Scaler parameters saved to '{file_path}'.")

def load_scaler_parameters(file_path):
    """
    Loads the MinMaxScaler parameters and feature names from a JSON file.

    Parameters:
    - file_path (str): Path to the JSON file containing scaler parameters.

    Returns:
    - scaler (MinMaxScaler): Reconstructed scaler with loaded parameters.
    - feature_names (list): List of feature names used in the scaler.
    """
    try:
        with open(file_path, 'r') as f:
            loaded_params = json.load(f)
        scaler = MinMaxScaler()
        scaler.data_min_ = np.array(loaded_params['data_min_'])
        scaler.data_max_ = np.array(loaded_params['data_max_'])
        scaler.data_range_ = np.array(loaded_params['data_range_'])
        scaler.scale_ = np.array(loaded_params['scale_'])
        scaler.min_ = np.array(loaded_params['min_'])
        feature_names = loaded_params['feature_names']
        logging.info(f"Scaler parameters and feature names loaded from '{file_path}'.")
        print(f"Scaler parameters and feature names loaded from '{file_path}'.")
        return scaler, feature_names
    except Exception as e:
        logging.error(f"Error loading scaler parameters from '{file_path}': {e}")
        print(f"Error loading scaler parameters from '{file_path}': {e}")
        return None, None

def process_audio_file(audio_file_path, output_csv='data.csv', scaler_json_path='minmax_scaler_params.json'):
    """
    Processes a single WAV file to extract features and save them to a CSV file.
    Utilizes an existing MinMaxScaler if available, otherwise fits a new scaler and saves its parameters.

    Parameters:
    - audio_file_path (str): Path to the WAV file.
    - output_csv (str): Path to the output CSV file.
    - scaler_json_path (str): Path to the MinMaxScaler JSON parameter file.
    """
    # Check if the audio file exists
    if not os.path.isfile(audio_file_path):
        logging.error(f"The file '{audio_file_path}' does not exist.")
        print(f"The file '{audio_file_path}' does not exist. Exiting the script.")
        return

    logging.info(f"Starting feature extraction for '{audio_file_path}'.")
    print(f"Starting feature extraction for '{audio_file_path}'.")

    # Extract features
    features = extract_features(audio_file_path)
    if not features:
        logging.error(f"Feature extraction failed for {audio_file_path}")
        print(f"Feature extraction failed for {audio_file_path}. Please check the log for details.")
        return

    # Create a DataFrame from the features
    df = pd.DataFrame([features])

    # Handle missing values if any
    if df.isnull().values.any():
        logging.info("Missing values detected. Filling missing values with 0.")
        df.fillna(0, inplace=True)
        print("Missing values detected. Filled missing values with 0.")

    # Remove 'file_name' if present
    if 'file_name' in df.columns:
        df.drop(columns=['file_name'], inplace=True)

    # Get feature columns
    feature_columns = list(df.columns)

    # Initialize or load the MinMaxScaler
    if os.path.exists(scaler_json_path):
        scaler, saved_feature_names = load_scaler_parameters(scaler_json_path)
        if scaler is None or saved_feature_names is None:
            print("Failed to load scaler parameters. Exiting the script.")
            return
        # Ensure the current features match the saved feature names
        try:
            X = df[saved_feature_names]
        except KeyError as e:
            logging.error(f"Feature columns mismatch: {e}")
            print(f"Feature columns mismatch: {e}. Exiting the script.")
            return

        # Transform the data
        try:
            X_scaled = scaler.transform(X)
            logging.info("Transformed data using the existing scaler.")
            print("Transformed data using the existing scaler.")
        except Exception as e:
            logging.error(f"Error transforming data with the existing scaler: {e}")
            print(f"Error transforming data with the existing scaler: {e}. Exiting the script.")
            return
    else:
        # Initialize a new scaler and fit-transform the data
        scaler = MinMaxScaler()
        logging.info("Initialized a new MinMaxScaler.")
        print("Initialized a new MinMaxScaler.")

        try:
            X = df[feature_columns]
            X_scaled = scaler.fit_transform(X)
            logging.info("Fitted and transformed data using a new scaler.")
            print("Fitted and transformed data using a new scaler.")
            # Save the scaler parameters with feature names
            save_scaler_parameters(scaler, scaler_json_path, feature_columns)
        except Exception as e:
            logging.error(f"Error fitting and transforming data: {e}")
            print(f"Error fitting and transforming data: {e}. Exiting the script.")
            return

    # Create a scaled DataFrame
    df_scaled = pd.DataFrame(X_scaled, columns=feature_columns)

    # Save the scaled features to CSV
    try:
        df_scaled.to_csv(output_csv, index=False)
        logging.info(f"Feature extraction complete. Data saved to '{output_csv}'.")
        print(f"Feature extraction complete. Data saved to '{output_csv}'.")
    except Exception as e:
        logging.error(f"Error saving scaled data to '{output_csv}': {e}")
        print(f"Error saving scaled data to '{output_csv}': {e}. Exiting the script.")
        return

def main():
    """
    Main function to execute the feature extraction process.
    """
    # Define the path to the 'extracted_vocal' folder and 'audio.wav'
    current_directory = os.getcwd()
    extracted_vocal_folder = os.path.join(current_directory, 'extracted_vocal')
    audio_file = os.path.join(extracted_vocal_folder, 'audio.wav')

    # Define the output CSV path
    output_csv = os.path.join(current_directory, 'data.csv')

    # Define the scaler JSON path
    scaler_json_path = os.path.join(current_directory, 'minmax_scaler_params.json')

    # Start processing
    process_audio_file(audio_file_path=audio_file, output_csv=output_csv, scaler_json_path=scaler_json_path)

if __name__ == "__main__":
    main()