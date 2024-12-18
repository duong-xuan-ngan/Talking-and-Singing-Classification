import os
import pandas as pd
import numpy as np 
import librosa
from glob import glob
from joblib import Parallel, delayed
import logging
from sklearn.preprocessing import MinMaxScaler
import joblib
from tqdm import tqdm 

# Setup logging to log all the errors
logging.basicConfig(
    filename='test_feature_extraction.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

# Function to read and parse the time_ranges.csv file
def read_time_ranges_csv(csv_path):
    """
    Reads the time_ranges.csv file and returns a dictionary mapping
    filenames to their corresponding time ranges (Start Time to End Time).
    """
    time_ranges = {}
    if not os.path.exists(csv_path):
        logging.error(f"Time ranges CSV file does not exist: {csv_path}")
        print(f"Time ranges CSV file does not exist: {csv_path}. Please ensure the file is present.")
        return time_ranges  # Return empty dict if CSV file doesn't exist

    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)

        # Combine Start Time and End Time into a single string for each file
        for _, row in df.iterrows():
            filename = row['Filename'].strip()
            start_time = row['Start Time']
            end_time = row['End Time']
            time_ranges[filename] = f"{start_time} to {end_time}"
        
        logging.info(f"Successfully loaded time ranges from {csv_path}")
    except Exception as e:
        logging.error(f"Error reading time_ranges.csv: {e}")
        print(f"Error reading time_ranges.csv: {e}")
    
    return time_ranges

# Extract features of the audio files
def extract_features(file_path):
    try:
        # Load the audio file with 5 seconds duration
        y, sr = librosa.load(file_path, duration=5)  
    
        # Trim silence
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)

        # Initialize a dictionary to store all the extracted features
        features = {'file_name': os.path.basename(file_path)}
        
        # Helper function to compute mean and std for a given feature
        def add_features(feature, name_prefix):
            feature_mean = np.mean(feature, axis=1)
            feature_std = np.std(feature, axis=1)
            for i, (mean, std) in enumerate(zip(feature_mean, feature_std), 1):
                features[f'{name_prefix}_mean_{i}'] = mean
                features[f'{name_prefix}_std_{i}'] = std

        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=13)
        add_features(mfcc, 'mfcc')

        # Extract Chroma
        chroma = librosa.feature.chroma_stft(y=y_trimmed, sr=sr)
        add_features(chroma, 'chroma')

        # Extract Spectral Contrast
        spec_contrast = librosa.feature.spectral_contrast(y=y_trimmed, sr=sr)
        add_features(spec_contrast, 'spec_contrast')

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

def main():
    # Determine the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define base directory relative to the script's location
    base_dir = os.path.join(script_dir, 'test_data')  # Pointing to test_data
    
    # Verify if the test_data directory exists
    if not os.path.isdir(base_dir):
        logging.error(f"Test data directory does not exist: {base_dir}")
        print(f"Test data directory does not exist: {base_dir}. Please ensure the 'test_data' folder is present.")
        return

    # Find all .wav files in the test_data directory
    test_files = glob(os.path.join(base_dir, '*.wav'))

    # Log and print number of files found
    logging.info(f"Found {len(test_files)} audio files in 'test_data'.")
    print(f"Found {len(test_files)} audio files in 'test_data'.")

    if len(test_files) == 0:
        logging.error("No audio files found in 'test_data'. Exiting the script.")
        print("No audio files found in 'test_data'. Please check the folder.")
        return

    # Read and parse the time_ranges.csv file
    time_ranges_csv_path = os.path.join(script_dir, 'time_ranges.csv')  # Adjust the filename
    time_ranges = read_time_ranges_csv(time_ranges_csv_path)

    if not time_ranges:
        logging.warning("No time ranges found. The 'time_range' column will be empty.")
        print("No time ranges found. The 'time_range' column will be empty.")

    # Extract features in parallel with progress bar
    features_list = Parallel(n_jobs=-1)(
        delayed(extract_features)(file) for file in tqdm(test_files, desc="Extracting Features") 
    )

    # Pair features with file names correctly
    labeled_features = []
    for feats, file in zip(features_list, test_files):
        if feats:
            # Add time_range if available
            file_name = feats['file_name']
            time_range = time_ranges.get(file_name, "N/A")  # Use "N/A" if time range is not found
            feats['time_range'] = time_range
            labeled_features.append(feats)
        else:
            logging.warning(f"Features extraction failed for {file}")

    # Check if there are any features
    if len(labeled_features) == 0:
        logging.error("No features extracted from any files. Exiting the script.")
        print("Feature extraction failed for all files. Please check the log for details.")
        return

    # Create DataFrame from labeled_features
    df = pd.DataFrame(labeled_features)
    df.fillna(0, inplace=True)  # Ensure no missing values

    # Load the pre-fitted scaler
    scaler_path = os.path.join(script_dir, 'minmax_scaler.save')
    if not os.path.exists(scaler_path):
        logging.error(f"Scaler file does not exist: {scaler_path}")
        print(f"Scaler file does not exist: {scaler_path}. Please ensure the scaler is trained and saved.")
        return

    try:
        scaler = joblib.load(scaler_path)
        logging.info("Pre-fitted scaler loaded successfully.")
        print("Pre-fitted scaler loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load the scaler: {e}")
        print(f"Failed to load the scaler: {e}")
        return

    # Apply the pre-fitted scaler to the features
    feature_columns = df.columns.drop(['file_name', 'time_range'])
    X = df[feature_columns]
    file_names = df['file_name']
    time_ranges_column = df['time_range']

    try:
        X_scaled = scaler.transform(X)
    except Exception as e:
        logging.error(f"Error during feature scaling: {e}")
        print(f"Error during feature scaling: {e}")
        return

    # Convert scaled data back to DataFrame
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_columns)
    X_scaled_df['file_name'] = file_names.reset_index(drop=True)
    X_scaled_df['time_range'] = time_ranges_column.reset_index(drop=True)

    # Save the scaled test data to CSV
    output_csv = os.path.join(script_dir, 'test_features_scaled.csv')
    try:
        X_scaled_df.to_csv(output_csv, index=False)
        logging.info(f"Scaled test features saved to '{output_csv}'")
        print(f"Scaled test features saved to '{output_csv}'")
    except Exception as e:
        logging.error(f"Failed to save scaled features to CSV: {e}")
        print(f"Failed to save scaled features to CSV: {e}")
        return

    print("Completed!")

if __name__ == "__main__":
    main()
