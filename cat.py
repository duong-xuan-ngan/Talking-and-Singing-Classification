import os
import pandas as pd
import numpy as np 
import librosa
from glob import glob
from joblib import Parallel, delayed
import logging
from sklearn.preprocessing import MinMaxScaler
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm 

# Setup logging to record errors and activity
logging.basicConfig(
    filename='feature_extraction.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

# Function to extract features from an audio file
def extract_features(file_path):
    try:
        # Load audio file, limiting to 30 seconds
        y, sr = librosa.load(file_path, duration=30)

        # Trim silence below a set decibel threshold
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)

        # Dictionary to store extracted features
        features = {'file_name': os.path.basename(file_path)}

        # Helper to add mean and std for each feature to the dictionary
        def add_features(feature, name_prefix):
            feature_mean = np.mean(feature, axis=1)
            feature_std = np.std(feature, axis=1)
            for i, (mean, std) in enumerate(zip(feature_mean, feature_std), 1):
                features[f'{name_prefix}_mean_{i}'] = mean
                features[f'{name_prefix}_std_{i}'] = std

        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=13)
        add_features(mfcc, 'mfcc')

        # Extract chroma features
        chroma = librosa.feature.chroma_stft(y=y_trimmed, sr=sr)
        add_features(chroma, 'chroma')

        # Extract spectral contrast features
        spec_contrast = librosa.feature.spectral_contrast(y=y_trimmed, sr=sr)
        add_features(spec_contrast, 'spec_contrast')

        # Calculate zero-crossing rate and add mean/std
        zcr = librosa.feature.zero_crossing_rate(y=y_trimmed)
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)

        # Calculate spectral centroid and add mean/std
        spec_centroid = librosa.feature.spectral_centroid(y=y_trimmed, sr=sr)
        features['spec_centroid_mean'] = np.mean(spec_centroid)
        features['spec_centroid_std'] = np.std(spec_centroid)

        # Calculate spectral roll-off and add mean/std
        spec_rolloff = librosa.feature.spectral_rolloff(y=y_trimmed, sr=sr)
        features['spec_rolloff_mean'] = np.mean(spec_rolloff)
        features['spec_rolloff_std'] = np.std(spec_rolloff)

    except Exception as e:
        # Log any errors that occur
        logging.error(f"Error processing {file_path}: {e}")
        return None

    return features

# Function to save scaler parameters and feature names to a JSON file
def save_scaler_parameters(scaler, feature_names, file_path):
    scaler_params = {
        'data_min_': scaler.data_min_.tolist(),
        'data_max_': scaler.data_max_.tolist(),
        'data_range_': scaler.data_range_.tolist(),
        'scale_': scaler.scale_.tolist(),
        'min_': scaler.min_.tolist(),
        'feature_names': feature_names
    }
    with open(file_path, 'w') as f:
        json.dump(scaler_params, f, indent=4)
    logging.info(f"Scaler parameters and feature names saved to '{file_path}'.")

# Function to load scaler parameters and feature names from a JSON file
def load_scaler_parameters(file_path):
    with open(file_path, 'r') as f:
        loaded_params = json.load(f)
    scaler = MinMaxScaler()
    # Load scaler parameters into a MinMaxScaler object
    scaler.data_min_ = np.array(loaded_params['data_min_'])
    scaler.data_max_ = np.array(loaded_params['data_max_'])
    scaler.data_range_ = np.array(loaded_params['data_range_'])
    scaler.scale_ = np.array(loaded_params['scale_'])
    scaler.min_ = np.array(loaded_params['min_'])
    feature_names = loaded_params['feature_names']
    return scaler, feature_names

def main():
    # Determine the script's directory and set the base data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, 'data')

    # Define directories for singing and talking audio files
    singing_dir = os.path.join(base_dir, 'singing')
    talking_dir = os.path.join(base_dir, 'talking')

    # Check if directories exist and log errors if they do not
    if not os.path.isdir(singing_dir):
        logging.error(f"Singing directory does not exist: {singing_dir}")
        print(f"Singing directory does not exist: {singing_dir}")
    if not os.path.isdir(talking_dir):
        logging.error(f"Talking directory does not exist: {talking_dir}")
        print(f"Talking directory does not exist: {talking_dir}")

    # Gather all wav files from the directories
    singing_files = glob(os.path.join(singing_dir, '*.wav'))
    talking_files = glob(os.path.join(talking_dir, '*.wav'))

    # Combine file paths with labels (1 for singing, 0 for talking)
    all_files = [(file, 1) for file in singing_files] + [(file, 0) for file in talking_files]

    # Exit if no files are found
    if len(all_files) == 0:
        logging.error("No audio files found. Exiting the script.")
        print("No audio files found in the specified directories. Please check the paths and file extensions.")
        return

    # Extract features from all audio files using parallel processing
    features_list = Parallel(n_jobs=-1)(
        delayed(extract_features)(file) for file, label in tqdm(all_files, desc="Extracting Features") 
    )

    # Combine extracted features with labels
    labeled_features = []
    for feats, (file, label) in zip(features_list, all_files):
        if feats:
            feats['label'] = label
            labeled_features.append(feats)
        else:
            logging.warning(f"Features extraction failed for {file}")

    # Create a dataframe from the features
    df = pd.DataFrame(labeled_features)
    
    # **Rearrange columns to move 'file_name' to the end**
    if 'file_name' in df.columns:
        cols = list(df.columns)
        cols.remove('file_name')  # Remove 'file_name' from its current position
        cols.append('file_name')   # Append 'file_name' at the end
        df = df[cols]               # Reorder the DataFrame columns

    # Save to csv
    df.fillna(0, inplace=True)
    df.to_csv('features.csv', index=False)
    logging.info("Features extracted and saved to 'features.csv'.")

    # **Scale features.csv using the scaler parameters JSON file**
    scaler_params_path = 'minmax_scaler_params.json'
    try:
        # Load the csv to process features and labels
        df = pd.read_csv('features.csv')
    except pd.errors.EmptyDataError:
        logging.error("features.csv is empty. Exiting the script.")
        print("features.csv is empty. No data to process.")
        return

    # Separate features and labels, preserve file names
    file_names = df['file_name']
    X = df.drop(['label', 'file_name'], axis=1)
    y = df['label']

    # Initialize scaler and feature names
    scaler = None
    feature_names = X.columns.tolist()

    if os.path.exists(scaler_params_path):
        # Load existing scaler and validate feature names
        try:
            scaler, loaded_feature_names = load_scaler_parameters(scaler_params_path)
            if loaded_feature_names != feature_names:
                logging.error("Feature names in the scaler parameters do not match the current data.")
                print("Feature names in the scaler parameters do not match the current data.")
                return
            # Apply scaler to the entire dataset
            X_scaled = scaler.transform(X)
            logging.info("features.csv scaled using existing scaler parameters.")
        except Exception as e:
            logging.error(f"Error loading scaler parameters: {e}")
            print(f"Error loading scaler parameters: {e}")
            return
    else:
        # Fit a new scaler if parameters do not exist
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        save_scaler_parameters(scaler, feature_names, scaler_params_path)
        logging.info("New scaler fitted and features.csv scaled.")

    # Create a scaled dataframe
    scaled_df = pd.concat(
        [pd.DataFrame(X_scaled, columns=feature_names),
         y.reset_index(drop=True),
         file_names.reset_index(drop=True)],
        axis=1
    )

    # Rearrange columns to move 'file_name' to the end
    if 'file_name' in scaled_df.columns:
        cols = list(scaled_df.columns)
        cols.remove('file_name')
        cols.append('file_name')
        scaled_df = scaled_df[cols]

    # Save the scaled features to a new CSV
    scaled_df.to_csv('features_scaled.csv', index=False)
    logging.info("Scaled features saved to 'features_scaled.csv'.")

    # **Proceed to split the scaled data into training and testing sets**

    # Separate features and labels from the scaled dataframe
    X_scaled = scaled_df.drop(['label', 'file_name'], axis=1)
    y = scaled_df['label']
    file_names = scaled_df['file_name']

    # Split data into training and testing sets, stratifying by labels
    X_train, X_test, y_train, y_test, train_file_names, test_file_names = train_test_split(
        X_scaled, y, file_names, test_size=0.2, random_state=42, stratify=y
    )

    # Since features.csv is already scaled, we don't need to scale again.
    # However, if you want to ensure that train and test sets use the same scaler:
    # (Optional) You can verify that X_train and X_test are already scaled.

    # Save scaled training and testing data to csv
    train_normalized_df = pd.concat(
        [X_train.reset_index(drop=True),
         y_train.reset_index(drop=True),
         train_file_names.reset_index(drop=True)],
        axis=1
    )
    test_normalized_df = pd.concat(
        [X_test.reset_index(drop=True),
         y_test.reset_index(drop=True),
         test_file_names.reset_index(drop=True)],
        axis=1
    )

    # Rearrange columns to move 'file_name' to the end
    for df_split in [train_normalized_df, test_normalized_df]:
        if 'file_name' in df_split.columns:
            cols = list(df_split.columns)
            cols.remove('file_name')
            cols.append('file_name')
            df_split = df_split[cols]

    train_normalized_df.to_csv('train_features_scaled.csv', index=False)
    test_normalized_df.to_csv('test_features_scaled.csv', index=False)
    logging.info("Training and testing scaled features saved to 'train_features_scaled.csv' and 'test_features_scaled.csv'.")

    print("Completed!")

# Entry point of the script
if __name__ == "__main__":
    main()
