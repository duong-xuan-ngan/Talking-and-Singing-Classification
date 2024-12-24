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

# Setup logging to log all the errors
logging.basicConfig(
    filename='feature_extraction.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

# Function to extract features from a single audio file
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=30)  # Load the audio file, duration has to be 30s

        # Trim silence from the audio signal
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)  # Trim the part where the audio is less than 20db

        # Initialize a dictionary to store all the extracted features
        features = {'file_name': os.path.basename(file_path)}

        # Helper function to compute mean and std, then store in the dictionary
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

# Function to save scaler parameters along with feature names to a JSON file
def save_scaler_parameters(scaler, feature_names, file_path):
    scaler_params = {
        'data_min_': scaler.data_min_.tolist(),
        'data_max_': scaler.data_max_.tolist(),
        'data_range_': scaler.data_range_.tolist(),
        'scale_': scaler.scale_.tolist(),
        'min_': scaler.min_.tolist(),
        'feature_names': feature_names  # Include feature names
    }
    with open(file_path, 'w') as f:
        json.dump(scaler_params, f, indent=4)
    logging.info(f"Scaler parameters and feature names saved to '{file_path}'.")

# Function to load scaler parameters along with feature names from a JSON file
def load_scaler_parameters(file_path):
    with open(file_path, 'r') as f:
        loaded_params = json.load(f)
    scaler = MinMaxScaler()
    scaler.data_min_ = np.array(loaded_params['data_min_'])
    scaler.data_max_ = np.array(loaded_params['data_max_'])
    scaler.data_range_ = np.array(loaded_params['data_range_'])
    scaler.scale_ = np.array(loaded_params['scale_'])
    scaler.min_ = np.array(loaded_params['min_'])
    feature_names = loaded_params['feature_names']
    return scaler, feature_names

def main():
    # Determine the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define base directory relative to the script's location
    base_dir = os.path.join(script_dir, 'data')

    # Log and print the base directory
    logging.info(f"Base directory set to: {base_dir}")
    print(f"Base directory set to: {base_dir}")

    # Define singing and talking directories
    singing_dir = os.path.join(base_dir, 'singing')
    talking_dir = os.path.join(base_dir, 'talking')

    # Verify if directories exist
    if not os.path.isdir(singing_dir):
        logging.error(f"Singing directory does not exist: {singing_dir}")
        print(f"Singing directory does not exist: {singing_dir}")
    if not os.path.isdir(talking_dir):
        logging.error(f"Talking directory does not exist: {talking_dir}")
        print(f"Talking directory does not exist: {talking_dir}")

    # Glob files with both lowercase and uppercase extensions
    singing_files = glob(os.path.join(singing_dir, '*.wav'))
    talking_files = glob(os.path.join(talking_dir, '*.wav'))

    # Log and print number of files found
    logging.info(f"Found {len(singing_files)} singing files and {len(talking_files)} talking files.")
    print(f"Found {len(singing_files)} singing files and {len(talking_files)} talking files.")

    # Combine all files with labels
    all_files = [(file, 1) for file in singing_files] + [(file, 0) for file in talking_files]

    # Log the number of files
    logging.info(f"Total files to process: {len(all_files)}")
    print(f"Total files to process: {len(all_files)}")

    # If no files were found then log the error and print error messages
    if len(all_files) == 0:
        logging.error("No audio files found. Exiting the script.")
        print("No audio files found in the specified directories. Please check the paths and file extensions.")
        return

    # Extract features in parallel with progress bar
    features_list = Parallel(n_jobs=-1)(
        delayed(extract_features)(file) for file, label in tqdm(all_files, desc="Extracting Features") 
    )

    # Pair features with labels correctly
    labeled_features = []
    for feats, (file, label) in zip(features_list, all_files):
        if feats:
            feats['label'] = label
            labeled_features.append(feats)
        else:
            logging.warning(f"Features extraction failed for {file}")

    logging.info(f"Successfully extracted features for {len(labeled_features)} files.")
    print(f"Successfully extracted features for {len(labeled_features)} files.")

    # Check if there are any features
    if len(labeled_features) == 0:
        logging.error("No features extracted. Exiting the script.")
        print("Feature extraction failed for all files. Please check the log for details.")
        return

    # Create DataFrame from labeled_features
    df = pd.DataFrame(labeled_features)
    df.fillna(0, inplace=True)  # Corrected parameter name

    # Save to CSV
    df.to_csv('features.csv', index=False)
    logging.info("Feature extraction complete. Data saved to 'features.csv'")
    print("Feature extraction complete. Data saved to 'features.csv'")

    # Load the dataset
    try:
        df = pd.read_csv('features.csv')
    except pd.errors.EmptyDataError:
        logging.error("features.csv is empty. Exiting the script.")
        print("features.csv is empty. No data to process.")
        return

    # Check for missing values
    if df.isnull().sum().any():
        logging.warning("Missing values detected in 'features.csv'. Filling missing values with 0.")
        df.fillna(0, inplace=True)

    # **New Step:** Preserve 'file_name' before dropping
    file_names = df['file_name']

    # Separate features and labels
    X = df.drop(['label', 'file_name'], axis=1)  # Drop 'file_name' along with 'label'
    y = df['label']

    # **New Step:** Split 'file_name' alongside features and labels
    X_train, X_test, y_train, y_test, train_file_names, test_file_names = train_test_split(
        X, y, file_names, test_size=0.2, random_state=42, stratify=y
    )

    logging.info(f"Training set size: {X_train.shape[0]} samples")
    logging.info(f"Testing set size: {X_test.shape[0]} samples")
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")

    # Path to the scaler parameters JSON file
    scaler_params_path = 'minmax_scaler_params.json'

    # Check if the scaler parameters file exists
    if os.path.exists(scaler_params_path):
        # Load the scaler from the JSON file
        try:
            scaler, feature_names = load_scaler_parameters(scaler_params_path)
            logging.info(f"Loaded scaler parameters from '{scaler_params_path}'.")
            print(f"Loaded scaler parameters from '{scaler_params_path}'.")
        except Exception as e:
            logging.error(f"Failed to load scaler parameters: {e}")
            print(f"Failed to load scaler parameters: {e}")
            return

        # Ensure that the feature names match
        current_feature_names = X.columns.tolist()
        if feature_names != current_feature_names:
            logging.error("Feature names in the scaler parameters do not match the current data.")
            print("Feature names in the scaler parameters do not match the current data.")
            return

        # Transform the data using the loaded scaler
        try:
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            logging.info("Data scaled using the loaded MinMaxScaler.")
            print("Data scaled using the loaded MinMaxScaler.")
        except Exception as e:
            logging.error(f"Error during scaling: {e}")
            print(f"Error during scaling: {e}")
            return

    else:
        # If the scaler parameters file does not exist, fit a new scaler and save it
        logging.info(f"Scaler parameters file '{scaler_params_path}' not found. Fitting a new scaler.")
        print(f"Scaler parameters file '{scaler_params_path}' not found. Fitting a new scaler.")

        # Initialize the MinMaxScaler
        scaler = MinMaxScaler()

        # Fit the scaler on the training data and transform both training and testing data
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Save the scaler parameters and feature names to a JSON file
        feature_names = X.columns.tolist()  # Extract feature names
        save_scaler_parameters(scaler, feature_names, scaler_params_path)

        logging.info("Scaler fitted and parameters saved.")
        print("Scaler fitted and parameters saved.")

    # If scaler was loaded or newly fitted, proceed to save the scaled data
    # Convert scaled data back to DataFrames
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)

    # **New Step:** Add 'file_name' back to the scaled DataFrames
    X_train_scaled_df = X_train_scaled_df.reset_index(drop=True)
    X_test_scaled_df = X_test_scaled_df.reset_index(drop=True)
    train_file_names = train_file_names.reset_index(drop=True)
    test_file_names = test_file_names.reset_index(drop=True)

    # Concatenate the labels and 'file_name' back to the scaled features
    train_normalized_df = pd.concat([X_train_scaled_df, y_train.reset_index(drop=True), train_file_names], axis=1)
    test_normalized_df = pd.concat([X_test_scaled_df, y_test.reset_index(drop=True), test_file_names], axis=1)

    # **New Step:** Reorder columns to have 'file_name' at the end
    # Get list of columns
    train_columns = list(train_normalized_df.columns)
    test_columns = list(test_normalized_df.columns)

    # Remove 'file_name' from its current position
    train_columns.remove('file_name')
    test_columns.remove('file_name')

    # Append 'file_name' to the end
    train_columns.append('file_name')
    test_columns.append('file_name')

    # Reorder the DataFrames
    train_normalized_df = train_normalized_df[train_columns]
    test_normalized_df = test_normalized_df[test_columns]

    # Save the scaled data to CSV files
    train_normalized_df.to_csv('train_features_scaled.csv', index=False)
    test_normalized_df.to_csv('test_features_scaled.csv', index=False)

    logging.info("Scaled data saved to 'train_features_scaled.csv' and 'test_features_scaled.csv'.")
    print("Scaled data saved to 'train_features_scaled.csv' and 'test_features_scaled.csv'.")

    print("Completed!")

# Entry point of the script
if __name__ == "__main__":
    main()
