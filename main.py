
=======
import os
import pandas as pd
import numpy as np 
import librosa
from glob import glob
from joblib import Parallel, delayed
import logging
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.model_selection import train_test_split
from tqdm import tqdm 

# setup logging to log all the errors
logging.basicConfig(
    filename='feature_extraction.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

# extract features of the audio files
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=30) # load the audio file, duration has to be 30s

        # the amptitude values of the audio signal 
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)  # trim the part where the audio is less than 20db

        # initialize a dictionary to store all the extracted features
        features = {'file_name': os.path.basename(file_path)}
        
        # initialize a helper function which computes the mean and standard deviation then stores in the dictionary
        def add_features(feature, name_prefix):
            feature_mean = np.mean(feature, axis=1)
            feature_std = np.std(feature, axis=1)
            for i, (mean, std) in enumerate(zip(feature_mean, feature_std), 1):
                features[f'{name_prefix}_mean_{i}'] = mean
                features[f'{name_prefix}_std_{i}'] = std

        # extract mfcc
        mfcc = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=13)
        # store the data
        add_features(mfcc, 'mfcc')

        # extract chroma
        chroma = librosa.feature.chroma_stft(y=y_trimmed, sr=sr)
        # store the data
        add_features(chroma, 'chroma')

        # extract spectral contrast
        spec_contrast = librosa.feature.spectral_contrast(y=y_trimmed, sr=sr)
        # store the data
        add_features(spec_contrast, 'spec_contrast')

        # extract zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y=y_trimmed)
        # store the data
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)

        # extract spectral centroid
        spec_centroid = librosa.feature.spectral_centroid(y=y_trimmed, sr=sr)
        features['spec_centroid_mean'] = np.mean(spec_centroid)
        features['spec_centroid_std'] = np.std(spec_centroid)
        
        # extract spectral roll-off
        spec_rolloff = librosa.feature.spectral_rolloff(y=y_trimmed, sr=sr)
        features['spec_rolloff_mean'] = np.mean(spec_rolloff)
        features['spec_rolloff_std'] = np.std(spec_rolloff)
    
    # log all the errors
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        return None
    
    # returrn the dictionary
    return features

def main():
    # determine the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # define base directory relative to the script's location
    base_dir = os.path.join(script_dir, 'data')
    
    # log and print the base directory
    logging.info(f"Base directory set to: {base_dir}")
    print(f"Base directory set to: {base_dir}")

    # define singing and talking directories
    singing_dir = os.path.join(base_dir, 'singing')
    talking_dir = os.path.join(base_dir, 'talking')

    # verify if directories exist
    if not os.path.isdir(singing_dir):
        logging.error(f"Singing directory does not exist: {singing_dir}")
        print(f"Singing directory does not exist: {singing_dir}")
    if not os.path.isdir(talking_dir):
        logging.error(f"Talking directory does not exist: {talking_dir}")
        print(f"Talking directory does not exist: {talking_dir}")

    # glob files with both lowercase and uppercase extensions
    singing_files = glob(os.path.join(singing_dir, '*.wav'))
    talking_files = glob(os.path.join(talking_dir, '*.wav'))

    # log and print number of files found
    logging.info(f"Found {len(singing_files)} singing files and {len(talking_files)} talking files.")
    print(f"Found {len(singing_files)} singing files and {len(talking_files)} talking files.")

    # combine all files with labels
    all_files = [(file, 1) for file in singing_files] + [(file, 0) for file in talking_files]

    # log the number of files
    logging.info(f"Total files to process: {len(all_files)}")
    print(f"Total files to process: {len(all_files)}")

    # if no files were found then log the error and print error messages
    if len(all_files) == 0:
        logging.error("No audio files found. Exiting the script.")
        print("No audio files found in the specified directories. Please check the paths and file extensions.")
        return

    # extract features in parallel with progress bar
    features_list = Parallel(n_jobs=-1)( # use all the available CPU cores 
        # delay the execution, return a callable objects then execute all of called functions in parallel
        delayed(extract_features)(file) for file, label in tqdm(all_files, desc="Extracting Features") 
    )

    # pair features with labels correctly
    labeled_features = []
    for feats, (file, label) in zip(features_list, all_files):
        if feats:
            feats['label'] = label
            labeled_features.append(feats)
        else:
            logging.warning(f"Features extraction failed for {file}")

    logging.info(f"Successfully extracted features for {len(labeled_features)} files.")
    print(f"Successfully extracted features for {len(labeled_features)} files.")

    # check if there are any features
    if len(labeled_features) == 0:
        logging.error("No features extracted. Exiting the script.")
        print("Feature extraction failed for all files. Please check the log for details.")
        return

    # create DataFrame from labeled_features
    df = pd.DataFrame(labeled_features)
    df.fillna(0, inplace=True)  # corrected parameter name

    # save to CSV
    df.to_csv('features.csv', index=False)
    logging.info("Feature extraction complete. Data saved to 'features.csv'")
    print("Feature extraction complete. Data saved to 'features.csv'")

    # load the dataset
    try:
        df = pd.read_csv('features.csv')
    except pd.errors.EmptyDataError:
        logging.error("features.csv is empty. Exiting the script.")
        print("features.csv is empty. No data to process.")
        return

    # check for missing values
    if df.isnull().sum().any():
        logging.warning("Missing values detected in 'features.csv'. Filling missing values with 0.")
        df.fillna(0, inplace=True)

    # Separate features and labels
    X = df.drop('label', axis=1)
    y = df['label']

    # Split into training and testing sets (e.g., 80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logging.info(f"Training set size: {X_train.shape[0]} samples")
    logging.info(f"Testing set size: {X_test.shape[0]} samples")
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")

    # Separate features and labels, drop 'file_name' before scaling
    X = df.drop(['label', 'file_name'], axis=1)  # Drop 'file_name' along with 'label'
    y = df['label']

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Split data into training and testing sets (if not done already)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit the scaler on the training data and transform both training and testing data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert scaled data back to DataFrames
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)

    # Add 'file_name' column back to the scaled data (you can access the original 'file_name' column from the original df)
    X_train_scaled_df['file_name'] = df.loc[X_train.index, 'file_name'].reset_index(drop=True)
    X_test_scaled_df['file_name'] = df.loc[X_test.index, 'file_name'].reset_index(drop=True)

    # Concatenate the labels back to the scaled features
    train_normalized_df = pd.concat([X_train_scaled_df, y_train.reset_index(drop=True)], axis=1)
    test_normalized_df = pd.concat([X_test_scaled_df, y_test.reset_index(drop=True)], axis=1)

    # Save the scaled data to CSV files
    train_normalized_df.to_csv('Talking-and-Singing-Classification/train_features_scaled.csv', index=False)
    test_normalized_df.to_csv('Talking-and-Singing-Classification/test_features_scaled.csv', index=False)

    # Save the scaler for future use
    joblib.dump(scaler, 'minmax_scaler.save')

    print("Completed!")

if __name__ == "__main__":
    main()
