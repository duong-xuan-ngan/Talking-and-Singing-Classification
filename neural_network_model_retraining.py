# Filename: neural_network_model_retraining.py

import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE  # For handling class imbalance
from datetime import datetime
import logging

# -----------------------------
# Setup Logging
# -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger()

# -----------------------------
# Create Directory Structure
# -----------------------------
def create_directories():
    """
    Create necessary directories for models and backups if they don't exist.
    """
    directories = ['models', 'model_backups']
    for dir in directories:
        if not os.path.exists(dir):
            os.makedirs(dir)
            logger.info(f"Directory '{dir}' created.")
        else:
            logger.info(f"Directory '{dir}' already exists.")

# -----------------------------
# Backup Existing Model Function (Method 2)
# -----------------------------
def backup_existing_model(model_path='models/best_neural_network_model.h5', backup_dir='model_backups'):
    """
    Backup the existing model to a specified directory with a timestamp.
    
    Parameters:
        model_path (str): Path to the existing model file.
        backup_dir (str): Directory where backups will be stored.
    """
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
        logger.info(f"Directory '{backup_dir}' created for backups.")
    if os.path.isfile(model_path):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_path = os.path.join(backup_dir, f"best_neural_network_model_backup_{timestamp}.h5")
        shutil.copy(model_path, backup_path)
        logger.info(f"Backup of the existing model saved to '{backup_path}'")
    else:
        logger.info(f"No existing model found at '{model_path}'. Skipping backup.")

# -----------------------------
# Load Existing Model Function
# -----------------------------
def load_pretrained_model(model_path='models/best_neural_network_model.h5'):
    """
    Load the existing trained model if it exists.
    
    Parameters:
        model_path (str): Path to the existing model file.
    
    Returns:
        model (Sequential): Loaded Keras model or None if not found.
    """
    if os.path.exists(model_path):
        try:
            model = load_model(model_path)
            logger.info(f"Loaded existing model from '{model_path}'")
            return model
        except Exception as e:
            logger.error(f"Error loading the existing model: {e}")
            logger.info("Proceeding to create a new model.")
            return None
    else:
        logger.info(f"No existing model found at '{model_path}'. Creating a new model.")
        return None

# -----------------------------
# Split wrong_audios.csv into train and test (Data Preparation)
# -----------------------------
def split_wrong_audios(wrong_audios_path='wrong_audios.csv',
                      train_output_path='train_features_scaled.csv',
                      test_output_path='test_features_scaled.csv',
                      test_size=0.2,
                      random_state=42):
    """
    Split wrong_audios.csv into training and testing datasets.
    
    Parameters:
        wrong_audios_path (str): Path to the wrong_audios.csv file
        train_output_path (str): Path to save training data
        test_output_path (str): Path to save testing data
        test_size (float): Proportion of data to use for testing (default: 0.2)
        random_state (int): Random seed for reproducibility
    """
    try:
        # Check if wrong_audios.csv exists
        if not os.path.exists(wrong_audios_path):
            logger.error(f"File '{wrong_audios_path}' not found.")
            return False

        # Load the data
        data = pd.read_csv(wrong_audios_path)
        
        if data.empty:
            logger.error(f"'{wrong_audios_path}' is empty.")
            return False
            
        # Log initial data info
        logger.info(f"Loaded data from {wrong_audios_path}")
        logger.info(f"Total samples: {len(data)}")
        logger.info(f"Columns: {', '.join(data.columns)}")
        
        # Calculate split sizes
        total_samples = len(data)
        test_samples = int(total_samples * test_size)
        train_samples = total_samples - test_samples
        
        # Shuffle the data
        shuffled_data = data.sample(frac=1, random_state=random_state)
        
        # Split the data
        train_data = shuffled_data.iloc[:train_samples]
        test_data = shuffled_data.iloc[train_samples:]
        
        # Save the split datasets
        train_data.to_csv(train_output_path, index=False)
        test_data.to_csv(test_output_path, index=False)
        
        # Log split information
        logger.info(f"\nData split complete:")
        logger.info(f"Training set ({(1-test_size)*100}%): {len(train_data)} samples")
        logger.info(f"Testing set ({test_size*100}%): {len(test_data)} samples")
        logger.info(f"\nFiles saved:")
        logger.info(f"Training data: {train_output_path}")
        logger.info(f"Testing data: {test_output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error splitting data: {e}")
        return False

def load_data(train_path='train_features_scaled.csv', test_path='test_features_scaled.csv'):
    """
    Load preprocessed and scaled training and testing datasets.
    Creates empty files with headers if they don't exist.
    
    Parameters:
        train_path (str): Path to the training data CSV file.
        test_path (str): Path to the testing data CSV file.
    
    Returns:
        train_df (DataFrame): Training data.
        test_df (DataFrame): Testing data.
    """
    # Check if files exist, if not, create them with headers
    for file_path in [train_path, test_path]:
        if not os.path.exists(file_path):
            logger.warning(f"File '{file_path}' not found. Creating empty file with headers...")
            # Create empty DataFrame with required columns
            empty_df = pd.DataFrame(columns=['file_name', 'label'] + 
                                  [f'feature_{i}' for i in range(1, 21)])  # Adjust feature count as needed
            empty_df.to_csv(file_path, index=False)
            logger.info(f"Created empty file: {file_path}")

    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        if train_df.empty or test_df.empty:
            logger.warning("One or both datasets are empty. Please ensure data is properly loaded before proceeding.")
        else:
            logger.info("Data loaded successfully.")
            logger.info(f"Training Data Shape: {train_df.shape}")
            logger.info(f"Testing Data Shape: {test_df.shape}")
        
        return train_df, test_df
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        logger.error("Please check the file format and contents.")
        exit()
def perform_eda(train_df):
    """
    Perform EDA: Class distribution and feature correlations.
    
    Parameters:
        train_df (DataFrame): Training data.
    """
    # Class Distribution
    plt.figure(figsize=(6,4))
    sns.countplot(x='label', data=train_df)
    plt.title('Class Distribution in Training Set')
    plt.xticks([0,1], ['Talking', 'Singing'])
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.show()

    # Log class counts
    class_counts = train_df['label'].value_counts()
    logger.info("Class Distribution:\n" + str(class_counts))

    # Get numeric columns only for correlation
    numeric_columns = train_df.select_dtypes(include=['float64', 'int64']).columns

    # Correlation Matrix for numeric columns only
    plt.figure(figsize=(20,18))
    corr = train_df[numeric_columns].corr()
    sns.heatmap(corr, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Correlation Matrix')
    plt.show()

# -----------------------------
# Handle Class Imbalance
# -----------------------------
def handle_class_imbalance(X, y, method='none'):
    """
    Handle class imbalance using specified method.
    
    Parameters:
        X (ndarray): Feature matrix.
        y (ndarray): Labels.
        method (str): Method to handle imbalance ('none' or 'smote').
    
    Returns:
        X_res (ndarray): Resampled feature matrix.
        y_res (ndarray): Resampled labels.
    """
    if method == 'none':
        logger.info("No class imbalance handling applied.")
        return X, y
    elif method == 'smote':
        # Count samples in each class
        class_counts = pd.Series(y).value_counts()
        min_samples = min(class_counts)

        if min_samples < 6:
            logger.warning(f"Not enough samples for SMOTE (minimum class has {min_samples} samples). Using k_neighbors={min_samples-1}")
            # Use k_neighbors = number of samples in minority class - 1
            smote = SMOTE(random_state=42, k_neighbors=min_samples-1)
        else:
            smote = SMOTE(random_state=42)  # Default k_neighbors=5

        logger.info("Applying SMOTE to handle class imbalance...")
        X_res, y_res = smote.fit_resample(X, y)
        logger.info("After SMOTE, class distribution:")
        logger.info(str(pd.Series(y_res).value_counts()))
        return X_res, y_res
    else:
        logger.warning(f"Unknown method '{method}'. No class imbalance handling applied.")
        return X, y

# -----------------------------
# Create Model Function
# -----------------------------
def create_fine_tuned_model(existing_model, input_dim, temporal=False, lstm_units=64, dropout_rate=0.3):
    """
    Create a new model for fine-tuning while preserving the original model.
    """
    if existing_model:
        if temporal:  # LSTM branch
            # For temporal data
            inputs = Input(shape=(None, input_dim))
            x = LSTM(lstm_units, return_sequences=False)(inputs)
            x = Dropout(dropout_rate)(x)
            
            # Copy architecture from existing model but create new layers
            for layer in existing_model.layers[1:]:  # Skip input layer
                config = layer.get_config()
                if isinstance(layer, Dense):
                    x = Dense(
                        units=layer.units,
                        activation=layer.activation,
                        name=f'dense_{len(layers)}'
                    )(x)
                elif isinstance(layer, Dropout):
                    x = Dropout(
                        rate=layer.rate,
                        name=f'dropout_{len(layers)}'
                    )(x)
            
            new_model = Model(inputs=inputs, outputs=x)
        else:
            # For non-temporal data
            inputs = Input(shape=(input_dim,), name='input_layer')
            x = inputs

            # Initialize new_model to hold the layers we'll build
            layers = []
            
            # Copy architecture and weights from existing model
            for layer in existing_model.layers:
                if isinstance(layer, Dense):
                    x = Dense(
                        units=layer.units,
                        activation=layer.activation,
                        name=f'dense_{len(layers)}'
                    )(x)
                    layers.append(x)
                elif isinstance(layer, Dropout):
                    x = Dropout(
                        rate=layer.rate,
                        name=f'dropout_{len(layers)}'
                    )(x)
                    layers.append(x)
            
            # Create the new model
            new_model = Model(inputs=inputs, outputs=x)
            
            # Copy weights for all layers except input
            for i, layer in enumerate(existing_model.layers):
                if isinstance(layer, (Dense, Dropout)) and i > 0:  # Skip input layer
                    new_model.layers[i].set_weights(layer.get_weights())
                    
                    # Freeze all layers except the last two
                    if i < len(existing_model.layers) - 2:
                        new_model.layers[i].trainable = False

        # Compile the model
        new_model.compile(
            optimizer='adam',
            loss='binary_crossentropy', 
            metrics=['accuracy']
        )
        
        logger.info("Created new fine-tuned model with knowledge from existing model")
        return new_model
    else:
        logger.error("No existing model provided for fine-tuning")
        return None

def train_model(model, X_train, y_train, class_weights=None, epochs=100, batch_size=32, validation_split=0.2):
    """
    Train the Neural Network with early stopping.
    
    Parameters:
        model (Sequential): Compiled Keras model.
        X_train (ndarray): Training features.
        y_train (ndarray): Training labels.
        class_weights (dict): Weights associated with classes.
        epochs (int): Maximum number of epochs.
        batch_size (int): Batch size for training.
        validation_split (float): Fraction of training data for validation.
    
    Returns:
        history (History): Training history.
    """
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stop],
        class_weight=class_weights,
        verbose=1
    )

    return history

# -----------------------------
# Evaluate Model Function
# -----------------------------
def evaluate_model(model, history, X_test, y_test):
    """
    Evaluate the trained model on the test set and plot training history.
    
    Parameters:
        model (Sequential): Trained Keras model.
        history (History): Training history.
        X_test (ndarray): Testing features.
        y_test (ndarray): Testing labels.
    """
    # Predictions
    y_pred_prob = model.predict(X_test).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=['Talking', 'Singing'])
    roc_auc = roc_auc_score(y_test, y_pred_prob)

    logger.info(f"\n--- Model Evaluation ---")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info("Confusion Matrix:")
    logger.info(str(conf_matrix))
    logger.info("Classification Report:")
    logger.info("\n" + class_report)
    logger.info(f"ROC-AUC Score: {roc_auc:.4f}")

    # Plotting Training History
    plt.figure(figsize=(12,5))

    # Accuracy Plot
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss Plot
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


def save_new_model(model, version_prefix='fine_tuned_model_v', models_dir='fine_tuned_models'):
    """
    Save only the new fine-tuned model, keeping original model intact.
    """
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_path = os.path.join(models_dir, f"{version_prefix}{timestamp}.h5")
    
    model.save(model_path)
    logger.info(f"Fine-tuned model saved as '{model_path}'")
    
    # Save latest version in separate directory
    latest_path = os.path.join(models_dir, f"{version_prefix}latest.h5")
    model.save(latest_path)
    logger.info(f"Fine-tuned model also saved as latest version: '{latest_path}'")
    existing_model = load_pretrained_model()
    
    # Create new fine-tuned model
    fine_tuned_model = create_fine_tuned_model(
        existing_model,
        input_dim=X_train.shape[1],
        temporal=temporal,
        lstm_units=64,
        dropout_rate=0.3
    )
    
    if temporal:
        # Reshape data for LSTM if temporal=True
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    
    # Train the fine-tuned model
    history = train_model(
        fine_tuned_model,  # Use fine_tuned_model instead of model
        X_train, y_train,
        class_weights=class_weights,
        epochs=100,
        batch_size=32,
        validation_split=0.2
    )
    
    # Evaluate the fine-tuned model
    evaluate_model(fine_tuned_model, history, X_test, y_test)
    
    # Save only the new model
    save_new_model(fine_tuned_model)
    
    logger.info("\nFine-tuning completed successfully!")
    logger.info("Original model preserved in 'models' directory")
    logger.info("Fine-tuned model saved in 'fine_tuned_models' directory")

def main():
    # Create directories
    create_directories()
    
    # First, split the wrong_audios.csv into train and test sets
    split_success = split_wrong_audios(
        wrong_audios_path='wrong_audios.csv',
        train_output_path='train_features_scaled.csv',
        test_output_path='test_features_scaled.csv',
        test_size=0.2,
        random_state=42
    )
    
    if not split_success:
        logger.error("Failed to split data. Cannot proceed.")
        return
    
    # Load pre-trained model
    pretrained_model = load_pretrained_model()
    if pretrained_model is None:
        logger.error("Cannot proceed without pre-trained model")
        return

    # Load the split data
    train_df, test_df = load_data()
    if train_df is None or test_df is None:
        logger.error("Cannot proceed without valid data")
        return
    
    # Extract features and labels
    feature_cols = [col for col in train_df.columns if col not in ['file_name', 'label']]
    X_train = train_df[feature_cols].values
    y_train = train_df['label'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['label'].values
    
    # Handle class imbalance
    X_train, y_train = handle_class_imbalance(X_train, y_train, method='smote')
    if X_train is None or y_train is None:
        logger.error("Cannot proceed after class imbalance handling failed")
        return
    
    # Set temporal flag to True to use LSTM
    temporal = True  # Change this to True
    
    # Create and train fine-tuned model 
    fine_tuned_model = create_fine_tuned_model(
        existing_model=pretrained_model,
        input_dim=X_train.shape[1],
        temporal=temporal,  # Pass temporal flag
        lstm_units=64,
        dropout_rate=0.3
    )
    
    # Reshape input data for LSTM
    if temporal:
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    
    # Train model
    history = train_model(
        fine_tuned_model,
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2
    )
    
    # Evaluate model
    evaluate_model(fine_tuned_model, history, X_test, y_test)
    
    # Save fine-tuned model
    save_new_model(fine_tuned_model)
    
    logger.info("Fine-tuning completed successfully!")

if __name__ == "__main__":
    main()
    main()
