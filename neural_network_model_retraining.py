# Filename: neural_network_model_retraining.py

import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Input
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
    directories = ['models', 'model_backups', 'fine_tuned_models']
    for dir in directories:
        if not os.path.exists(dir):
            os.makedirs(dir)
            logger.info(f"Directory '{dir}' created.")
        else:
            logger.info(f"Directory '{dir}' already exists.")

# -----------------------------
# Backup Existing Model Function
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
        logger.info(f"Backup of the existing model saved to '{backup_path}'.")
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
        model (Model): Loaded Keras model or None if not found.
    """
    if os.path.exists(model_path):
        try:
            model = load_model(model_path)
            logger.info(f"Loaded existing model from '{model_path}'.")
            return model
        except Exception as e:
            logger.error(f"Error loading the existing model: {e}")
            logger.info("Proceeding to create a new model.")
            return None
    else:
        logger.info(f"No existing model found at '{model_path}'. Cannot proceed.")
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
        logger.info(f"Loaded data from '{wrong_audios_path}'.")
        logger.info(f"Total samples: {len(data)}.")
        logger.info(f"Columns: {', '.join(data.columns)}.")

        # Calculate split sizes
        total_samples = len(data)
        test_samples = int(total_samples * test_size)
        train_samples = total_samples - test_samples

        # Shuffle the data
        shuffled_data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)

        # Split the data
        train_data = shuffled_data.iloc[:train_samples]
        test_data = shuffled_data.iloc[train_samples:]

        # Save the split datasets
        train_data.to_csv(train_output_path, index=False)
        test_data.to_csv(test_output_path, index=False)

        # Log split information
        logger.info(f"\nData split complete:")
        logger.info(f"Training set ({(1 - test_size) * 100}%): {len(train_data)} samples.")
        logger.info(f"Testing set ({test_size * 100}%): {len(test_data)} samples.")
        logger.info(f"\nFiles saved:")
        logger.info(f"Training data: '{train_output_path}'.")
        logger.info(f"Testing data: '{test_output_path}'.")

        return True

    except Exception as e:
        logger.error(f"Error splitting data: {e}")
        return False

# -----------------------------
# Load Data Function
# -----------------------------
def load_data(train_path='train_features_scaled.csv', test_path='test_features_scaled.csv'):
    """
    Load preprocessed and scaled training and testing datasets.

    Parameters:
        train_path (str): Path to the training data CSV file.
        test_path (str): Path to the testing data CSV file.

    Returns:
        train_df (DataFrame): Training data.
        test_df (DataFrame): Testing data.
    """
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        if train_df.empty or test_df.empty:
            logger.error("One or both datasets are empty. Please ensure data is properly loaded before proceeding.")
            return None, None

        logger.info("Data loaded successfully.")
        logger.info(f"Training Data Shape: {train_df.shape}.")
        logger.info(f"Testing Data Shape: {test_df.shape}.")

        return train_df, test_df

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        logger.error("Please check the file format and contents.")
        return None, None

# -----------------------------
# Perform EDA Function
# -----------------------------
def perform_eda(train_df):
    """
    Perform EDA: Class distribution and feature correlations.

    Parameters:
        train_df (DataFrame): Training data.
    """
    # Class Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x='label', data=train_df)
    plt.title('Class Distribution in Training Set')
    plt.xticks([0, 1], ['Talking', 'Singing'])
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.show()

    # Log class counts
    class_counts = train_df['label'].value_counts()
    logger.info("Class Distribution:\n" + str(class_counts))

    # Get numeric columns only for correlation
    numeric_columns = train_df.select_dtypes(include=['float64', 'int64']).columns

    # Correlation Matrix for numeric columns only
    plt.figure(figsize=(20, 18))
    corr = train_df[numeric_columns].corr()
    sns.heatmap(corr, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Correlation Matrix')
    plt.show()

# -----------------------------
# Handle Class Imbalance Function
# -----------------------------
def handle_class_imbalance(X, y, method='none'):
    """
    Handle class imbalance using specified method.
    """
    if method == 'none':
        logger.info("No class imbalance handling applied.")
        return X, y
    elif method == 'smote':
        # Count samples in each class
        class_counts = pd.Series(y).value_counts()
        min_samples = min(class_counts)

        if min_samples < 2:
            logger.error(f"Not enough samples for SMOTE (minimum class has {min_samples} samples). SMOTE cannot be applied.")
            logger.info("Proceeding without applying SMOTE.")
            return X, y
        elif min_samples < 6:
            logger.warning(f"Not enough samples for SMOTE (minimum class has {min_samples} samples). Using k_neighbors={min_samples - 1}")
            # Use k_neighbors = number of samples in minority class - 1
            smote = SMOTE(random_state=42, k_neighbors=min_samples - 1)
        else:
            smote = SMOTE(random_state=42)  # Default k_neighbors=5

        logger.info("Applying SMOTE to handle class imbalance...")
        try:
            X_res, y_res = smote.fit_resample(X, y)
            logger.info("After SMOTE, class distribution:")
            logger.info(str(pd.Series(y_res).value_counts()))
            return X_res, y_res
        except Exception as e:
            logger.error(f"SMOTE failed: {e}")
            logger.info("Proceeding without applying SMOTE.")
            return X, y
    else:
        logger.warning(f"Unknown method '{method}'. No class imbalance handling applied.")
        return X, y

# Create Fine-Tuned Model Function
# -----------------------------
def create_fine_tuned_model(existing_model, input_dim, dropout_rate=0.3):
    """
    Fine-tunes the existing model by retraining only the last layer.
    
    Parameters:
        existing_model (Model): Pre-trained Keras model.
        input_dim (int): Number of input features.
        dropout_rate (float): Dropout rate for the last layer.
    
    Returns:
        new_model (Model): Compiled Keras model ready for training.
    """
    if not existing_model:
        logger.error("No existing model provided for fine-tuning.")
        return None

    # Freeze all layers except the last one
    for layer in existing_model.layers[:-1]:
        layer.trainable = False
        logger.debug(f"Layer '{layer.name}' frozen.")

    # Optionally, you can replace the last layer if needed
    # For example, to change activation or units
    # Here, we'll assume it's a binary classification with one unit and sigmoid activation
    last_layer = existing_model.layers[-1]
    if not isinstance(last_layer, Dense) or last_layer.units != 1 or last_layer.activation != tf.keras.activations.sigmoid:
        # Replace the last layer if it doesn't match the desired configuration
        x = existing_model.layers[-2].output
        x = Dense(1, activation='sigmoid', name='output_layer')(x)
        new_model = Model(inputs=existing_model.input, outputs=x)
        logger.info("Replaced the last layer with a new Dense layer.")
    else:
        new_model = existing_model
        logger.info("Last layer is already suitable for retraining.")

    # Compile the model
    new_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return new_model


# -----------------------------
# Train Model Function
# -----------------------------
def train_model(model, X_train, y_train, class_weights=None, epochs=100, batch_size=32, validation_split=0.2):
    """
    Train the Neural Network with early stopping.

    Parameters:
        model (Model): Compiled Keras model.
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
        model (Model): Trained Keras model.
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
    plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# -----------------------------
# Save New Model Function
# -----------------------------
def save_new_model(model, version_prefix='fine_tuned_model_v', models_dir='fine_tuned_models'):
    """
    Save the fine-tuned model with a timestamp and as the latest version.

    Parameters:
        model (Model): Keras model to save.
        version_prefix (str): Prefix for the model filename.
        models_dir (str): Directory to save the models.
    """
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        logger.info(f"Directory '{models_dir}' created for saving fine-tuned models.")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_path = os.path.join(models_dir, f"{version_prefix}{timestamp}.h5")
    latest_path = os.path.join(models_dir, f"{version_prefix}latest.h5")

    model.save(model_path)
    model.save(latest_path)

    logger.info(f"Fine-tuned model saved as '{model_path}' and '{latest_path}'.")

# -----------------------------
# Main Function
# -----------------------------
def main():
    # Step 1: Create necessary directories
    create_directories()

    # Step 2: Backup existing model
    backup_existing_model(
        model_path='models/best_neural_network_model.h5',
        backup_dir='model_backups'
    )

    # Step 3: Split the wrong_audios.csv into train and test sets
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

    # Step 4: Load pre-trained model
    pretrained_model = load_pretrained_model(model_path='models/best_neural_network_model.h5')
    # If no pre-trained model exists, decide whether to proceed or exit
    if pretrained_model is None:
        logger.error("Cannot proceed without a pre-trained model.")
        return

    # Step 5: Load the split data
    train_df, test_df = load_data(
        train_path='train_features_scaled.csv',
        test_path='test_features_scaled.csv'
    )
    if train_df is None or test_df is None:
        logger.error("Cannot proceed without valid data.")
        return

    # Step 6: Extract features and labels
    feature_cols = [col for col in train_df.columns if col not in ['file_name', 'label']]
    X_train = train_df[feature_cols].values
    y_train = train_df['label'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['label'].values

    # Step 7: Handle class imbalance
    X_train, y_train = handle_class_imbalance(X_train, y_train, method='smote')

    # Step 8: Define fine-tuning parameters
    temporal = False  # Set to False as data is not temporal
    trainable_layers = 2  # Number of top pre-trained layers to keep trainable

    # Step 9: Create fine-tuned model
    fine_tuned_model = create_fine_tuned_model(
        existing_model=pretrained_model,
        input_dim=X_train.shape[1],
        temporal=temporal,
        lstm_units=64,  # Not used since temporal=False
        dropout_rate=0.3,
        trainable_layers=trainable_layers
    )

    if fine_tuned_model is None:
        logger.error("Failed to create fine-tuned model.")
        return

    # Step 10: Train the model
    history = train_model(
        model=fine_tuned_model,
        X_train=X_train,
        y_train=y_train,
        class_weights=None,  # Already handled by SMOTE
        epochs=100,
        batch_size=32,
        validation_split=0.2
    )

    # Step 11: Evaluate the model
    evaluate_model(
        model=fine_tuned_model,
        history=history,
        X_test=X_test,
        y_test=y_test
    )

    # Step 12: Save the fine-tuned model
    save_new_model(
        model=fine_tuned_model,
        version_prefix='fine_tuned_model_v',
        models_dir='fine_tuned_models'
    )

    logger.info("Fine-tuning completed successfully!")

if __name__ == "__main__":
    main()
