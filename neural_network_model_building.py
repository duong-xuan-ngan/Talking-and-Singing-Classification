# Filename: neural_network_model_building.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.utils import class_weight
from sklearn.preprocessing import MinMaxScaler
import keras_tuner as kt
from imblearn.over_sampling import SMOTE  # For handling class imbalance
from tqdm import tqdm  # For progress bars

# Suppress TensorFlow warnings for cleaner output
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# -----------------------------
# Stage 1: Loading the Data
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
        print("Data loaded successfully.")
        print(f"Training Data Shape: {train_df.shape}")
        print(f"Testing Data Shape: {test_df.shape}")
        return train_df, test_df
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Ensure that 'train_features_scaled.csv' and 'test_features_scaled.csv' exist in the working directory.")
        exit()

# -----------------------------
# Stage 2: Exploratory Data Analysis (EDA)
# -----------------------------
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
    
    # Print class counts
    class_counts = train_df['label'].value_counts()
    print("Class Distribution:\n", class_counts)
    
    # Correlation Matrix
    plt.figure(figsize=(20,18))
    corr = train_df.corr()
    sns.heatmap(corr, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Correlation Matrix')
    plt.show()

# -----------------------------
# Stage 3: Handling Class Imbalance (Optional)
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
        print("No class imbalance handling applied.")
        return X, y
    elif method == 'smote':
        print("Applying SMOTE to handle class imbalance...")
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        print("After SMOTE, class distribution:")
        print(pd.Series(y_res).value_counts())
        return X_res, y_res
    else:
        print(f"Unknown method '{method}'. No class imbalance handling applied.")
        return X, y

# -----------------------------
# Stage 4: Building the Neural Network Model
# -----------------------------
def create_model(input_dim, dropout_rate=0.3, optimizer='adam'):
    """
    Define and compile the Neural Network architecture.
    
    Parameters:
        input_dim (int): Number of input features.
        dropout_rate (float): Dropout rate for regularization.
        optimizer (str): Optimizer for compiling the model.
    
    Returns:
        model (Sequential): Compiled Keras model.
    """
    model = Sequential()
    
    # Input Layer
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dropout(dropout_rate))
    
    # Hidden Layers
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(dropout_rate))
    
    # Output Layer
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
    return model

# -----------------------------
# Stage 5: Training the Model
# -----------------------------
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
# Stage 6: Evaluating the Model
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
    
    print(f"\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    
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

# -----------------------------
# Stage 7: Hyperparameter Tuning with Keras Tuner
# -----------------------------
def hyperparameter_tuning(X_train, y_train, input_dim, max_trials=10, executions_per_trial=2):
    """
    Perform hyperparameter tuning using Keras Tuner.
    
    Parameters:
        X_train (ndarray): Training features.
        y_train (ndarray): Training labels.
        input_dim (int): Number of input features.
        max_trials (int): Maximum number of trials.
        executions_per_trial (int): Number of executions per trial.
    
    Returns:
        best_model (Sequential): Best model found by the tuner.
        history_best (History): Training history of the best model.
    """
    def build_model_hp(hp):
        model = Sequential()
        
        # Input Layer
        model.add(Dense(
            units=hp.Int('units_input', min_value=64, max_value=256, step=32),
            activation='relu',
            input_dim=input_dim
        ))
        model.add(Dropout(rate=hp.Float('dropout_input', min_value=0.2, max_value=0.5, step=0.1)))
        
        # Hidden Layers
        for i in range(hp.Int('num_layers', 1, 3)):
            model.add(Dense(
                units=hp.Int(f'units_{i}', min_value=32, max_value=128, step=32),
                activation='relu'
            ))
            model.add(Dropout(rate=hp.Float(f'dropout_{i}', min_value=0.2, max_value=0.5, step=0.1)))
        
        # Output Layer
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile the model
        model.compile(
            optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop']),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    # Initialize the tuner
    tuner = kt.RandomSearch(
        build_model_hp,
        objective='val_accuracy',
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory='keras_tuner_dir',
        project_name='audio_classification_nn'
    )
    
    # Display the search space
    tuner.search_space_summary()
    
    # Define early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Perform the hyperparameter search
    tuner.search(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"""
    The hyperparameter search is complete. The optimal number of units in the input layer is {best_hps.get('units_input')},
    the optimizer is {best_hps.get('optimizer')}, and the number of hidden layers is {best_hps.get('num_layers')}.
    """)
    
    # Build the best model
    best_model = tuner.hypermodel.build(best_hps)
    
    # Train the best model
    history_best = best_model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )
    
    return best_model, history_best

# -----------------------------
# Stage 8: Final Evaluation of the Optimized Model
# -----------------------------
def final_evaluation(best_model, history_best, X_test, y_test):
    """
    Evaluate the optimized model on the test set.
    
    Parameters:
        best_model (Sequential): Optimized Keras model.
        history_best (History): Training history of the optimized model.
        X_test (ndarray): Testing features.
        y_test (ndarray): Testing labels.
    """
    # Predictions
    y_pred_prob_best = best_model.predict(X_test).ravel()
    y_pred_best = (y_pred_prob_best >= 0.5).astype(int)
    
    # Metrics
    accuracy_best = accuracy_score(y_test, y_pred_best)
    conf_matrix_best = confusion_matrix(y_test, y_pred_best)
    class_report_best = classification_report(y_test, y_pred_best, target_names=['Talking', 'Singing'])
    roc_auc_best = roc_auc_score(y_test, y_pred_prob_best)
    
    print(f"\n--- Optimized Model Evaluation ---")
    print(f"Accuracy: {accuracy_best:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix_best)
    print("Classification Report:")
    print(class_report_best)
    print(f"ROC-AUC Score: {roc_auc_best:.4f}")
    
    # Plotting Training History
    plt.figure(figsize=(12,5))
    
    # Accuracy Plot
    plt.subplot(1,2,1)
    plt.plot(history_best.history['accuracy'], label='Train Accuracy')
    plt.plot(history_best.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Optimized Model Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss Plot
    plt.subplot(1,2,2)
    plt.plot(history_best.history['loss'], label='Train Loss')
    plt.plot(history_best.history['val_loss'], label='Validation Loss')
    plt.title('Optimized Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# -----------------------------
# Stage 9: Saving the Final Model and Scaler
# -----------------------------
def save_model_and_scaler(model, scaler, scaler_path='minmax_scaler.save', model_path='best_neural_network_model.h5'):
    """
    Save the trained model and scaler for future use.
    
    Parameters:
        model (Sequential): Trained Keras model.
        scaler (MinMaxScaler): Fitted scaler object.
        scaler_path (str): Path to save the scaler.
        model_path (str): Path to save the model.
    """
    # Save the Neural Network model
    model.save(model_path)
    print(f"Neural Network model saved as '{model_path}'")
    
    # Save the scaler
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved as '{scaler_path}'")

# -----------------------------
# Stage 10: Main Function
# -----------------------------
def main():
    # -----------------------------
    # Load the Data
    # -----------------------------
    train_df, test_df = load_data()
    
    # -----------------------------
    # Perform EDA
    # -----------------------------
    perform_eda(train_df)
    
    # -----------------------------
    # Prepare Features and Labels
    # -----------------------------
    X_train = train_df.drop('label', axis=1).values
    y_train = train_df['label'].values
    X_test = test_df.drop('label', axis=1).values
    y_test = test_df['label'].values
    
    # -----------------------------
    # Handle Class Imbalance (Optional)
    # -----------------------------
    # Choose method: 'none' or 'smote'
    X_train, y_train = handle_class_imbalance(X_train, y_train, method='smote')
    
    # -----------------------------
    # Feature Scaling (Optional if already scaled)
    # -----------------------------
    # If you haven't scaled your data yet or rescaled after SMOTE, uncomment the following:
    # scaler = MinMaxScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    # save_model_and_scaler function expects scaler, so you need to pass it
    # However, assuming data is already scaled, we'll load the scaler
    scaler = joblib.load('minmax_scaler.save')  # Ensure this file exists
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    # -----------------------------
    # Compute Class Weights (Optional if not using SMOTE)
    # -----------------------------
    # If you used SMOTE, you might not need class weights. Otherwise, uncomment the following:
    # class_weights = class_weight.compute_class_weight(
    #     class_weight='balanced',
    #     classes=np.unique(y_train),
    #     y=y_train
    # )
    # class_weights = dict(enumerate(class_weights))
    # print("Class Weights:", class_weights)
    # Otherwise:
    class_weights = None  # Set to None if using SMOTE
    
    # -----------------------------
    # Build the Neural Network Model
    # -----------------------------
    input_dim = X_train.shape[1]
    model = create_model(input_dim=input_dim, dropout_rate=0.3, optimizer='adam')
    model.summary()
    
    # -----------------------------
    # Train the Model
    # -----------------------------
    history = train_model(
        model,
        X_train, y_train,
        class_weights=class_weights,
        epochs=100,
        batch_size=32,
        validation_split=0.2
    )
    
    # -----------------------------
    # Evaluate the Model
    # -----------------------------
    evaluate_model(model, history, X_test, y_test)
    
    # -----------------------------
    # Hyperparameter Tuning
    # -----------------------------
    best_model, history_best = hyperparameter_tuning(
        X_train, y_train,
        input_dim=input_dim,
        max_trials=10,
        executions_per_trial=2
    )
    
    # -----------------------------
    # Final Evaluation of Optimized Model
    # -----------------------------
    final_evaluation(best_model, history_best, X_test, y_test)
    
    # -----------------------------
    # Save the Final Model and Scaler
    # -----------------------------
    save_model_and_scaler(best_model, scaler, scaler_path='minmax_scaler.save', model_path='best_neural_network_model.h5')
    
    print("\nNeural Network model building process completed successfully!")

# -----------------------------
# Execute the Main Function
# -----------------------------
if __name__ == "__main__":
    main()
