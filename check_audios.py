import numpy as np
import pandas as pd
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('model_results/best_neural_network_model_20250103_182931.h5')

# Load the input test data
input_file = 'testing.csv'  # Use the test dataset
data = pd.read_csv(input_file)

# Ensure 'file_name' and 'label' columns exist
required_columns = ['file_name', 'label']
for col in required_columns:
    if col not in data.columns:
        raise KeyError(f"Missing required column: '{col}'")

# Define feature columns explicitly and ensure correct order
feature_columns = [col for col in data.columns if col not in ['label', 'file_name']]

# Extract features and labels
X = data[feature_columns].values
y_true = data['label'].values  # Extract true labels
file_names = data['file_name'].values  # Extract file names

# Observe what is the X
print("X is: ")
print(X)

# Print the shape after excluding label
print("Shape after excluding 'file_name' and 'label':", X.shape)
print("Labels shape:", y_true.shape)

# Ensure the input shape is correct for prediction
print("Input shape for prediction:", X.shape)

# Make predictions (No scaling applied since data is already scaled)
predictions_prob = model.predict(X, batch_size=1, verbose=1)

# Convert probabilities to binary predictions
predictions = (predictions_prob >= 0.5).astype(int).flatten()

# Detailed prediction output
for i in range(len(predictions_prob)):
    prob = predictions_prob[i][0]  # Probability of 'Singing'
    pred = predictions[i]  # Binary prediction
    label = "Singing" if pred == 1 else "Talking"
    actual_label = "Singing" if y_true[i] == 1 else "Talking"
    print(f"File '{file_names[i]}':")
    print(f"  Probability of Singing: {prob:.4f}")
    print(f"  Predicted Label: {label} (Actual: {actual_label})")

# Find misclassified samples
incorrect_indices = np.where(predictions != y_true)[0]

# Print out the misclassifications
print("\nWrong predictions (File Name: Predicted Label -> True Label):")
for i in incorrect_indices:
    pred_label = "Singing" if predictions[i] == 1 else "Talking"
    actual_label = "Singing" if y_true[i] == 1 else "Talking"
    prob = predictions_prob[i][0]  # Get the probability
    print(f"File '{file_names[i]}': The result is '{pred_label}' (probability: {prob:.4f}) but actually is '{actual_label}'")

# Optionally, save misclassified samples to a CSV file
if len(incorrect_indices) > 0:
    # Get current misclassifications
    wrong_audios = data.iloc[incorrect_indices].copy()
    
    # Reorder columns: features, then label, then file_name
    column_order = feature_columns + ['label', 'file_name']
    wrong_audios = wrong_audios[column_order]
    
    try:
        # Try to read existing wrong_audios.csv
        existing_wrong_audios = pd.read_csv('wrong_audios.csv')
        print("\nFound existing wrong_audios.csv with", len(existing_wrong_audios), "samples")
        
        # Ensure existing data has the correct column order
        existing_wrong_audios = existing_wrong_audios[column_order]
        
        # Combine existing and new misclassifications
        combined_wrong_audios = pd.concat([existing_wrong_audios, wrong_audios], axis=0)
        
        # Remove duplicates based on file_name to keep latest prediction
        combined_wrong_audios = combined_wrong_audios.drop_duplicates(subset=['file_name'], keep='last')
        
        # Ensure final data has correct column order
        combined_wrong_audios = combined_wrong_audios[column_order]
        
        # Save the updated dataset
        combined_wrong_audios.to_csv('wrong_audios.csv', index=False)
        print(f"Updated wrong_audios.csv - Total misclassified samples: {len(combined_wrong_audios)}")
        print(f"Added {len(wrong_audios)} new misclassifications")
        
    except FileNotFoundError:
        # If wrong_audios.csv doesn't exist, create it with current misclassifications
        wrong_audios.to_csv('wrong_audios.csv', index=False)
        print(f"\nCreated new wrong_audios.csv with {len(wrong_audios)} misclassified samples")
    
    print("Format: features + label + file_name")
else:
    print("\nNo misclassifications found in this run.")
