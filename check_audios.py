import numpy as np
import pandas as pd
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('best_neural_network_model.h5')

# Load the input test data
input_file = 'testing.csv'  # Use the test dataset
data = pd.read_csv(input_file)

# Ensure 'file_name' and 'label' columns exist
required_columns = ['file_name', 'label']
for col in required_columns:
    if col not in data.columns:
        raise KeyError(f"Missing required column: '{col}'")

# Define feature columns explicitly
feature_columns = [col for col in data.columns if col not in ['file_name', 'label']]

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
predictions_prob = model.predict(X, batch_size=32, verbose=1)

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
    print(f"File '{file_names[i]}': The result is '{pred_label}' but actually is '{actual_label}'")

# Optionally, save misclassified samples to a CSV file
if len(incorrect_indices) > 0:
    wrong_audios = data.iloc[incorrect_indices].copy()
    wrong_audios['Predicted_Label'] = ["Singing" if pred == 1 else "Talking" for pred in predictions[incorrect_indices]]
    wrong_audios['True_Label'] = ["Singing" if label == 1 else "Talking" for label in y_true[incorrect_indices]]
    wrong_audios.to_csv('wrong_audios.csv', index=False)
    print("\nFile 'wrong_audios.csv' has been created with incorrect predictions.")
else:
    print("\nNo misclassifications found.")
