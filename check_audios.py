import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model = tf.keras.models.load_model('best_neural_network_model.keras')

# Load and preprocess the input data
input_file = 'test_features_scaled.csv'
data = pd.read_csv(input_file)

# Split features and true labels
X = data.iloc[:, :-1].values  # Exclude the label column
true_labels = data.iloc[:, -1].values  # Assuming the last column contains true labels

# Scale the features using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Make predictions
predictions_prob = model.predict(X_scaled)
predicted_labels = (predictions_prob >= 0.5).astype(int).flatten()

# Find incorrectly classified samples
incorrect_indices = np.where(predicted_labels != true_labels)[0]

# Print out the wrong predictions with sample numbers
print("Wrong predictions (Sample Number: Predicted Label -> True Label):")
for i in incorrect_indices:
    pred_label = "Singing" if predicted_labels[i] == 1 else "Talking"
    true_label = "Singing" if true_labels[i] == 1 else "Talking"
    print(f"Sample {i + 1}: The result is '{pred_label}' but actually is '{true_label}'")

# Extract incorrect samples and their details
wrong_audios = data.iloc[incorrect_indices].copy()
wrong_audios['Predicted_Label'] = ["Singing" if pred == 1 else "Talking" for pred in predicted_labels[incorrect_indices]]
wrong_audios['True_Label'] = ["Singing" if label == 1 else "Talking" for label in true_labels[incorrect_indices]]

# Save the wrong samples to a CSV file
wrong_audios.to_csv('wrong_audios.csv', index=False)

print("File 'wrong_audios.csv' has been created with incorrect predictions.")
