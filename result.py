import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model = tf.keras.models.load_model('best_neural_network_model.keras')

# Load and preprocess the input data
input_file = 'test_features_scaled.csv'
data = pd.read_csv(input_file)

# Print the shape of the data to check the number of features
print("Original data shape:", data.shape)

# If the label is the last column, remove it
X = data.iloc[:, :-1].values  # Exclude the label column

# Print the shape after excluding the label
print("Shape after excluding label:", X.shape)

# Use MinMaxScaler (consistent with original preprocessing)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Ensure the input shape is correct for prediction
print("Scaled input shape:", X_scaled.shape)

# Make predictions
predictions_prob = model.predict(X_scaled)

# Convert probabilities to binary predictions
predictions = (predictions_prob >= 0.5).astype(int)

# Detailed prediction output
for i in range(len(predictions_prob)):
    prob = predictions_prob[i][0]  # Get the probability for the positive class
    pred = predictions[i][0]  # Get the binary prediction
    label = "Singing" if pred == 1 else "Talking"
    print(f"Sample {i + 1}:")
    print(f"  Probability of Singing: {prob:.4f}")
    print(f"  Predicted Label: {label}")

