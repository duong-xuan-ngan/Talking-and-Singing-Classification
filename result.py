import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = load_model('best_neural_network_model.keras')

# Load and preprocess the input data
input_file = 'test_features_scaled.csv'
data = pd.read_csv(input_file)

# Print the shape of the data to check the number of features
print("Original data shape:", data.shape)  # Should print (n_samples, 71) if label is included

# If the label is the last column, remove it
X = data.iloc[:, :-1].values  # Exclude the label column

# Print the shape after excluding the label
print("Shape after excluding label:", X.shape)  # Should print (n_samples, 70)

# Apply scaling if necessary (use the same scaler as during training)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Ensure the input shape is correct for prediction
print("Scaled input shape:", X_scaled.shape)  # Should print (n_samples, 70)

# Make predictions
predictions = model.predict(X_scaled)

# Map predictions to labels
labels = ["talking", "singing"]
for i, pred in enumerate(predictions):
    predicted_label = labels[np.argmax(pred)]
    print(f"Sample {i + 1}: {predicted_label}")
