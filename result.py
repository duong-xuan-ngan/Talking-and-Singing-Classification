# result.py
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import os

logging.basicConfig(level=logging.INFO)

def predict_single_batch(model_path, input_data):
    """
    Make prediction on a single audio sample using a trained neural network model.
    
    Parameters:
    model_path (str): Path to the saved model
    input_data (pandas.DataFrame): Single row of feature data
    
    Returns:
    tuple: (prediction, prediction_probability)
    """
    try:
        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
            
        # Load the model with custom configuration to avoid batch_shape error
        model = tf.keras.models.load_model(
            model_path,
            compile=False,  # Don't compile the model
            custom_objects=None
        )
        
        # Prepare input - reshape to add batch dimension
        X = input_data.values.reshape(1, -1)  # Reshape to (1, n_features)
        X = np.array(X, dtype=np.float32)  # Ensure float32 type
        
        # Log shape for debugging
        logging.info(f"Input shape for prediction: {X.shape}")
        
        # Make prediction without batch_size parameter
        prediction_prob = model.predict(
            X,
            verbose=0
        )
        
        # Convert probability to binary prediction
        prediction = (prediction_prob >= 0.5).astype(int)[0][0]
        
        # Get final probability value
        prob = prediction_prob[0][0]
        label = "Singing" if prediction == 1 else "Talking"
        
        print(f"\nPrediction Results:")
        print(f"Probability of Singing: {prob:.4f}")
        print(f"Predicted Label: {label}")
        
        return prediction, prob

    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        raise
