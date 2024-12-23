# Filename: prediction_api.py
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import logging
from flask import Flask, request, jsonify
from result import predict_single_batch  # Ensure result.py is accessible

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

app = Flask(__name__)

# Define the path to your model
MODEL_PATH = 'best_neural_network_model.h5'

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to handle prediction requests.
    Expects JSON data with feature values.
    """
    try:
        data = request.get_json()
        if not data:
            logging.error("No input data provided")
            return jsonify({'status': 'error', 'message': 'No input data provided'}), 400
        
        # Convert JSON data to DataFrame
        input_data = pd.DataFrame([data])
        
        # Ensure the input has the correct number of features
        expected_features = 70  # Update based on your model's expected features
        if input_data.shape[1] != expected_features:
            error_msg = f'Expected {expected_features} features, got {input_data.shape[1]}'
            logging.error(error_msg)
            return jsonify({'status': 'error', 'message': error_msg}), 400
        
        # Make prediction using result.py's function
        prediction, probability = predict_single_batch(MODEL_PATH, input_data)
        
        # Map prediction to label
        label = "Singing" if prediction == 1 else "Talking"
        
        # Calculate confidence
        confidence = f"{probability * 100:.2f}%" if prediction == 1 else f"{(1 - probability) * 100:.2f}%"
        
        # Prepare response
        response = {
            'status': 'success',
            'prediction': label,
            'confidence': confidence
        }
        
        logging.info(f"Prediction successful: {response}")
        return jsonify(response), 200
    
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Prediction error: {str(e)}'}), 500

if __name__ == '__main__':
    # Run the Flask app on a different port to avoid conflicts with the web app
    app.run(host='0.0.0.0', port=5001)
