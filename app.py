from flask import Flask, render_template, request, jsonify
import os
import logging
import requests
import pandas as pd
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

app = Flask(__name__)

# Define paths
PREDICTION_API_URL = 'http://localhost:5001/predict'  # URL of the Prediction API
TEMP_DIR = 'temp_processing'

def save_audio_blob(audio_data):
    """Save the recorded audio blob to a WAV file."""
    os.makedirs(TEMP_DIR, exist_ok=True)
    temp_path = os.path.join(TEMP_DIR, 'audio.wav')
    
    try:
        with open(temp_path, 'wb') as f:
            f.write(audio_data)
        return temp_path
    except Exception as e:
        logging.error(f"Error saving audio blob: {str(e)}")
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_type = request.form.get('input_type')
        
        if input_type == 'url':
            # Handle YouTube URL input
            audio_url = request.form.get('audio_url')
            if not audio_url:
                return jsonify({'status': 'error', 'message': 'No audio URL provided'}), 400
            os.environ['AUDIO_INPUT_TYPE'] = 'url'
            os.environ['AUDIO_INPUT_VALUE'] = audio_url
            
        elif input_type == 'record':
            # Handle the recorded audio blob
            if 'audio_data' not in request.files:
                return jsonify({'status': 'error', 'message': 'No audio data provided'}), 400
            
            audio_file = request.files['audio_data']
            try:
                temp_path = save_audio_blob(audio_file.read())
                os.environ['AUDIO_INPUT_TYPE'] = 'record'
                os.environ['AUDIO_INPUT_VALUE'] = temp_path
            except Exception as e:
                return jsonify({'status': 'error', 'message': f'Error saving audio: {str(e)}'}), 500
            
        else:
            return jsonify({'status': 'error', 'message': 'Invalid input type'}), 400

        # Process the audio data
        try:
            from data_process import main as process_main
            process_main()
        except Exception as e:
            logging.error(f"Error in data processing: {str(e)}")
            return jsonify({'status': 'error', 'message': f'Data processing error: {str(e)}'}), 500

        # Load the processed data and make prediction via Prediction API
        try:
            data = pd.read_csv('testing.csv')  # Ensure 'testing.csv' is generated correctly
            
            # Convert DataFrame to JSON
            input_json = data.to_dict(orient='records')[0]  # Assuming single record
            
            # Send POST request to Prediction API
            response = requests.post(PREDICTION_API_URL, json=input_json)
            
            # Log the response for debugging
            logging.info(f"Prediction API response status: {response.status_code}")
            logging.info(f"Prediction API response data: {response.json()}")
            
            if response.status_code == 200:
                prediction = response.json()
                return jsonify({
                    'status': 'success',
                    'prediction': prediction.get('prediction'),
                    'confidence': prediction.get('confidence')
                }), 200
            else:
                error_msg = response.json().get('message', 'Unknown error from Prediction API')
                return jsonify({'status': 'error', 'message': f'Prediction API error: {error_msg}'}), 500
            
        except Exception as e:
            logging.error(f"Error in prediction request: {str(e)}")
            return jsonify({'status': 'error', 'message': f'Prediction request error: {str(e)}'}), 500

    except Exception as e:
        logging.error(f"General error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
    
    finally:
        # Clean up temporary files
        if os.path.exists(TEMP_DIR):
            import shutil
            shutil.rmtree(TEMP_DIR)

if __name__ == '__main__':
    app.run(debug=True)
