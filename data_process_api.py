from flask import Flask, request, jsonify
import os
import logging
import sys  # Add missing import
from data_process import main as process_audio
import shutil

print("Script starting...")
# Configure logging to show more information
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Log system information
logger.debug(f"Python version: {sys.version}")
logger.debug(f"Current working directory: {os.getcwd()}")

app = Flask(__name__)

# Set host to 0.0.0.0 to make it accessible from outside
HOST = '0.0.0.0'
PORT = 8080

TEMP_DIR = 'temp_processing'

# Add a test route
@app.route('/', methods=['GET'])
def test():
    return jsonify({'status': 'success', 'message': 'API is running'}), 200

@app.route('/process', methods=['POST'])
def process():
    try:
        input_type = request.form.get('input_type')
        
        if input_type == 'url':
            audio_url = request.form.get('audio_url')
            if not audio_url:
                return jsonify({'status': 'error', 'message': 'No audio URL provided'}), 400
            os.environ['AUDIO_INPUT_TYPE'] = 'url'
            os.environ['AUDIO_INPUT_VALUE'] = audio_url
            
        elif input_type == 'record':
            if 'audio_data' not in request.files:
                return jsonify({'status': 'error', 'message': 'No audio data provided'}), 400
            
            audio_file = request.files['audio_data']
            temp_path = os.path.join(TEMP_DIR, 'audio.wav')
            os.makedirs(TEMP_DIR, exist_ok=True)
            
            audio_file.save(temp_path)
            os.environ['AUDIO_INPUT_TYPE'] = 'record'
            os.environ['AUDIO_INPUT_VALUE'] = temp_path
            
        else:
            return jsonify({'status': 'error', 'message': 'Invalid input type'}), 400

        # Process the audio
        try:
            process_audio()
            return jsonify({
                'status': 'success',
                'message': 'Audio processed successfully',
                'data_path': 'testing.csv'
            }), 200
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Processing error: {str(e)}'
            }), 500
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
        
    finally:
        # Clean up
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)

if __name__ == '__main__':
    try:
        # Create TEMP_DIR if it doesn't exist
        os.makedirs(TEMP_DIR, exist_ok=True)
        
        # Add startup message
        logger.info("Starting Processing API server...")
        logger.info(f"* Running on http://{HOST}:{PORT}")
        logger.info("* Debug mode: on")
        logger.info("* Press CTRL+C to quit")
        
        # Run the app with explicit host and port
        app.run(host=HOST, port=PORT, debug=True)
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
    finally:
        # Clean up temp directory
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
