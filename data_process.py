import os
import logging
import warnings
from get_audio import process_youtube_video, clean_up, record_audio_threaded  # Add record_audio_threaded
from extract_vocal import main as extract_vocal_main
from csv_convert import process_audio_file
import shutil
import time  # Add this import

# Configure logging and suppress warnings
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings('ignore')
# Set the working directory and add ffmpeg to PATH
script_dir = os.path.dirname(os.path.abspath(__file__))
ffmpeg_path = os.path.join(script_dir, "ffmpeg", "bin")
if os.path.exists(ffmpeg_path):
    os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ.get("PATH", "")
os.chdir(script_dir)

def main():
    try:
        input_type = os.environ.get('AUDIO_INPUT_TYPE')
        input_value = os.environ.get('AUDIO_INPUT_VALUE')
        
        if not input_type or not input_value:
            raise ValueError("Missing input type or value")

        # Create all necessary directories
        directories = ['processed_audio', 'extracted_vocal', 'temp_processing']
        for dir_name in directories:
            os.makedirs(dir_name, exist_ok=True)
        
        time.sleep(1)  # Wait for directories to be created

        processed_audio_path = os.path.join('processed_audio', 'audio.wav')

        if input_type == 'record':
            # The audio file should already be in WAV format from record_audio_threaded
            shutil.copy2(input_value, processed_audio_path)
            
        elif input_type == 'url':
            # Process YouTube URL and ensure the file is created
            process_youtube_video(input_value, 'processed_audio')
            if not os.path.exists(processed_audio_path):
                raise FileNotFoundError(f"Failed to create audio file at {processed_audio_path}")

        # Verify file exists before proceeding
        if not os.path.exists(processed_audio_path):
            raise FileNotFoundError(f"Audio file not found at {processed_audio_path}")

        time.sleep(2)  # Wait before vocal extraction
        # Extract vocals
        extract_vocal_main()
        time.sleep(2)  # Wait after vocal extraction

        # Process the extracted vocal for prediction
        vocal_path = os.path.join('extracted_vocal', 'audio.wav')
        if not os.path.exists(vocal_path):
            raise FileNotFoundError("Vocal extraction failed")

        time.sleep(1)  # Wait before CSV conversion
        # Convert to features and save as CSV
        process_audio_file(vocal_path, output_csv='testing.csv')
        time.sleep(1)  # Wait after CSV conversion

    except Exception as e:
        logging.error(f"Error in data processing: {str(e)}")
        raise
    finally:
        # Clean up temporary files
        time.sleep(1)  # Wait before cleanup
        for temp_dir in ['temp_processing']:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
