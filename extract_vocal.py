import os
import logging
import warnings
import shutil
import glob
from spleeter.separator import Separator
import argparse

# Suppress TensorFlow and Spleeter logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Configure logging
logging.basicConfig(
    filename='vocal_extraction.log',
    filemode='a',  # Append mode
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def extract_vocals(input_wav, output_dir='extracted_vocal', output_filename='audio.wav'):
    """
    Extracts vocals from a WAV file using Spleeter, saves them as 'audio.wav' in the output directory,
    and deletes the accompaniment file.

    :param input_wav: Path to the input WAV file.
    :param output_dir: Directory to save the extracted vocals.
    :param output_filename: Desired filename for the extracted vocals (e.g., 'audio.wav').
    :return: Path to the extracted vocals WAV file.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize Spleeter's separator with a 2-stem model (vocals and accompaniment)
    separator = Separator('spleeter:2stems')  # or 'spleeter:4stems' for more detailed separation

    # Perform the separation
    try:
        # Separate the audio into vocals and accompaniment
        separator.separate_to_file(input_wav, output_dir, codec='wav', synchronous=True)
    except Exception as e:
        logging.error(f"Failed to extract vocals from '{input_wav}': {e}")
        raise RuntimeError(f"Failed to extract vocals from '{input_wav}': {e}")

    # Construct the path to the extracted vocals and accompaniment files
    base_filename = os.path.splitext(os.path.basename(input_wav))[0]
    vocals_wav_path = os.path.join(output_dir, base_filename, 'vocals.wav')
    accompaniment_wav_path = os.path.join(output_dir, base_filename, 'accompaniment.wav')

    # Verify that the vocals file exists
    if not os.path.exists(vocals_wav_path):
        error_msg = f"Vocals WAV file not found at '{vocals_wav_path}'."
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)

    # Define the desired output path with 'audio.wav' name
    desired_output_path = os.path.join(output_dir, output_filename)

    try:
        # Move/rename 'vocals.wav' to 'audio.wav'
        shutil.move(vocals_wav_path, desired_output_path)
        print(f"Extracted vocals saved to '{desired_output_path}'.")
        logging.info(f"Extracted vocals saved to '{desired_output_path}'.")
    except Exception as e:
        error_msg = f"Failed to move '{vocals_wav_path}' to '{desired_output_path}': {e}"
        print(error_msg)
        logging.error(error_msg)
        raise RuntimeError(error_msg)

    # Delete the accompaniment.wav file if it exists
    if os.path.exists(accompaniment_wav_path):
        try:
            os.remove(accompaniment_wav_path)
            print(f"Deleted accompaniment file '{accompaniment_wav_path}'.")
            logging.info(f"Deleted accompaniment file '{accompaniment_wav_path}'.")
        except Exception as e:
            logging.warning(f"Failed to delete accompaniment file '{accompaniment_wav_path}': {e}")
            print(f"Warning: Failed to delete accompaniment file '{accompaniment_wav_path}': {e}")

    # Optionally, remove the subdirectory if empty after deletion
    subdir = os.path.join(output_dir, base_filename)
    try:
        if not os.listdir(subdir):
            shutil.rmtree(subdir)
            print(f"Deleted empty directory '{subdir}'.")
            logging.info(f"Deleted empty directory '{subdir}'.")
    except Exception as e:
        logging.warning(f"Failed to delete directory '{subdir}': {e}")
        print(f"Warning: Failed to delete directory '{subdir}': {e}")

    return desired_output_path

def clean_up(temp_dir):
    """
    Removes the temporary directory used for processing.

    :param temp_dir: Directory to be removed.
    """
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary files in '{temp_dir}'.")
            logging.info(f"Cleaned up temporary files in '{temp_dir}'.")
        except Exception as e:
            print(f"Failed to clean up temporary files in '{temp_dir}': {e}")
            logging.error(f"Failed to clean up temporary files in '{temp_dir}': {e}")
    else:
        print(f"The directory '{temp_dir}' does not exist. Nothing to clean.")
        logging.info(f"The directory '{temp_dir}' does not exist. Nothing to clean.")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Extract vocals from a single WAV file using Spleeter.')
    parser.add_argument('--input_file', type=str, default='processed_audio/audio.wav',
                        help='Path to the input WAV file.')
    parser.add_argument('--output_dir', type=str, default='extracted_vocal',
                        help='Directory to save extracted vocals.')
    parser.add_argument('--cleanup', type=str, default='', 
                        help='Directory to clean up after processing.')
    return parser.parse_args()

def main():
    try:
        # Initialize the separator
        separator = Separator('spleeter:2stems')
        
        # Input and output paths
        input_path = os.path.join('processed_audio', 'audio.wav')
        output_dir = 'temp_separation'
        final_dir = 'extracted_vocal'
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(final_dir, exist_ok=True)
        
        # Separate vocals
        separator.separate_to_file(input_path, output_dir)
        
        # Move the vocals to final directory
        vocals_path = os.path.join(output_dir, 'audio', 'vocals.wav')
        final_path = os.path.join(final_dir, 'audio.wav')
        shutil.move(vocals_path, final_path)
        
        # Clean up
        shutil.rmtree(output_dir, ignore_errors=True)
        
    except Exception as e:
        print(f"Error in vocal extraction: {str(e)}")
        raise e

if __name__ == "__main__":
    main()