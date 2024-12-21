import os
import logging
import warnings
from pydub import AudioSegment
import glob
import time
import shutil

# Suppress unnecessary logs and warnings
logging.getLogger('pydub').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Configure logging
logging.basicConfig(
    filename='audio_processing.log',
    filemode='a',  # Append mode
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Function to convert any audio file to WAV format
def convert_to_wav(input_audio, output_dir='converted_wav'):
    """
    Converts an audio file to WAV format.

    :param input_audio: Path to the input audio file.
    :param output_dir: Directory to save the converted WAV file.
    :return: Path to the converted WAV file.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract the base filename without extension
    base_filename = os.path.splitext(os.path.basename(input_audio))[0]
    # Define the output WAV file path
    output_wav = os.path.join(output_dir, f"{base_filename}.wav")

    try:
        # Load the audio file
        audio = AudioSegment.from_file(input_audio)
        # Export as WAV
        audio.export(output_wav, format='wav')
        print(f"Converted '{input_audio}' to '{output_wav}'.")
        logging.info(f"Converted '{input_audio}' to '{output_wav}'.")
    except Exception as e:
        error_msg = f"Failed to convert '{input_audio}' to WAV: {e}"
        print(error_msg)
        logging.error(error_msg)
        raise RuntimeError(error_msg)

    return output_wav

# Function to determine the next chunk file number based on existing files
def get_next_chunk_number(output_dir='split_audio'):
    """
    Determines the next chunk file number based on existing files in the output directory.

    :param output_dir: Directory where split chunk files are saved.
    :return: The next available chunk number as an integer.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Find all files matching the pattern 'chunk*.wav'
    existing_files = glob.glob(os.path.join(output_dir, 'chunk*.wav'))

    max_number = 0

    # Iterate over existing files to find the highest number
    for file in existing_files:
        basename = os.path.basename(file)
        name, _ = os.path.splitext(basename)
        number_part = ''.join(filter(str.isdigit, name))
        if number_part.isdigit():
            number = int(number_part)
            if number > max_number:
                max_number = number

    return max_number + 1

# Function to split a WAV file into multiple chunks of specified length
def split_wav_file(input_wav, output_dir='split_audio', chunk_length_sec=600, start_number=1):
    """
    Splits a WAV file into multiple chunks of specified length with sequential naming.

    :param input_wav: Path to the input WAV file.
    :param output_dir: Directory where split files will be saved.
    :param chunk_length_sec: Length of each chunk in seconds.
    :param start_number: The starting number for naming the split files.
    :return: The next available chunk number after splitting.
    """
    try:
        # Load the WAV file
        audio = AudioSegment.from_wav(input_wav)
    except Exception as e:
        error_msg = f"Failed to load WAV file '{input_wav}': {e}"
        print(error_msg)
        logging.error(error_msg)
        raise RuntimeError(error_msg)

    # Calculate the total length in milliseconds
    total_length_ms = len(audio)

    # Calculate the number of chunks needed
    num_chunks = total_length_ms // (chunk_length_sec * 1000)
    if total_length_ms % (chunk_length_sec * 1000) != 0:
        num_chunks += 1

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    current_number = start_number

    # Split and export each chunk
    for i in range(num_chunks):
        start_ms = i * chunk_length_sec * 1000
        end_ms = start_ms + chunk_length_sec * 1000
        chunk = audio[start_ms:end_ms]

        output_filename = f'chunk{current_number}.wav'
        output_path = os.path.join(output_dir, output_filename)

        try:
            chunk.export(output_path, format='wav')
            print(f"Exported '{output_filename}'.")
            logging.info(f"Exported '{output_filename}'.")
        except Exception as e:
            error_msg = f"Failed to export '{output_filename}': {e}"
            print(error_msg)
            logging.error(error_msg)

        current_number += 1

    return current_number

# Function to clean up temporary files (if any are created)
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
            error_msg = f"Failed to clean up temporary files: {e}"
            print(error_msg)
            logging.error(error_msg)

# Function to process a single audio file
def process_audio_file(input_audio, split_output_dir='split_audio'):
    """
    Processes an audio file: converts to WAV (if necessary) and splits into chunks.

    :param input_audio: Path to the input audio file.
    :param split_output_dir: Directory to save all split chunk files.
    """
    temp_dir = 'temp_processing'
    try:
        os.makedirs(temp_dir, exist_ok=True)

        # Step 1: Convert to WAV if necessary
        if not input_audio.lower().endswith('.wav'):
            print("Converting audio to WAV format...")
            wav_path = convert_to_wav(input_audio, output_dir=temp_dir)
        else:
            wav_path = input_audio
            print(f"Input file is already in WAV format: '{wav_path}'.")
            logging.info(f"Input file is already in WAV format: '{wav_path}'.")

        time.sleep(1)  # Brief pause

        # Step 2: Determine the next available chunk number
        current_number = get_next_chunk_number(split_output_dir)
        print(f"Starting chunk numbering from {current_number}.")
        logging.info(f"Starting chunk numbering from {current_number}.")

        # Step 3: Split the WAV file into 10-minute (600 seconds) segments
        print("Splitting the WAV file into 10-minute segments...")
        current_number = split_wav_file(
            input_wav=wav_path,
            output_dir=split_output_dir,
            chunk_length_sec=600,
            start_number=current_number
        )
        print("Splitting completed successfully.")
        logging.info("Splitting completed successfully.")

    except Exception as e:
        error_msg = f"An error occurred during processing: {e}"
        print(error_msg)
        logging.error(error_msg)
    finally:
        # Clean up temporary files if conversion was done
        if not input_audio.lower().endswith('.wav'):
            clean_up(temp_dir)

    print(f"\nAll split audio files are saved in the '{split_output_dir}' directory.")
    logging.info(f"All split audio files are saved in the '{split_output_dir}' directory.")

# Main function to orchestrate the processing
def main():
    # Prompt the user to enter the path to the audio file
    input_audio = input("Enter the path to the audio file: ").strip()

    # Validate the input
    if not input_audio:
        print("No audio file path was entered. Exiting the program.")
        logging.warning("No audio file path was entered. Exiting the program.")
        return

    if not os.path.isfile(input_audio):
        error_msg = f"The file '{input_audio}' does not exist. Please provide a valid file path."
        print(error_msg)
        logging.error(error_msg)
        return

    # Define the output directory for split chunks
    split_output_dir = 'split_audio'

    # Process the audio file
    process_audio_file(input_audio, split_output_dir)

# Entry point of the script
if __name__ == "__main__":
    main()
