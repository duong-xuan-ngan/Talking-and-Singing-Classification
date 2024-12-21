import os
import logging
import warnings
import shutil
import glob
from spleeter.separator import Separator
import concurrent.futures
from tqdm import tqdm
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

def get_next_chunk_number(output_dir='split_vocal'):
    """
    Determines the next chunk file number based on existing files in the output directory.

    :param output_dir: Directory where split vocal files are saved.
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

def extract_vocals(input_wav, output_dir='split_vocal', output_filename='chunk1.wav'):
    """
    Extracts vocals from a WAV file using Spleeter, saves them as 'chunkN.wav' in the output directory,
    and deletes the accompaniment file.

    :param input_wav: Path to the input WAV file.
    :param output_dir: Directory to save the extracted vocals.
    :param output_filename: Desired filename for the extracted vocals (e.g., 'chunk1.wav').
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

    # Define the desired output path with 'chunkN.wav' name
    desired_output_path = os.path.join(output_dir, output_filename)

    try:
        # Move/rename 'vocals.wav' to 'chunkN.wav'
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

def process_split_audio_parallel(split_audio_dir='split_audio', split_vocal_dir='split_vocal', max_workers=4):
    """
    Processes each WAV file in the split_audio directory to extract vocals using parallel processing.

    :param split_audio_dir: Directory containing split audio WAV files.
    :param split_vocal_dir: Directory to save the extracted vocals.
    :param max_workers: Maximum number of threads to use.
    """
    # Check if the split_audio directory exists
    if not os.path.exists(split_audio_dir):
        print(f"The directory '{split_audio_dir}' does not exist. Please check the path.")
        logging.error(f"The directory '{split_audio_dir}' does not exist.")
        return

    # Find all WAV files in the split_audio directory
    wav_files = glob.glob(os.path.join(split_audio_dir, '*.wav'))

    if not wav_files:
        print(f"No WAV files found in '{split_audio_dir}'. Nothing to process.")
        logging.info(f"No WAV files found in '{split_audio_dir}'.")
        return

    print(f"Found {len(wav_files)} WAV file(s) in '{split_audio_dir}'. Starting vocal extraction with {max_workers} workers...")
    logging.info(f"Starting vocal extraction for {len(wav_files)} files with {max_workers} workers.")

    # Determine existing chunks to start numbering from
    existing_chunks = glob.glob(os.path.join(split_vocal_dir, 'chunk*.wav'))
    existing_numbers = [int(os.path.splitext(os.path.basename(f))[0].replace('chunk', '')) for f in existing_chunks if os.path.splitext(os.path.basename(f))[0].replace('chunk', '').isdigit()]
    next_chunk_number = max(existing_numbers, default=0) + 1

    # Function to process a single file with assigned chunk number
    def process_file(wav_file, chunk_number):
        output_filename = f"chunk{chunk_number}.wav"
        try:
            extract_vocals(wav_file, split_vocal_dir, output_filename)
        except Exception as e:
            logging.error(f"Error processing '{wav_file}': {e}")
            print(f"Error processing '{wav_file}': {e}")

    # Assign chunk numbers to each file
    tasks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for wav_file in wav_files:
            tasks.append(executor.submit(process_file, wav_file, next_chunk_number))
            next_chunk_number += 1

        # Use tqdm to display progress
        for future in tqdm(concurrent.futures.as_completed(tasks), total=len(tasks), desc="Extracting Vocals", unit="file"):
            pass  # Errors are handled in process_file

    print(f"\nVocal extraction completed. Extracted vocals are saved in the '{split_vocal_dir}' directory.")
    logging.info(f"Vocal extraction completed for all files. Vocals saved in '{split_vocal_dir}'.")

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
    parser = argparse.ArgumentParser(description='Extract vocals from WAV files in a directory using Spleeter.')
    parser.add_argument('--input_dir', type=str, default='split_audio', help='Directory containing input WAV files.')
    parser.add_argument('--output_dir', type=str, default='split_vocal', help='Directory to save extracted vocals.')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers.')
    parser.add_argument('--cleanup', type=str, default='', help='Directory to clean up after processing.')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Process split_audio to extract vocals using parallel processing
    process_split_audio_parallel(args.input_dir, args.output_dir, max_workers=args.workers)

    # Optional: Clean up temporary preprocessing directory if specified
    if args.cleanup:
        clean_up(args.cleanup)

    print(f"\nAll extracted vocals are available in the '{args.output_dir}' directory.")
    print(f"Refer to 'vocal_extraction.log' for detailed logs.")

if __name__ == "__main__":
    main()
