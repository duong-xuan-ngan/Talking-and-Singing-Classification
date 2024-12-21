import os
import wave
import contextlib
import shutil
import glob
from pydub import AudioSegment
import sys

def get_wav_duration(filepath):
    """
    Returns the duration of a WAV file in seconds.
    """
    try:
        with contextlib.closing(wave.open(filepath, 'r')) as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)
            return duration
    except wave.Error as e:
        print(f"Error reading {filepath}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error with {filepath}: {e}")
        return None

def split_into_30s_chunks(input_wav, output_dir='temp_chunks', chunk_length_sec=30):
    """
    Splits a WAV file into multiple 30-second chunks.
    
    :param input_wav: Path to the input WAV file.
    :param output_dir: Directory to save the chunks.
    :param chunk_length_sec: Length of each chunk in seconds.
    :return: List of chunk file paths.
    """
    try:
        audio = AudioSegment.from_wav(input_wav)
    except Exception as e:
        print(f"Failed to load WAV file '{input_wav}': {e}")
        return []
    
    # Define chunk length in milliseconds
    chunk_length_ms = chunk_length_sec * 1000  # 30 seconds
    
    # Total length in ms
    total_length_ms = len(audio)
    
    # Calculate number of full 30s chunks
    num_chunks = total_length_ms // chunk_length_ms
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    chunk_files = []
    
    for i in range(int(num_chunks)):
        start_ms = i * chunk_length_ms
        end_ms = start_ms + chunk_length_ms
        chunk = audio[start_ms:end_ms]
        
        chunk_filename = f"chunk_temp_{i+1}.wav"
        chunk_path = os.path.join(output_dir, chunk_filename)
        
        try:
            chunk.export(chunk_path, format='wav')
            chunk_files.append(chunk_path)
            print(f"Created chunk: {chunk_path}")
        except Exception as e:
            print(f"Failed to export chunk '{chunk_path}': {e}")
    
    return chunk_files

def process_chunks(chunk_files, final_folder_path, starting_number=1):
    """
    Processes each chunk:
    - Deletes chunks not exactly 30 seconds.
    - Moves and renames chunks exactly 30 seconds to the final folder.
    
    :param chunk_files: List of chunk file paths.
    :param final_folder_path: Directory to save the final chunks.
    :param starting_number: The starting number for naming.
    :return: Updated starting_number after processing.
    """
    os.makedirs(final_folder_path, exist_ok=True)
    chunk_counter = starting_number
    
    for chunk_file in chunk_files:
        duration = get_wav_duration(chunk_file)
        
        if duration is None:
            print(f"Skipping file due to read error: {chunk_file}")
            continue
        
        # Check if duration is exactly 30 seconds (allowing small tolerance)
        if abs(duration - 30.0) < 0.1:
            # Define new filename
            new_filename = f"chunk{chunk_counter}.wav"
            new_filepath = os.path.join(final_folder_path, new_filename)
            
            # Ensure unique filename
            while os.path.exists(new_filepath):
                chunk_counter += 1
                new_filename = f"chunk{chunk_counter}.wav"
                new_filepath = os.path.join(final_folder_path, new_filename)
            
            try:
                shutil.move(chunk_file, new_filepath)
                print(f"Moved and renamed: {chunk_file} --> {new_filepath} (Duration: {duration:.2f} seconds)")
                chunk_counter += 1
            except Exception as e:
                print(f"Failed to move {chunk_file}: {e}")
        else:
            # Delete the chunk
            try:
                os.remove(chunk_file)
                print(f"Deleted: {chunk_file} (Duration: {duration:.2f} seconds)")
            except Exception as e:
                print(f"Failed to delete {chunk_file}: {e}")
    
    return chunk_counter

def process_wav_files(source_directory='split_vocal', final_folder_path='final'):
    """
    Processes WAV files in the source directory:
    - Splits each WAV file into 30-second chunks.
    - Deletes chunks not equal to 30 seconds.
    - Moves and renames chunks equal to 30 seconds to the final folder.
    
    Parameters:
    - source_directory (str): The path to the 'split_vocal' directory.
    - final_folder_path (str): The fixed path to the folder where qualifying WAV chunks will be moved.
    """
    if not os.path.isdir(source_directory):
        print(f"The specified path '{source_directory}' is not a directory or does not exist.")
        sys.exit(1)
    
    # Create the final folder if it doesn't exist
    os.makedirs(final_folder_path, exist_ok=True)
    
    # Initialize a counter for renaming
    # Find the next available chunk number
    existing_chunks = glob.glob(os.path.join(final_folder_path, 'chunk*.wav'))
    existing_numbers = [int(os.path.splitext(os.path.basename(f))[0].replace('chunk', '')) 
                        for f in existing_chunks 
                        if os.path.splitext(os.path.basename(f))[0].replace('chunk', '').isdigit()]
    chunk_counter = max(existing_numbers, default=0) + 1
    
    # Temporary directory to store chunks
    temp_chunks_dir = os.path.join(source_directory, 'temp_chunks')
    os.makedirs(temp_chunks_dir, exist_ok=True)
    
    # Iterate over files in the source directory
    for root, dirs, files in os.walk(source_directory):
        # Avoid processing files in the final folder or temp_chunks
        if os.path.abspath(root) in [os.path.abspath(final_folder_path), os.path.abspath(temp_chunks_dir)]:
            continue
        
        for file in files:
            if file.lower().endswith('.wav'):
                filepath = os.path.join(root, file)
                print(f"Processing file: {filepath}")
                # Split into 30s chunks
                chunk_files = split_into_30s_chunks(filepath, temp_chunks_dir, chunk_length_sec=30)
                # Process each chunk
                chunk_counter = process_chunks(chunk_files, final_folder_path, starting_number=chunk_counter)
                # Optionally, delete the original split_vocal file after processing
                try:
                    os.remove(filepath)
                    print(f"Deleted original file after splitting: {filepath}")
                except Exception as e:
                    print(f"Failed to delete original file '{filepath}': {e}")
    
    # Clean up temporary chunks directory
    try:
        shutil.rmtree(temp_chunks_dir)
        print(f"Cleaned up temporary chunks directory '{temp_chunks_dir}'.")
    except Exception as e:
        print(f"Failed to clean up temporary chunks directory '{temp_chunks_dir}': {e}")
    
    print("\nOperation Completed.")
    print(f"All 30-second WAV chunks have been moved to '{final_folder_path}'.")
    print(f"Chunks not exactly 30 seconds have been deleted.")

def main():
    # Define the input and output directories
    current_directory = os.getcwd()
    source_directory = os.path.join(current_directory, 'split_vocal')
    final_folder_path = os.path.join(current_directory, 'final')
    
    # Call the function with the fixed final folder path
    process_wav_files(source_directory, final_folder_path)

if __name__ == "__main__":
    main()
