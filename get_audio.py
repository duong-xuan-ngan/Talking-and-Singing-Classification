import os
import logging
import warnings
from pydub import AudioSegment
import yt_dlp  # Ensure yt_dlp is installed: pip install yt-dlp
import sounddevice as sd  # For recording audio
import soundfile as sf  # For saving recorded audio
import threading
import sys
import numpy as np
import shutil
import time

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

def download_youtube_audio(youtube_url, output_dir='downloaded_audio'):
    """
    Downloads audio from a YouTube URL using yt_dlp and saves it as 'audio.wav'.

    :param youtube_url: The YouTube video URL.
    :param output_dir: Directory to save the downloaded audio file.
    :return: Path to the downloaded WAV audio file.
    """
    os.makedirs(output_dir, exist_ok=True)
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_dir, 'audio.%(ext)s'),  # Fixed filename: audio.mp3 initially
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',  # Convert directly to WAV
            'preferredquality': '192',  # Quality parameter is ignored for WAV, but kept for consistency
        }],
        'quiet': True,
        'no_warnings': True,
        'restrictfilenames': True,  # Ensures filenames are safe (only ASCII characters)
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
            # After post-processing, the file will be 'audio.wav'
            downloaded_file = os.path.join(output_dir, 'audio.wav')
            if not os.path.isfile(downloaded_file):
                error_msg = f"Expected output file '{downloaded_file}' not found after download."
                print(error_msg)
                logging.error(error_msg)
                raise RuntimeError(error_msg)
            print(f"Downloaded and converted audio to '{downloaded_file}'.")
            logging.info(f"Downloaded and converted audio from '{youtube_url}' to '{downloaded_file}'.")
    except Exception as e:
        error_msg = f"Failed to download and convert audio from YouTube URL '{youtube_url}': {e}"
        print(error_msg)
        logging.error(error_msg)
        raise RuntimeError(error_msg)

    return downloaded_file


# Function to record audio from the microphone with enforced duration constraints
def record_audio_threaded(output_dir='recorded_audio', samplerate=44100, channels=2, min_duration=10, max_duration=600):
    """
    Records audio from the microphone with enforced minimum and maximum durations.

    :param output_dir: Directory to save the recorded audio file.
    :param samplerate: Sampling rate in Hz.
    :param channels: Number of audio channels.
    :param min_duration: Minimum recording duration in seconds.
    :param max_duration: Maximum recording duration in seconds.
    :return: Path to the recorded audio file.
    """
    os.makedirs(output_dir, exist_ok=True)
    recorded_file = os.path.join(output_dir, 'audio.wav')

    print(f"Recording... Minimum duration is {min_duration} seconds. Maximum duration is {max_duration} seconds.")
    print("Press Enter to stop recording after the minimum duration.")
    logging.info("Started recording audio from microphone.")

    # Initialize a list to store recorded data
    recorded_data = []

    # Define a callback function to capture audio data
    def callback(indata, frames, time_info, status):
        if status:
            print(f"Recording status: {status}", flush=True)
            logging.warning(f"Recording status: {status}")
        recorded_data.append(indata.copy())

    # Function to handle recording
    def record():
        try:
            with sf.SoundFile(recorded_file, mode='w', samplerate=samplerate,
                             channels=channels, subtype='PCM_16') as file:
                with sd.InputStream(samplerate=samplerate, channels=channels, callback=callback):
                    start_time = time.time()
                    while recording[0]:
                        current_time = time.time()
                        elapsed_time = current_time - start_time
                        if elapsed_time >= max_duration:
                            print("Maximum recording duration reached. Stopping recording.")
                            logging.info("Maximum recording duration reached. Stopping recording.")
                            recording[0] = False
                        if recorded_data:
                            file.write(np.concatenate(recorded_data))
                            recorded_data.clear()
                        time.sleep(0.1)
        except Exception as e:
            error_msg = f"Failed during recording: {e}"
            print(error_msg)
            logging.error(error_msg)
            recording[0] = False

    # Flag to control recording
    recording = [True]

    # Start the recording thread
    record_thread = threading.Thread(target=record)
    record_thread.start()

    # Start timing for minimum duration
    start_record_time = time.time()
    min_duration_reached = False

    # Wait for the user to press Enter to stop recording
    try:
        while recording[0]:
            if not min_duration_reached:
                elapsed = time.time() - start_record_time
                if elapsed >= min_duration:
                    min_duration_reached = True
                    print("You can now press Enter to stop recording.")
            if sys.platform.startswith('win'):
                import msvcrt
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key == b'\r':  # Enter key
                        if min_duration_reached:
                            break
                        else:
                            print(f"Please record for at least {min_duration} seconds.")
            else:
                import select
                if select.select([sys.stdin], [], [], 0)[0]:
                    input_char = sys.stdin.read(1)
                    if input_char == '\n':
                        if min_duration_reached:
                            break
                        else:
                            print(f"Please record for at least {min_duration} seconds.")
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    # Stop recording
    recording[0] = False
    record_thread.join()

    print(f"Recording stopped. Audio saved to '{recorded_file}'.")
    logging.info(f"Stopped recording audio. Saved to '{recorded_file}'.")
    return recorded_file

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
    output_wav = os.path.join(output_dir, 'audio.wav')  # Directly name the output as audio.wav


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

# Function to enforce duration constraints on WAV files
def enforce_duration_constraints(wav_path, min_duration=10, max_duration=600):
    """
    Ensures that the WAV file duration is between min_duration and max_duration.
    Trims the audio if it exceeds max_duration.
    Raises an error if the audio is shorter than min_duration.

    :param wav_path: Path to the WAV file.
    :param min_duration: Minimum allowed duration in seconds.
    :param max_duration: Maximum allowed duration in seconds.
    :return: Path to the processed WAV file (trimmed if necessary).
    """
    try:
        audio = AudioSegment.from_wav(wav_path)
        duration_sec = len(audio) / 1000  # pydub works in milliseconds

        if duration_sec < min_duration:
            error_msg = f"Audio duration {duration_sec:.2f}s is shorter than the minimum required {min_duration}s."
            print(error_msg)
            logging.error(error_msg)
            raise ValueError(error_msg)

        if duration_sec > max_duration:
            print(f"Audio duration {duration_sec:.2f}s exceeds the maximum of {max_duration}s. Trimming the audio.")
            logging.info(f"Trimming audio from {duration_sec:.2f}s to {max_duration}s.")
            trimmed_audio = audio[:max_duration * 1000]  # Trim to max_duration in ms
            trimmed_wav_path = os.path.join(os.path.dirname(wav_path), 'audio.wav')
            trimmed_audio.export(trimmed_wav_path, format='wav')
            print(f"Trimmed audio saved to '{trimmed_wav_path}'.")
            logging.info(f"Trimmed audio saved to '{trimmed_wav_path}'.")
            return trimmed_wav_path


        # If within constraints, return the original path
        return wav_path

    except Exception as e:
        error_msg = f"Failed to enforce duration constraints on '{wav_path}': {e}"
        print(error_msg)
        logging.error(error_msg)
        raise RuntimeError(error_msg)

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

# Function to process a single audio input (file, YouTube URL, or recording)
def process_audio_input(input_type, input_value, output_dir='processed_audio'):
    """
    Processes an audio input, ensuring it meets duration constraints:
    - Downloads and processes YouTube URLs.
    - Records, converts, and saves audio from the microphone.
    - Converts and saves local audio files.

    :param input_type: Type of input ('file', 'youtube', or 'record').
    :param input_value: The input value (file path, YouTube URL, or None for recording).
    :param output_dir: Directory to save the processed WAV file.
    """
    temp_dir = 'temp_processing'
    try:
        os.makedirs(temp_dir, exist_ok=True)

        # Step 1: Handle input based on its type
        if input_type == 'youtube':
            print("Downloading audio from YouTube URL...")
            logging.info(f"Starting download from YouTube URL: {input_value}")
            downloaded_audio = download_youtube_audio(input_value, output_dir=temp_dir)
            wav_path = convert_to_wav(downloaded_audio, output_dir=temp_dir)
        elif input_type == 'file':
            # Step 2: Convert to WAV if necessary
            if not input_value.lower().endswith('.wav'):
                print("Converting audio to WAV format...")
                wav_path = convert_to_wav(input_value, output_dir=temp_dir)
            else:
                wav_path = input_value
                print(f"Input file is already in WAV format: '{wav_path}'.")
                logging.info(f"Input file is already in WAV format: '{wav_path}'.")
        elif input_type == 'record':
            # Step 2: Record audio from the microphone with minimum and maximum duration
            wav_path = record_audio_threaded(output_dir=temp_dir, min_duration=10, max_duration=600)
            # Ensure the recorded file is in WAV format
            if not wav_path.lower().endswith('.wav'):
                print("Converting recorded audio to WAV format...")
                wav_path = convert_to_wav(wav_path, output_dir=temp_dir)
        else:
            raise ValueError("Invalid input type specified.")

        # Step 3: Enforce duration constraints
        wav_path = enforce_duration_constraints(wav_path, min_duration=10, max_duration=600)

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Define the final output path
        final_output_path = os.path.join(output_dir, 'audio.wav')
        shutil.move(wav_path, final_output_path)
        print(f"Processed audio saved to '{final_output_path}'.")
        logging.info(f"Processed audio saved to '{final_output_path}'.")


    except Exception as e:
        error_msg = f"An error occurred during processing: {e}"
        print(error_msg)
        logging.error(error_msg)
    finally:
        # Clean up temporary files
        clean_up(temp_dir)

    print(f"\nAll processed audio files are saved in the '{output_dir}' directory.")
    logging.info(f"All processed audio files are saved in the '{output_dir}' directory.")

# Main function to orchestrate the processing
def main():
    """
    Main function to prompt the user for input and process the audio accordingly.
    """
    print("Audio Processing Script")
    print("-----------------------")
    print("Choose the type of input:")
    print("1. Local Audio File")
    print("2. YouTube URL")
    print("3. Record Audio from Microphone")

    choice = input("Enter 1, 2, or 3: ").strip()

    if choice == '1':
        input_type = 'file'
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

    elif choice == '2':
        input_type = 'youtube'
        input_audio = input("Enter the YouTube video URL: ").strip()

        # Validate the input
        if not input_audio:
            print("No YouTube URL was entered. Exiting the program.")
            logging.warning("No YouTube URL was entered. Exiting the program.")
            return

        # Basic validation for YouTube URL
        if not (input_audio.startswith("http://") or input_audio.startswith("https://")):
            error_msg = "The provided URL does not seem valid. Please enter a valid YouTube URL."
            print(error_msg)
            logging.error(error_msg)
            return
    elif choice == '3':
        input_type = 'record'
        input_audio = None  # Not needed for recording
        print("You will now record audio. Minimum duration is 10 seconds and maximum is 10 minutes.")
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")
        logging.warning(f"User entered invalid choice: '{choice}'. Exiting the program.")
        return

    # Define the output directory for processed audio
    output_dir = 'processed_audio'

    # Process the audio input
    process_audio_input(input_type, input_audio, output_dir)

# Entry point of the script
if __name__ == "__main__":
    main()
