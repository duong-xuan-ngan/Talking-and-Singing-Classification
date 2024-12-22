import os
import logging
import warnings
from split import main as split_main
from extract_vocal import main as extract_vocal_main
from split_check import main as split_check_main
from csv_convert import main as csv_convert_main

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
        # Step 1: Run split.py
        print("\nStep 1: Running audio splitting...")
        split_main()

        # Step 2: Run extract_vocal.py
        print("\nStep 2: Running vocal extraction...")
        extract_vocal_main()

        # Step 3: Run split_check.py
        print("\nStep 3: Running split check...")
        split_check_main()

        # Step 4: Run csv_convert.py
        print("\nStep 4: Running feature extraction and CSV conversion...")
        csv_convert_main()

        print("\nAll processing steps completed successfully!")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        logging.error(f"Error in processing: {str(e)}")

if __name__ == "__main__":
    main()
