import os
import subprocess

# Set the working directory to the location of data_process.py
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

scripts = ["split.py", "extract_vocal.py", "split_check.py", "csv_convert.py"]

def run_script(script_name):
    try:
        print(f"Running {script_name}...")
        subprocess.run(["python", script_name], check=True)
        print(f"Finished {script_name}.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script_name}: {e}")
        exit(1)

# Run each script in sequence
for script in scripts:
    run_script(script)
