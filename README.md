# Talking and Singing Classification Project

This project focuses on classifying audio data into either talking or singing categories. The pipeline includes data collection, audio processing, feature extraction, and model training. The application also includes a web interface for ease of use.

## Project Structure

The project is divided into three branches:

### 1. **Model Building Branch**

Contains the following Python files:

- `check_audios.py`: Validates audio files, makes predictions using a trained model, identifies misclassified samples, and updates a misclassification log.
- `neural_network_model_building.py`: Builds and trains a neural network for classification. Includes functionality for exploratory data analysis (EDA), class imbalance handling, hyperparameter tuning, and model evaluation.
- `neural_network_retraining_.py`: Handles the retraining of the neural network to improve accuracy. This includes backing up existing models, splitting mislabeled data for retraining, and fine-tuning the neural network.

#### Workflow in `check_audios.py`

- Loads a trained neural network model.
- Reads and verifies the structure of a test dataset from a CSV file.
- Predicts labels for audio files and identifies misclassified samples.
- Logs prediction probabilities and details for each file, including correct and incorrect predictions.
- Updates or creates a `wrong_audios.csv` file with misclassified samples.

#### Workflow in `neural_network_model_building.py`

- Loads preprocessed and scaled training and testing datasets.
- Performs exploratory data analysis (EDA) to analyze class distribution and feature correlations.
- Addresses class imbalance using methods like SMOTE or class weighting.
- Builds a neural network model using Keras with options for dropout and optimizer selection.
- Trains the model using early stopping to prevent overfitting.
- Evaluates the model on test data, generating metrics such as accuracy, confusion matrix, and ROC-AUC score.
- Performs hyperparameter tuning using Keras Tuner to optimize the model architecture.
- Saves the best model and training history plots to a results directory.

#### Workflow in `neural_network_retraining_.py`

- Creates necessary directories for models and backups.
- Backs up existing models with a timestamp.
- Splits mislabeled audio data into training and testing sets.
- Loads the pre-trained model and evaluates its compatibility with the new data.
- Handles class imbalance using SMOTE or other techniques.
- Fine-tunes the model by freezing certain layers and adding new dense layers.
- Retrains the model using early stopping and evaluates its performance.
- Saves the fine-tuned model and generates evaluation metrics, plots, and logs.

### 2. **Data Branch**

Contains the following Python file:

- `main.py`: Extracts audio features and preprocesses them for training. It includes steps for feature extraction, scaling, and saving the processed data.

#### Workflow in `main.py`

- Loads audio files from specified directories (`singing` and `talking`).
- Extracts features such as MFCC, Chroma, Spectral Contrast, Zero Crossing Rate, Spectral Centroid, and Spectral Roll-off.
- Scales features using `MinMaxScaler` and saves the scaler parameters.
- Splits data into training and testing sets while stratifying by labels.
- Saves processed data into CSV files (`train_features_scaled.csv` and `test_features_scaled.csv`).
- Logs errors and processes details for better debugging and traceability.

### 3. **Web Branch**

Contains the following Python files:

- `app.py`: Main application file that provides a web interface for users. Includes endpoints for processing audio input, interacting with the Prediction API, and displaying results.
- `csv_convert.py`: Converts processed audio data into CSV format for analysis.

#### Workflow in `csv_convert.py`

- Extracts features such as MFCC, Chroma, Spectral Contrast, Zero Crossing Rate, Spectral Centroid, and Spectral Roll-off from audio files.

- Handles missing values by filling them with zero.

- Initializes or loads a `MinMaxScaler` for feature scaling.

- Saves scaled data to a CSV file.

- Supports saving and loading scaler parameters for consistency.

- `data_process.py`: Handles data processing tasks for the web application.

- `extract_vocal.py`: Extracts vocals from audio files using Spleeter.

#### Workflow in `extract_vocal.py`

- Extracts vocals from audio files using the Spleeter library.

- Saves the extracted vocals as `audio.wav` in the specified directory.

- Deletes accompaniment files and cleans up temporary files.

- Ensures the output directory exists and handles any file or directory errors gracefully.

- `get_audio.py`: Downloads audio from YouTube videos and records audio from a microphone.

#### Workflow in `get_audio.py`

- Allows users to:
  - Record audio via microphone, enforcing minimum and maximum durations.
  - Download and convert audio from YouTube URLs.
  - Convert existing audio files to WAV format.
- Enforces duration constraints, ensuring that audio files meet the required length.
- Cleans up temporary files after processing to optimize storage usage.
- Saves processed audio files to a specified directory.

#### Workflow in `app.py`

- Accepts user input via a web interface, allowing uploads of recorded audio or YouTube URLs.
- Processes audio files, extracting necessary features and saving them as CSV.
- Communicates with the Prediction API to obtain classification results for the input audio.
- Handles errors during processing and prediction gracefully, providing meaningful feedback to users.

## Workflow Overview

### Data Collection

1. Download videos from YouTube (songs and talking videos) using `pydub` and save them in WAV format.
2. Extract audio using `ffmpeg`.

### Audio Preprocessing

1. Extract vocals from the audio using `Spleeter`.
2. Trim parts of the audio with volume levels below 20 dB.

### Feature Extraction

Extract the following features from the processed audio:

- Mel-Frequency Cepstral Coefficients (MFCC)
- Chroma Features
- Spectral Contrast
- Spectral Roll-off
- Zero Crossing Rate
- Spectral Centroid

### Data Scaling

- Scale the extracted features using `MinMaxScaler`.

### Model Training

- Train the neural network using the extracted and scaled features.
- Retrain the model as necessary to enhance performance.

## Requirements

### Model Branch and Prediction API

To run the **Model Branch** and **Prediction API**, you need to use **Python 3.11.6** and install dependencies from `requirements.txt`:

```plaintext
# Python 3.11.6

# Core dependencies
numpy==1.24.4
six==1.17.0
setuptools>=41.0.0

# Data processing
pandas==1.5.3
scipy==1.14.1
scikit-learn==1.6.0
imbalanced-learn==0.12.4
joblib==1.4.2
threadpoolctl==3.5.0

# TensorFlow and related
tensorflow==2.18.0
tensorflow-intel==2.18.0
tensorflow-io-gcs-filesystem==0.31.0
tensorboard==2.18.0
tensorboard-data-server==0.7.2
keras==3.7.0
keras-tuner==1.4.7
protobuf==5.29.1
grpcio==1.68.1
h5py==3.12.1
opt_einsum==3.4.0
astunparse==1.6.3
gast==0.6.0
google-pasta==0.2.0
absl-py==2.1.0
wrapt==1.17.0
termcolor==2.5.0
flatbuffers==24.3.25
ml-dtypes==0.4.1

# Visualization
matplotlib==3.9.3
seaborn==0.13.2
pillow==11.0.0
contourpy==1.3.1
cycler==0.12.1
fonttools==4.55.2
kiwisolver==1.4.7
pyparsing==3.2.0

# Utilities
tqdm==4.67.1
requests==2.32.3
urllib3==2.2.3
typing_extensions==4.12.2
packaging==24.2
python-dateutil==2.9.0.post0
pytz==2024.2
tzdata==2024.2

# Additional dependencies
Flask==2.2.5
Werkzeug==2.2.3
Jinja2==3.1.5
MarkupSafe==3.0.2
click==8.1.8
colorama==0.4.6
itsdangerous==2.2.0
blinker==1.9.0
certifi==2024.8.30
charset-normalizer==3.4.0
idna==3.10
```

Install these dependencies using:

```bash
pip install -r requirements.txt
```

### Web Branch

To run the **Web Branch**, you need to use **Python 3.11.6** in one terminal and **Python 3.10.0** in another terminal. Follow these steps:

1. In the first terminal (Python 3.11.6):

   - Install dependencies from `requirements.txt`:
     ```bash
     pip install -r requirements.txt
     ```
   - Ensure both services are running as per the instructions for Python 3.11.6 and Python 3.10.0 environments, as outlined in the **Web Branch** setup section.

2. In the second terminal (Python 3.10.0):

   - Install dependencies from `website.txt`:

     ```plaintext
     # Python 3.10.0

     # Core utilities
     six==1.17.0
     typing_extensions==4.12.2
     urllib3==2.3.0
     certifi==2024.12.14
     charset-normalizer==3.4.0
     idna==3.10
     packaging==24.3
     setuptools>=65.5.1
     wheel>=0.38.0

     # Scientific Computing
     numpy==1.22.4
     scipy==1.13.1
     pandas==1.3.5
     python-dateutil==2.9.0.post0
     pytz==2024.2

     # Machine Learning & Deep Learning
     tensorflow==2.8.0
     keras==2.8.0
     tensorboard==2.8.0
     scikit-learn==1.0.2
     protobuf==3.20.3

     # TensorFlow Dependencies
     absl-py==2.1.0
     astunparse==1.6.3
     flatbuffers==24.3.25
     gast==0.6.0
     google-auth==2.37.0
     google-pasta==0.2.0
     grpcio==1.68.1
     h5py==3.12.1
     keras-preprocessing==1.1.2
     opt_einsum==3.4.0
     tensorboard-data-server==0.6.1
     tensorboard-plugin-wit==1.8.1
     termcolor==2.5.0
     tf-estimator-nightly==2.8.0.dev2021122109
     wrapt==1.17.0

     # Audio Processing
     librosa==0.8.1
     audioread==3.0.1
     soundfile==0.12.1
     sounddevice==0.5.1
     pydub==0.25.1
     ffmpeg-python==0.2.0
     spleeter==2.3.2
     resampy==0.4.3

     # Web and API
     Flask==2.0.3
     Werkzeug==2.3.6
     Jinja2==3.1.5
     click==7.1.2
     itsdangerous==2.2.0
     blinker==1.9.0

     # Additional Dependencies
     joblib==1.4.2
     threadpoolctl==3.5.0
     pooch==1.8.2
     decorator==5.1.1
     numba==0.55.2
     llvmlite==0.38.1
     norbert==0.2.1
     typer==0.3.2
     yt-dlp==2024.12.13
     ```

   - Run the necessary services.
Additional Notes
Data Quality and Preprocessing

    Ensure the audio files are clear and free from background noise to improve model accuracy.
    Preprocessing steps such as trimming low-volume segments and vocal extraction are crucial for maintaining high-quality inputs.

Feature Selection

    The choice of features like MFCC, Chroma, and Zero Crossing Rate is based on their effectiveness in distinguishing between talking and singing.
    Additional features can be explored based on experimental results and model performance.

Model Tuning

    Hyperparameter tuning is a critical step for optimizing model performance. Use tools like Keras Tuner to experiment with different architectures and parameters.
    Regularly evaluate class imbalance to ensure the model is not biased towards one category.

Logging and Traceability

    Comprehensive logging in all scripts aids in debugging and ensures reproducibility.
    The use of wrong_audios.csv provides a systematic approach to iteratively improve model accuracy by focusing on misclassified samples.

Web Interface

    The web interface is designed to be user-friendly, supporting various input formats like recorded audio and YouTube links.
    Ensure the server running the interface is configured to handle file uploads securely and efficiently.

Environment Management

    Maintaining separate environments for the Model Branch and Web Branch ensures compatibility with dependencies and avoids conflicts.
    Use virtual environments (e.g., venv or conda) to manage dependencies and Python versions.

Scalability and Future Enhancements

    Consider adding support for additional audio classifications (e.g., music genres, accents, or languages).
    Implement real-time audio processing for applications such as live-streaming analysis or smart assistants.

Conclusion

The Talking and Singing Classification Project provides a comprehensive framework for classifying audio data into two distinct categories: talking and singing. By leveraging state-of-the-art audio processing techniques, feature extraction methods, and neural network architectures, this project achieves robust and accurate classification.

The inclusion of a web interface ensures accessibility and usability for non-technical users, making this project suitable for applications in entertainment, content moderation, and audio analysis.

This project is modular and scalable, with clearly defined workflows for each branch. Future enhancements can easily be integrated by expanding the feature set, optimizing the model, or improving the web interface. By adhering to the provided instructions and leveraging the outlined workflows, users can efficiently implement and extend the project to meet their specific needs.
