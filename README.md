# AI Real-Time Sign Language Recognition

## Overview

AI Real-Time Sign Language Recognition is a computer vision and deep learning project that recognizes selected hand gestures in real time through a webcam.

The system uses MediaPipe to extract hand landmarks and an LSTM neural network to classify a sequence of hand movements into one of four supported gestures.

The project provides a Flask-based web interface where the webcam feed and recognized gesture are displayed in real time.

## Currently Supported Gestures

- HATE
- HELLO
- NO
- YES

## Features

- Real-time hand gesture recognition
- Webcam-based input
- Hand landmark extraction using MediaPipe
- LSTM-based sequence classification
- Support for four trained gestures
- Confidence-based prediction
- Flask web interface
- Custom dataset collection
- Custom model training
- Real-time prediction

## Technologies Used

- Python
- OpenCV
- MediaPipe
- TensorFlow
- Keras
- NumPy
- Flask
- LSTM
- HTML
- CSS

## System Architecture

Webcam
↓
OpenCV
↓
MediaPipe Hand Landmarks
↓
30-Frame Sequence
↓
LSTM Neural Network
↓
Gesture Prediction
↓
Flask Web Interface

## Model Details

### Input

The model receives a sequence of 30 consecutive frames.

For each frame, MediaPipe extracts 21 hand landmarks.

Each landmark contains:

- X coordinate
- Y coordinate
- Z coordinate

Therefore:

21 landmarks × 3 values = 63 features per frame

Final input shape:

30 × 63

### Model Architecture

The LSTM model contains:

- LSTM layer with 128 units
- Dropout layer with 0.3 dropout rate
- LSTM layer with 64 units
- Dropout layer with 0.3 dropout rate
- Dense layer with 64 units
- Dropout layer with 0.2 dropout rate
- Softmax output layer with 4 classes

Output classes:

- HATE
- HELLO
- NO
- YES

## Dataset

The dataset was collected using the project's webcam-based data collection script.

Each gesture contains:

- 200 sequences
- 30 frames per sequence
- 63 landmark features per frame

Total dataset:

4 gestures × 200 sequences = 800 sequences

Dataset split:

- Training samples: 640
- Testing samples: 160

The collected dataset is stored locally in:

backend/data/

The dataset is excluded from Git tracking using .gitignore.

## Training

The model was trained using TensorFlow and Keras.

Training configuration:

- Number of gestures: 4
- Total sequences: 800
- Training sequences: 640
- Testing sequences: 160
- Sequence length: 30 frames
- Features per frame: 63
- Epochs: 40
- Batch size: 32

The trained model is saved as:

model/sign_language_model.h5

## Test Accuracy

The trained model achieved:

99.37% test accuracy

Class mapping:

| Class | Gesture |
|-------|---------|
| 0 | HATE |
| 1 | HELLO |
| 2 | NO |
| 3 | YES |

## Project Workflow

### 1. Data Collection

The webcam captures hand movements for each gesture.

MediaPipe detects the hand and extracts 21 hand landmarks from every frame.

### 2. Feature Extraction

The X, Y and Z coordinates of the detected landmarks are converted into numerical features.

Each frame produces 63 features.

### 3. Sequence Creation

Thirty consecutive frames are grouped together to create one input sequence.

### 4. Model Training

The collected sequences are used to train the LSTM neural network.

### 5. Real-Time Prediction

During application execution, webcam frames are continuously processed.

The latest 30 frames are passed to the trained model.

### 6. Gesture Classification

The model predicts the most likely gesture and its confidence score.

### 7. Web Display

The recognized gesture and webcam feed are displayed through the Flask web application.

## Project Structure

AI-Real-Time-Sign-Language-Recognition/
│
├── App.py
├── README.md
├── .gitignore
│
├── backend/
│   ├── collect_data.py
│   └── train_model.py
│
├── model/
│   └── sign_language_model.h5
│
├── templates/
│   └── index.html
│
├── A Real-Time Automatic Translation of Text to Sign Language.pptx
├── SIGN LANG REPORT.pdf
├── Mediapipe
└── Tensorflow

## Installation

### 1. Clone the Repository

git clone https://github.com/sandeep171199/AI-Real-Time-Sign-Language-Recognition.git

cd AI-Real-Time-Sign-Language-Recognition

### 2. Create a Virtual Environment

Python 3.11 is recommended for this project.

py -3.11 -m venv .venv

### 3. Activate the Virtual Environment

On Windows PowerShell:

.\.venv\Scripts\Activate.ps1

### 4. Install Dependencies

pip install numpy==1.23.5
pip install opencv-python==4.7.0
pip install mediapipe==0.10.9
pip install tensorflow==2.12.0
pip install matplotlib==3.7.5
pip install opencv-contrib-python==4.7.0.72
pip install contourpy==1.0.7
pip install flask
pip install pyttsx3
pip install scikit-learn==1.3.2

## Run the Application

Activate the virtual environment and run:

python App.py

The Flask application will start locally.

Open the following address in your browser:

http://127.0.0.1:5000

Allow camera access if requested.

Show one of the supported gestures in front of the webcam.

## Data Collection

New training data can be collected using:

python backend/collect_data.py

The current data collection script supports:

- HATE
- HELLO
- NO
- YES

Each gesture is collected as 200 sequences.

Each sequence contains 30 frames.

The collected sequences are saved in:

backend/data/

## Model Training

After collecting the required dataset, train the model using:

python backend/train_model.py

The training script loads the four supported gesture classes and trains the LSTM model.

After training, the model is saved to:

model/sign_language_model.h5

## How It Works

The application performs the following steps:

1. Open the webcam using OpenCV.
2. Capture the current video frame.
3. Process the frame using MediaPipe.
4. Detect the hand.
5. Extract 21 hand landmarks.
6. Store the 63 landmark values for the current frame.
7. Build a sequence of 30 frames.
8. Send the sequence to the trained LSTM model.
9. Calculate the prediction confidence.
10. Display the recognized gesture when the confidence passes the configured threshold.

If no hand is detected, the current sequence is reset.

## Flask Web Interface

The project includes a Flask-based web interface.

The interface displays:

- Project title
- Webcam feed
- Real-time hand landmark detection
- Recognized gesture
- Prediction confidence
- Frame information

The HTML interface is located at:

templates/index.html

The Flask application is implemented in:

App.py

## Applications

Potential applications include:

- Sign language learning systems
- Gesture-controlled applications
- Human-computer interaction
- Accessibility-oriented software
- Educational demonstrations
- Computer vision research projects

## Future Improvements

Possible future improvements include:

- Adding more sign language gestures
- Increasing the size and diversity of the dataset
- Supporting two-hand gestures
- Improving recognition stability
- Adding text-to-speech output
- Improving the web interface
- Deploying the application as an online service
- Improving performance under different lighting conditions
- Supporting continuous sentence-level sign recognition

## Current Limitations

The current version has the following limitations:

- Only four gestures are supported.
- The system currently focuses on single-hand landmark detection.
- Recognition performance can be affected by lighting conditions.
- Camera quality can affect recognition performance.
- The model requires a sequence of 30 frames for prediction.
- The current system is primarily intended as a project demonstration.

## Project Contribution

This project demonstrates the implementation of a real-time sign language recognition system using computer vision and deep learning.

The project includes:

- Custom dataset collection
- Hand landmark extraction
- LSTM model training
- Real-time gesture prediction
- Flask web interface
- Model integration
- Application configuration

The project was developed and modified as a personal learning and development project.

## License

This project is intended for educational and learning purposes.

Please respect the license and attribution requirements of any third-party code, libraries, models, datasets, or other resources used in the project.