# AI Real-Time Sign Language Recognition

## Overview

AI Real-Time Sign Language Recognition is a computer vision and deep learning project that recognizes selected hand gestures in real time through a webcam.

The system uses MediaPipe to extract hand landmarks and an LSTM neural network to classify a sequence of hand movements into one of four supported gestures.

## Currently Supported Gestures

- HATE
- HELLO
- NO
- YES

## Features

- Real-time webcam-based gesture recognition
- Hand landmark detection using MediaPipe
- Sequence-based gesture classification using LSTM
- Four-class gesture recognition
- Confidence-based prediction
- Flask web interface for displaying the live camera feed
- Local data collection and model training pipeline

## Technologies Used

- Python
- OpenCV
- MediaPipe
- TensorFlow / Keras
- NumPy
- Flask
- LSTM
- HTML / CSS

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

The model uses a sequence of 30 frames as input.

Each frame contains:

- 21 MediaPipe hand landmarks
- 3 coordinates per landmark (X, Y, Z)
- 63 features per frame

Input shape:

30 frames × 63 features

The LSTM model contains:

- LSTM layer: 128 units
- Dropout: 0.3
- LSTM layer: 64 units
- Dropout: 0.3
- Dense layer: 64 units
- Dropout: 0.2
- Output layer: 4 classes with Softmax activation

## Dataset

The current model was trained using 800 gesture sequences.

| Gesture | Samples |
|---|---:|
| HATE | 200 |
| HELLO | 200 |
| NO | 200 |
| YES | 200 |
| **Total** | **800** |

Each sample contains 30 consecutive frames of hand landmark data.

The training dataset is stored locally and is excluded from the GitHub repository.

## Training

The dataset was divided into:

- Training samples: 640
- Testing samples: 160

The model was trained for 40 epochs with a batch size of 32.

### Test Accuracy

The current model achieved:

**99.37% test accuracy**

This result is based on the current dataset and train/test split used for the project.

## Project Workflow

1. Capture video from the webcam.
2. Detect a hand using MediaPipe.
3. Extract the 21 hand landmarks.
4. Convert each frame into 63 numerical features.
5. Collect 30 consecutive frames.
6. Pass the sequence to the trained LSTM model.
7. Calculate the prediction probabilities.
8. Display the recognized gesture and confidence in the application.

## Project Structure

AI-Real-Time-Sign-Language-Recognition/

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
├── App.py
├── README.md
├── .gitignore
├── A Real-Time Automatic Translation of Text to Sign Language.pptx
├── SIGN LANG REPORT.pdf
├── Mediapipe
└── Tensorflow

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/sandeep171199/AI-Real-Time-Sign-Language-Recognition.git
cd AI-Real-Time-Sign-Language-Recognition