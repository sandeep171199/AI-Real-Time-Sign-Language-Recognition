# AI Real-Time Sign Language Recognition

A real-time sign language recognition system that uses computer vision and deep learning to recognize hand gestures through a webcam and display the predicted gesture in a web interface.

## Overview

AI Real-Time Sign Language Recognition is a computer vision and deep learning project designed to recognize selected hand gestures in real time.

The system captures live video from a webcam, detects hand landmarks using MediaPipe, extracts landmark features, and uses an LSTM neural network to classify a sequence of hand movements.

The recognized gesture is displayed through a Flask-based web application.

## Currently Supported Gestures

The current trained model recognizes four gestures:

- HATE
- HELLO
- NO
- YES

## Features

- Real-time webcam-based gesture recognition
- Hand landmark detection using MediaPipe
- Sequence-based gesture classification using LSTM
- Four-class gesture recognition
- Prediction confidence display
- Flask web interface
- Local gesture data collection
- LSTM model training pipeline
- TensorFlow/Keras trained model
- Real-time prediction through a browser

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

```text
Webcam
   |
   v
OpenCV
   |
   v
MediaPipe Hand Detection
   |
   v
Hand Landmark Extraction
   |
   v
63 Features per Frame
   |
   v
30-Frame Sequence
   |
   v
LSTM Neural Network
   |
   v
Gesture Prediction
   |
   v
Flask Web Interface

Model Details

The project uses an LSTM-based neural network to learn temporal patterns from sequences of hand landmarks.

Input Data

Each frame contains:

21 MediaPipe hand landmarks
3 coordinates per landmark
X, Y, and Z coordinates
63 features per frame

The model receives a sequence of 30 consecutive frames.

Input Shape
(30, 63)
LSTM Architecture
Input: (30, 63)
        |
        v
LSTM (128 units)
        |
        v
Dropout (0.3)
        |
        v
LSTM (64 units)
        |
        v
Dropout (0.3)
        |
        v
Dense (64 units, ReLU)
        |
        v
Dropout (0.2)
        |
        v
Dense (4 units, Softmax)
        |
        v
Gesture Prediction

## Dataset

The current model was trained using 800 gesture sequences.

Gesture	Samples
HATE	200
HELLO	200
NO	200
YES	200
Total	800

Each sample contains 30 consecutive frames of hand landmark data.

The training dataset is stored locally and is excluded from the GitHub repository using .gitignore.

##Training

The dataset was divided into:

Training samples: 640
Testing samples: 160

The model was trained using:

Epochs: 40
Batch size: 32
Test Accuracy

The current trained model achieved:

99.37% test accuracy

This result is based on the current locally collected dataset and the train/test split used for this project.

##Project Workflow
Start the webcam.
Capture live video frames.
Detect the user's hand using MediaPipe.
Extract 21 hand landmarks.
Convert the landmarks into 63 numerical features.
Collect 30 consecutive frames.
Pass the sequence to the trained LSTM model.
Calculate prediction probabilities.
Select the predicted gesture.
Display the gesture and confidence through the Flask web interface.

##Project Structure
AI-Real-Time-Sign-Language-Recognition/
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
├── App.py
├── README.md
├── .gitignore
├── A Real-Time Automatic Translation of Text to Sign Language.pptx
├── SIGN LANG REPORT.pdf
├── Mediapipe
└── Tensorflow

##Installation
1. Clone the Repository
git clone https://github.com/sandeep171199/AI-Real-Time-Sign-Language-Recognition.git

Move into the project directory:

cd AI-Real-Time-Sign-Language-Recognition
2. Create a Virtual Environment

Python 3.11 is recommended for the current dependency versions.

py -3.11 -m venv .venv
3. Activate the Virtual Environment
.\.venv\Scripts\Activate.ps1
4. Install Dependencies

##Install the compatible versions used during development:

pip install numpy==1.23.5
pip install opencv-python==4.7.0.72
pip install mediapipe==0.10.9
pip install tensorflow==2.12.0
pip install matplotlib==3.7.5
pip install opencv-contrib-python==4.7.0.72
pip install flask
pip install pyttsx3
pip install scikit-learn==1.3.2
Run the Application

After activating the virtual environment, run:

python App.py

The Flask application will start locally.

Open your browser and visit:

http://127.0.0.1:5000

Allow camera access when requested.

Show one of the supported gestures in front of the webcam to see the prediction.

##Data Collection

New training data can be collected using:

python backend\collect_data.py

The data collection script:

Opens the webcam.
Detects a hand using MediaPipe.
Extracts the 21 hand landmarks.
Converts them into 63 features.
Records 30-frame sequences.
Saves the sequences as NumPy .npy files.

The collected dataset is stored locally in:

backend/data/

The dataset is excluded from GitHub using .gitignore.

##Model Training

After collecting the required training data, run:

python backend\train_model.py

The training script loads the selected gesture data and trains the LSTM model.

The trained model is saved as:

model/sign_language_model.h5
How the Recognition Works

##The recognition pipeline works as follows:

Live Camera Feed
       |
       v
Hand Detection
       |
       v
MediaPipe Landmarks
       |
       v
21 Landmarks × 3 Coordinates
       |
       v
63 Features
       |
       v
30 Consecutive Frames
       |
       v
LSTM Model
       |
       v
Probability for Each Gesture
       |
       v
Predicted Gesture

The LSTM model processes the sequence of hand movements rather than relying on a single frame. This allows the model to learn temporal patterns within the gesture.

Flask Web Interface

The project includes a Flask-based web interface.

The interface provides:

Live webcam feed
Real-time gesture recognition
Prediction information
Browser-based access

The main Flask application is:

App.py

The HTML template is located at:

templates/index.html
Applications

Potential applications of the system include:

Educational sign language tools
Accessibility-focused applications
Gesture-controlled interfaces
Human-computer interaction
Real-time gesture recognition
Sign language learning systems

##Future Improvements

The project can be extended with:

More sign language gestures
Larger and more diverse datasets
Multiple-hand recognition
Prediction smoothing
Improved recognition under different lighting conditions
Continuous gesture recognition
Sentence formation
Text-to-speech output
Improved web interface
Cloud deployment
Current Limitations

The current implementation has four supported gestures and uses a locally collected dataset.

##Recognition performance can vary depending on:

Lighting conditions
Camera quality
Hand position
Background
Distance from the camera
Similarity between gestures

The reported 99.37% accuracy is based on the current dataset and test split and should not be interpreted as universal real-world accuracy.

Project Contribution

This project has been developed and customized as a hands-on implementation of real-time sign language recognition.

##The current implementation includes:

A customized four-class gesture dataset
MediaPipe-based hand landmark extraction
LSTM-based gesture classification
Model training pipeline
Trained TensorFlow/Keras model
Webcam-based real-time prediction
Flask web interface

##License

This project is intended for educational and portfolio purposes.

Please make sure the license and attribution requirements of any third-party code, libraries, models, datasets, or other resources used in the project are respected.