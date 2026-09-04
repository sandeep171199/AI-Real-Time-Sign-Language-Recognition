AI Real-Time Sign Language Recognition

A real-time sign language recognition system that uses computer vision and deep learning to recognize hand gestures through a webcam and display the predicted gesture in a web interface.

Overview

This project captures live webcam video, detects hand landmarks using MediaPipe, extracts numerical landmark features, and uses an LSTM-based deep learning model to classify sign language gestures.

The recognized gesture is displayed in real time through a Flask web application.

Currently Supported Gestures

The current trained model recognizes 4 gestures:

HATE
HELLO
NO
YES
System Workflow

Webcam
↓
Hand Detection
↓
MediaPipe Hand Landmarks
↓
Landmark Feature Extraction
↓
30-Frame Sequence
↓
LSTM Deep Learning Model
↓
Gesture Prediction
↓
Flask Web Interface

Features
Real-time webcam-based gesture recognition
Hand landmark detection using MediaPipe
21 hand landmarks per frame
63 numerical features per frame
30-frame temporal sequences
LSTM-based gesture classification
4-class gesture recognition
Real-time prediction confidence
Browser-based interface using Flask
Locally collected training data
Trained TensorFlow/Keras model
Technologies Used
Python
OpenCV
MediaPipe
TensorFlow / Keras
NumPy
Flask
HTML / CSS
LSTM Neural Networks
Model Details

The model uses a sequence-based LSTM architecture to learn temporal patterns from hand landmark movements.

Input
Sequence length: 30 frames
Features per frame: 63
Hand landmarks: 21
Coordinates per landmark: X, Y, Z
Architecture

Input: (30, 63)
↓
LSTM (128)
↓
Dropout
↓
LSTM (64)
↓
Dropout
↓
Dense (64, ReLU)
↓
Dropout
↓
Dense (4, Softmax)

Training

The current dataset contains:

HATE: 200 samples
HELLO: 200 samples
NO: 200 samples
YES: 200 samples

Total: 800 sequences

The dataset is kept locally and is not included in the GitHub repository.

Test Accuracy

The current trained model achieved approximately:

99.37% test accuracy

Note: This accuracy is based on the current locally collected dataset and test split.

Project Structure

AI-Real-Time-Sign-Language-Recognition/
|
├── App.py
├── README.md
├── .gitignore
|
├── backend/
| ├── collect_data.py
| └── train_model.py
|
├── model/
| └── sign_language_model.h5
|
└── templates/
└── index.html

Installation
1. Clone the repository

git clone https://github.com/sandeep171199/AI-Real-Time-Sign-Language-Recognition.git

cd AI-Real-Time-Sign-Language-Recognition

2. Create a virtual environment

python -m venv .venv

Activate it on Windows:

.venv\Scripts\Activate.ps1

3. Install dependencies

Install the required Python packages:

pip install numpy==1.23.5

pip install opencv-python==4.7.0

pip install mediapipe==0.10.9

pip install tensorflow==2.12.0

pip install flask

pip install pyttsx3

pip install scikit-learn==1.3.2

Run the Application

Start the Flask application:

python App.py

Open your browser and visit:

http://127.0.0.1:5000

Allow camera access and show one of the supported gestures in front of the webcam.

Data Collection

Training data can be collected using:

python backend/collect_data.py

The collector extracts hand landmarks from webcam frames and stores each 30-frame sequence as a NumPy .npy file.

The training dataset is intentionally excluded from GitHub using .gitignore.

Model Training

To train the model using the collected local dataset:

python backend/train_model.py

The trained model is saved as:

model/sign_language_model.h5

How It Works
The webcam captures live video.
MediaPipe detects the user's hand.
The 21 hand landmarks are extracted.
X, Y, and Z coordinates are converted into 63 numerical features.
Features from 30 consecutive frames form one sequence.
The LSTM model processes the sequence.
The model predicts one of the four supported gestures.
The prediction and confidence are displayed in the Flask web interface.
Future Improvements
Add more sign language gestures
Increase the size and diversity of the dataset
Improve robustness under different lighting conditions
Support multiple hands
Add continuous sentence formation
Add text-to-speech output
Improve prediction smoothing
Deploy the application as a web service
Project Contribution

This project has been developed and customized as a hands-on implementation of real-time sign language recognition.

The current implementation includes a customized 4-class dataset, model training pipeline, trained LSTM model, webcam-based prediction system, and Flask web interface.

License / Attribution

License

This project is intended for educational and portfolio purposes.

Please make sure the license of any third-party code, libraries, or resources used in the project is respected.