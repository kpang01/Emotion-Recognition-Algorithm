Speech Emotion Recognition (SER) using Deep Learning
This project implements a Speech Emotion Recognition (SER) system using deep learning techniques. It processes and classifies audio speech data from multiple datasets including RAVDESS, TESS, and EmoDB, and aims to predict emotions from spoken audio.

📁 Datasets Used
RAVDESS: Ryerson Audio-Visual Database of Emotional Speech and Song

TESS: Toronto emotional speech set

EmoDB: Berlin Database of Emotional Speech

Make sure to download and place the datasets in their corresponding directories as referenced in the code.

🧠 Model Summary
The model is a Convolutional Neural Network (CNN) trained on audio features extracted from the datasets. It uses:

Feature extraction via librosa

Audio playback support for inspection

Training with Keras Sequential API

Layers: Conv1D, MaxPooling1D, Dense, Dropout, BatchNormalization

🎙️ Speech Emotion Recognition (SER) System
This project is a Speech Emotion Recognition (SER) system built using deep learning and signal processing. It consists of two main components:

A Jupyter Notebook (Model.ipynb) that trains a Convolutional Neural Network (CNN) to classify emotions from speech.

A Flask-based API (real.py) that performs real-time emotion recognition on audio files (e.g., from Firebase).

📁 Project Structure
graphql
Copy
Edit
├── Model.ipynb                 # Jupyter notebook for training the CNN model
├── real.py                     # Flask API for real-time prediction
├── model.h5                    # Trained CNN model
├── features.csv                # Training features for scaler and encoder
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── Testing/                    # Temporary folder for audio chunks (runtime only)
🧠 Algorithm Overview
Model Training (Model.ipynb)
Datasets:

RAVDESS, TESS, EmoDB (emotion-labeled speech datasets)

Feature Extraction:

MFCCs (Mel-Frequency Cepstral Coefficients)

Chroma STFT, ZCR, RMS, Mel Spectrogram

CNN Architecture:

Conv1D, MaxPooling1D, BatchNormalization, Dense, Dropout

Trained with Keras Sequential API

Multi-class classification using Softmax

Evaluation:

Accuracy, confusion matrix, and classification report

Real-Time Prediction (real.py)
Input:

Accepts a POST request to /predict with a Firebase URL containing .wav audio.

Audio Processing:

Downloads audio from Firebase

Splits it into 7-second chunks using pydub

Prediction:

For each chunk:

Extracts features (same as during training)

Scales them using a saved StandardScaler

Predicts emotion using the CNN (model.h5)

Aggregates results to find the most likely emotion and its percentage

Output:

JSON response with:

final_emotion

percentage

predicted_emotions (for each chunk)


