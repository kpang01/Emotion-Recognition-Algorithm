import os
import tempfile
import urllib.request
from collections import Counter

import librosa
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from keras.models import load_model
from pydub import AudioSegment
from pydub.silence import split_on_silence
from pydub.utils import make_chunks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

app = Flask(__name__)

# Load the trained model for Voice Emotion Detection
model_path = 'F:/Python Code/Advance/SER/model.h5'
voice_emotion_model = load_model(model_path)
sample_rate = 22050

# Load the scaler and encoder used during training
scaler = StandardScaler()
encoder = OneHotEncoder()

# Load the scaler and encoder parameters from the training phase
features = pd.read_csv('F:/Python Code/Advance/SER/features.csv')
X = features.iloc[:, :-1].values
Y = features['labels'].values

Y_encoded = encoder.fit_transform(np.array(Y).reshape(-1, 1)).toarray()
x_train, x_test, y_train, y_test = train_test_split(X, Y_encoded, random_state=0, shuffle=True)
scaler.fit(x_train)

# Define utility functions
def extract_features(data):
    # ZCR, Chroma_stft, MFCC, RMS, MelSpectrogram
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))

    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))

    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))

    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))

    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))

    return result

def prediction(sample):
    try:
        # Preprocess the sample
        sample = scaler.transform(sample.reshape(1, -1))
        sample = np.expand_dims(sample, axis=2)

        # Make prediction
        pred = voice_emotion_model.predict(sample)
        predicted_label = np.argmax(pred)
        emotion_mapping = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'}
        predicted_emotion = emotion_mapping[predicted_label]

        return predicted_emotion
    except Exception as e:
        return f"Error predicting emotion: {str(e)}"

def voice_length(file_path):
    try:
        return librosa.get_duration(filename=file_path)
    except Exception as e:
        return f"Error getting voice length: {str(e)}"

def delete_previous_files(folder_path):
    try:
        for f in os.listdir(folder_path):
            os.remove(os.path.join(folder_path, f))
        print('Previous files cleaned.')
    except Exception as e:
        print(f"Error cleaning files: {str(e)}")

def splitting_audio(file_path):
    try:
        audio = AudioSegment.from_file(file_path, "wav")
        chunk_length_ms = 7000  # Split audio into 3-second chunks
        chunks = make_chunks(audio, chunk_length_ms)

        # Create 'Testing' directory if it doesn't exist
        if not os.path.exists('Testing'):
            os.makedirs('Testing')

        # Delete previous files inside the testing folder
        delete_previous_files('Testing')

        for i, chunk in enumerate(chunks):
            chunk_name = f"chunk_{i}.wav"
            chunk.export('Testing/' + chunk_name, format="wav")
    except Exception as e:
        print(f"Error splitting audio: {str(e)}")


# Define the utility function to calculate the final emotion and percentage
def calculate_final_emotion(prediction_list):
    # Count the occurrences of each emotion in the prediction list
    emotion_counts = Counter(prediction_list)
    
    # Calculate the total number of predictions
    total_predictions = len(prediction_list)
    
    # Calculate the percentage of each emotion
    emotion_percentages = {emotion: (count / total_predictions) * 100 for emotion, count in emotion_counts.items()}
    
    # Find the emotion with the highest percentage
    final_emotion = max(emotion_percentages, key=emotion_percentages.get)
    
    return final_emotion, emotion_percentages[final_emotion]

@app.route('/predict', methods=['POST'])
def predict_emotion():

    if 'firebase_url' not in request.form:
        return jsonify({'error': 'No Firebase URL provided'})

    firebase_url = request.form['firebase_url']

    try:
        # Download audio from Firebase URL
        with urllib.request.urlopen(firebase_url) as response:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_audio_file:
                tmp_audio_file.write(response.read())
                tmp_audio_file_path = tmp_audio_file.name

        # Split the audio file into chunks and predict emotion
        splitting_audio(tmp_audio_file_path)
        prediction_list = []

        # Iterate through each chunk and predict emotion
        fileList = os.listdir('Testing')
        for file in fileList:
            test_file_path = 'Testing/' + file

            if voice_length(test_file_path) < 1:
                continue

            # Extract features from the chunk
            data, sample_rate = librosa.load(test_file_path, duration=2.5, offset=0.6)
            features = extract_features(data)
            prediction_list.append(prediction(features))
            
        final_emotion, percentage = calculate_final_emotion(prediction_list)
        delete_previous_files('Testing')
        # Delete the temporary audio file
        os.remove(tmp_audio_file_path)

        # Return predicted emotions
        return jsonify({
            'final_emotion': final_emotion,
            'percentage': percentage,
            'predicted_emotions': prediction_list
        })

    except Exception as e:
        # Handle any errors and return error message
        return jsonify({'error': str(e)})

    
if __name__ == '__main__':
    app.run(debug=True)