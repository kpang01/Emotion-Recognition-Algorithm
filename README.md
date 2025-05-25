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

🛠️ Setup & Installation
bash
Copy
Edit
# Clone the repository
git clone https://github.com/your-repo/speech-emotion-recognition.git
cd speech-emotion-recognition

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
Requirements
Python 3.7+

numpy, pandas, librosa, seaborn, matplotlib

scikit-learn

keras / tensorflow

🚀 Running the Notebook
Run the Model.ipynb notebook using Jupyter:

bash
Copy
Edit
jupyter notebook Model.ipynb
Ensure that the dataset paths (Ravdess, Tess, EmoDB) are correctly set before running.

📊 Outputs
Emotion classification results

Confusion matrix and evaluation metrics

Spectrogram visualizations of audio data

📁 Project Structure
mathematica
Copy
Edit
├── Model.ipynb
├── README.md
├── requirements.txt
├── [Audio Dataset Folders]
└── models/
📌 Notes
You may need to adjust the file paths for datasets depending on your local setup.

This project filters warnings and deprecated messages for a cleaner output.
