import json
import PIL
from PIL import Image
import os
import tensorflow as tf
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import librosa
import csv
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import io
from pydub import AudioSegment

@st.cache()
def load_index_to_label_dict(
        path: str = "src/index_to_class_label.json"
        ) -> dict:
    """Retrieves and formats the
    index to class label
    lookup dictionary needed to
    make sense of the predictions.
    When loaded in, the keys are strings, this also
    processes those keys to integers."""
    with open(path, 'r') as f:
        index_to_class_label_dict = json.load(f)
    index_to_class_label_dict = {
        int(k): v for k, v in index_to_class_label_dict.items()}
    return index_to_class_label_dict

@st.cache()
def preprocess(path):
    y, sr = librosa.load(path, mono=True, duration=30)
    rmse = librosa.feature.rms(y=y)[0]
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    to_append = f'{np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
    for e in mfcc:
        to_append += f' {np.mean(e)}'
        
    file = open('dataset333.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(to_append.split())
    df = pd.read_csv('dataset333.csv', header=None)
    X = np.array(df).reshape(1,-1)
    return X

@st.cache()
def predict(
        audio_file,
        index_to_label_dict,
        model,
        ) -> list:
    return index_to_label_dict.get(model.predict(audio_file)[0])


if __name__ == '__main__':
    model = joblib.load("models/trained_model.joblib")
    index_to_class_label_dict = load_index_to_label_dict()

    st.title('Welcome To Software Engineering Project')
    instructions = """
        Upload your audio.
        The audio you upload will be fed
        through the Deep Neural Network in real-time
        and the output will be displayed to the screen.
        """
    st.write(instructions)

    uploaded_file = st.file_uploader('Upload An Audio File', type='wav')

    if not uploaded_file:  # if user uploaded file
        path = "Chorthippusbiguttulus46.wav"
        audio_file = preprocess(path)
        st.title("Here is the audio you've uploaded")
        st.audio(path)
        
    else:
        audio_bytes = uploaded_file.read()
        st.title("Here is the audio you've uploaded")
        st.audio(audio_bytes)
        file_var = AudioSegment.from_wav(uploaded_file) 
        file_var.export("temp.wav", format='wav')
        audio_file = preprocess("temp.wav")
        os.remove("temp.wav")


    prediction = predict(audio_file, index_to_class_label_dict,model)

    st.title("Here is the most likely insect species")

    st.write(prediction)
