# -*- coding: utf-8 -*-
"""predict_insectAudio.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ezQpcvx1IEvtiMQHN_7nD8cjYNDKDTNZ
"""

def create_dataset_features(insects= 'ManualTrain ManualValidation ManualTest'):
    """
    Preprocessing and feature extraction for new predictions.
    """
    header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label'
    header = header.split()

    file = open('dataset2.csv', 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
    insects = insects.split()
    for g in insects:
        for filename in os.listdir(f'D:/DataScience/AUDIO/{g}'):
            songname = f'D:/DataScience/AUDIO/{g}/{filename}'
            y, sr = librosa.load(songname, mono=True, duration=30)
            rmse = librosa.feature.rms(y=y)[0]
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
            for e in mfcc:
                to_append += f' {np.mean(e)}'
            to_append += f' {filename[0:14]}'
            file = open('dataset2.csv', 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())

def predict(df_test: pd.DataFrame, model = svm, scaler = scaler):
    """
    Predictions.
    """
    data = df_test.copy()
    data.head()# Dropping unneccesary columns
    data = data.drop(columns=['filename'],axis=1)
    X = scaler.transform(np.array(data.iloc[:, :-1], dtype = float))
    
    return model.predict(X)

ManualTrain = '/ManualTrain'
ManualValidation = '/ManualValidation'
ManualTest = '/ManualTest'
create_dataset_features(insects= 'ManualTrain ManualValidation ManualTest')

predict(df_test, svm, scaler)