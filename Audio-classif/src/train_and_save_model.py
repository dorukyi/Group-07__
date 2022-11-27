# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import IPython.display as ipd
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import warnings
warnings.filterwarnings('ignore')
# %matplotlib inline
# %load_ext tensorboard
import os
import csv 
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import svm
import joblib
import pickle

def preprocessing_data(path_train, path_test, path_validate):
  """
  Rewriting the directories names and file names, so they're conneceted to train, validate and test.
  """
  path_train = path_train
  path_test = path_test
  path_validate = path_validate

  for file in os.listdir(path_train):
    source = path_train+file
    new_file = file[0:file.index("_")] + ".wav"
    new_source = path_train+new_file
    os.rename(source, new_source)

  for file in os.listdir(path_test):
    source = path_test+file
    new_file = file[0:file.index("_")] + ".wav"
    new_source = path_test+new_file
    os.rename(source, new_source)
    
  for file in os.listdir(path_validate):
    source = path_validate+file
    new_file = file[0:file.index("_")] + ".wav"
    new_source = path_validate+new_file
    os.rename(source, new_source)
  
  arg_train = os.listdir(path_train)
  arg_test = os.listdir(path_test)
  arg_validate = os.listdir(path_validate)

  return arg_train, arg_test, arg_validate

def featurextraction(arg_train, arg_test, arg_validate):
  """
  creates important features of aall the files, and puts them in a .csv
  """
  path_train = 'ManualTrain/'
  path_test = 'ManualTest/'
  path_validate = 'ManualValidation/'

  header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
  for i in range(1, 21):
      header += f' mfcc{i}'
  header += ' label'
  header = header.split()

  file = open('dataset.csv', 'w', newline='')
  with file:
      writer = csv.writer(file)
      writer.writerow(header)
  insects = 'ManualTrain ManualValidation ManualTest'.split()
  for g in insects:
      for filename in os.listdir(os.path.join("Dataset/Audio",g)):
          songname = os.path.join("Dataset/Audio",g,filename)
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
          temp = ""
          for l in filename:
            if not l.isdigit():
              temp += l
          to_append += f' {temp[:-4]}'
          file = open('dataset.csv', 'a', newline='')
          with file:
              writer = csv.writer(file)
              writer.writerow(to_append.split())

def loading_dataframe(dataframe_csv):
  df = pd.read_csv(dataframe_csv)
  return df

# def set_label(row):
#   """
#   Setting labels for species. To binarize them later.
#   """
#   if (row["label"] == "Chorthippusbig") or (row["label"] == "Chorthippusbru") or (row["label"] == "Grylluscampest") or (row["label"] == "Nemobiussylves") or (row["label"] == "Oecanthuspellu") or (row["label"] == "Pholidopteragr") or  (row["label"] == "Pseudochorthip") or (row["label"] == "Roeselianaroes") or (row["label"] == "Tettigoniaviri"):
#       return "Orthoptera"
#   else:
#       return "Cicadidae"
  
#   return df

def preprocessing_data2(df):
  """
  Binarization labels. By dropping unneccesary columns and splitting the data in train and test sets.v
  """
  #df = df.assign(order=df.apply(set_label, axis=1)) 
  #df.drop(columns=['label'], axis = 1, inplace = True)
  df = df.drop(['filename'],axis=1)
  genre_list = df.iloc[:, -1]
  encoder = LabelEncoder()
  y = encoder.fit_transform(genre_list)
  scaler = StandardScaler()
  X = scaler.fit_transform(np.array(df.iloc[:, :-1]))
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  return X_train, X_test, y_train, y_test

def svmSVC(X_train, X_test, y_train, y_test):
  """
  Training and testing SVM. Outputting accuracy.
  """
  clf = svm.SVC()
  clf.fit(X_train, y_train)
  print(clf.score(X_test, y_test))
  joblib.dump(clf, "models/trained_model")

arg_train = 'Dataset/Audio/ManualTrain'
arg_test = 'Dataset/Audio/ManualTest'
arg_validate = 'Dataset/Audio/ManualValidation'
#arg_train, arg_test, arg_validate = preprocessing_data(path_train, path_test, path_validate)
featurextraction(arg_train, arg_test, arg_validate)
df = loading_dataframe("dataset.csv")

X_train, X_test, y_train, y_test = preprocessing_data2(df)
svmSVC(X_train, X_test, y_train, y_test)
