from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import initializers
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from sklearn.preprocessing import LabelEncoder
import numpy as np
from PIL import Image
import os
import pandas as pd
import joblib
import pickle

#from IPython.display import display
#import cv2

directory='Datasets/BIRDS450/images'

x_train = []
x_test = []
y_train =[]
y_test = []

new_path = os.path.join(directory,'train')
for img_name in os.listdir(new_path):
    for img_no in os.listdir(os.path.join(new_path,img_name)):
        temp = Image.open((os.path.join(new_path, img_name,img_no)))
        img=temp.copy()
        x_train.append(np.asarray(img))
        y_train.append(img_name)
        temp.close()

new_path = os.path.join(directory,'test')
for img_name in os.listdir(new_path):
    for img_no in os.listdir(os.path.join(new_path,img_name)):
        temp = Image.open((os.path.join(new_path, img_name,img_no)))
        img=temp.copy()
        x_test.append(np.asarray(img))
        y_test.append(img_name)
        temp.close()

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)
y_test = y_test.reshape(-1,1)
y_train = y_train.reshape(-1,1)

y_train = keras.utils.to_categorical(y_train, 400)
y_test = keras.utils.to_categorical(y_test, 400)

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

# Compute end step to finish pruning after 10 epochs.
batch_size = 100
epochs = 10
validation_split = 0.1 # 10% of training set will be used for validation set. 

num_images = y_train.shape[0] * (1 - validation_split)
end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

# Define model for pruning.
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                               final_sparsity=0.80,
                                                               begin_step=0,
                                                               end_step=end_step)
}

def define_Model():

    initializer = keras.initializers.GlorotNormal()

    model = Sequential()

    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(224, 224, 3), padding='same', kernel_initializer=initializer))
    # model.add(Dropout(0.3))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same', kernel_initializer=initializer))
    # model.add(Dropout(0.3))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, kernel_size=3, activation='relu', padding='same', kernel_initializer=initializer))
    # model.add(Dropout(0.3))
    # model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(100, activation='relu', kernel_initializer=initializer))
    model.add(Dense(1, activation='softmax', kernel_initializer=initializer))

    model_for_pruning = prune_low_magnitude(model, **pruning_params)

    model_for_pruning.compile(loss=keras.losses.CategoricalCrossentropy(), optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
    return model_for_pruning

model = define_Model()

epochs = 10     # Number of epochs hyperperameter (how many times the whole dataset will be seen by the network). This parameter can be tweaked.
batch_size = 100 #Number of samples in each batch hyperperameter. This parameter can be tweaked.

# x_train, y_train = preprocess(x_train, y_train) # This line can be uncommented in order to apply some normilisation to the training data.
# x_test, y_test = preprocess(x_test, y_test)     # This line can be uncommented in order to apply some normilisation to the test data.

# x_train, y_train = augmentation(x_train, y_train)

callbacks = [
  tfmot.sparsity.keras.UpdatePruningStep()]

model.fit(x_train, y_train, batch_size=batch_size,callbacks=callbacks,validation_split=validation_split, epochs=epochs, verbose=1, validation_data=(x_test, y_test)) # fit() starts the backpropagation algorithm.

model_for_export = tfmot.sparsity.keras.strip_pruning(model)

converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_and_pruned_tflite_model = converter.convert()

with open("models/model4.tflite", 'wb') as f:
  f.write(quantized_and_pruned_tflite_model)


