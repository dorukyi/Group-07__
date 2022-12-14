# -*- coding: utf-8 -*-

from birds_image_classification_v2 import baseline_cnn
import trained_classification_model_birdimages.h5

def predict_birdImage(img, img_file):
  "This function predicts for one image of a bird. It calls the trained model and set it weights in the cnn. It grabs the reshaped image and the image. The function returns the image and the label."
  model_base = baseline_cnn()
  model_base.load_weights("trained_classification_model_birdimages.h5")
  
  y_pred_test = model_base.predict(img)
  y_pred = np.argmax(y_pred_test, axis=1)

  return img_file, y_pred

import numpy as np
from PIL import Image
from numpy import asarray

def image_reshaping(path):
  "It reshapes the image so it suits the input size of the model."
  img_file = Image.open(path)
  img = img_file.resize((224,224))
  img = asarray(img)
  img = img*1./255
  img = np.expand_dims(img, axis = 0)
  return img, img_file

