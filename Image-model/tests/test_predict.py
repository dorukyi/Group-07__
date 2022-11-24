import json
from PIL import Image
import joblib
import os
import sys

sys.path.insert(0, 'C:/Users/sdoga/Desktop/Courses/Software-Engineering/Image_model2/src')
from Project import predict_proba

def test_predict():
    model=joblib.load("models/trained_model.joblib")
    with open('src/index_to_class_label.json', 'rb') as f:
        index_to_class_labels = json.load(f)
    index_to_class_labels = {
        int(k): v for k, v in index_to_class_labels.items()}
    img = Image.open("Datasets/BIRDS450/images/train/AFRICAN CROWNED CRANE/001.jpg")
    formatted_prediction = predict_proba(
        img,
        3,
        index_to_class_labels,
        model
        )
    assert formatted_prediction[0][0] == "African Crowned Crane"

test_predict()

