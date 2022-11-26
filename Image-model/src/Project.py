import json
import PIL
from PIL import Image
import os
import tensorflow as tf
import streamlit as st
import pandas as pd
import numpy as np

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


def load_files(keys: list):
    directory = "Datasets/BIRDS450/images/"
    files = []
    for key in keys:
        whole_path = os.path.join(directory,key)
        file_image = Image.open(whole_path)
        files.append(file_image)
    return files

@st.cache()
def load_file_structure(path: str = 'src/all_image_files.json') -> dict:
    """Retrieves JSON document outlining the file structure"""
    with open(path, 'r') as f:
        return json.load(f)


@st.cache()
def load_list_of_images_available(
        all_image_files: dict,
        image_files_dtype: str,
        bird_species: str
        ) -> list:
    """Retrieves list of available images given the current selections"""
    species_dict = all_image_files.get(image_files_dtype)
    list_of_files = species_dict.get(bird_species)
    return list_of_files

@st.cache()
def predict_proba(
            img: PIL.Image,
            k: int,
            index_to_class_labels: dict,
            path="models/model4.tflite"
            ):
        """
        Feeds single image through network and returns
        top k predicted labels and probabilities

        params
        ---------------
        img - PIL Image - Single image to feed through model
        k - int - Number of top predictions to return
        index_to_class_labels - dict - Dictionary
            to map indices to class labels
        show - bool - Whether or not to
            display the image before prediction - default False

        returns
        ---------------
        formatted_predictions - list - List of top k
            formatted predictions formatted to include a tuple of
            1. predicted label, 2. predicted probability as str
        """
        # Load TFLite model and allocate tensors.
        interpreter = tf.lite.Interpreter(model_path=path)
        interpreter.allocate_tensors()
        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Test model on random input data.
        input_shape = input_details[0]['shape']
        input_data = np.asarray(img).reshape(1, 224, 224, 3)
        input_data = input_data.astype("float32")
        interpreter.set_tensor(input_details[0]['index'], input_data)

        interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        probabilites = interpreter.get_tensor(output_details[0]['index'])

        #probabilites = model.predict(np.asarray(img).reshape(1, 224, 224, 3))
        indices = sorted(range(len(probabilites[0])), key=lambda i: probabilites[0][i],reverse=True)[:k]
        probabilites = sorted(probabilites[0],reverse=True)[:k]
        formatted_predictions = []

        for pred_prob, pred_idx in zip(probabilites, indices):
            predicted_label = index_to_class_labels[pred_idx].title()
            predicted_perc = pred_prob * 100
            formatted_predictions.append(
                (predicted_label, f"{predicted_perc:.3f}%"))

        return formatted_predictions

@st.cache()
def predict(
        img: Image.Image,
        index_to_label_dict: dict,
        k: int
        ) -> list:
    """Transforming input image according to ImageNet paper
    The Resnet was initially trained on ImageNet dataset
    and because of the use of transfer learning, I froze all
    weights and only learned weights on the final layer.
    The weights of the first layer are still what was
    used in the ImageNet paper and we need to process
    the new images just like they did.

    This function transforms the image accordingly,
    puts it to the necessary device (cpu by default here),
    feeds the image through the model getting the output tensor,
    converts that output tensor to probabilities using Softmax,
    and then extracts and formats the top k predictions."""
        
    formatted_predictions = predict_proba(img, k, index_to_label_dict)
    return formatted_predictions


if __name__ == '__main__':
    index_to_class_label_dict = load_index_to_label_dict()
    all_image_files = load_file_structure()

    st.title('Welcome To Software Engineering Project')
    instructions = """
        Upload your image.
        The image you upload will be fed
        through the Deep Neural Network in real-time
        and the output will be displayed to the screen.
        """
    st.write(instructions)

    file = st.file_uploader('Upload An Image')

    if file:  # if user uploaded file
        img = Image.open(file)
    else:
        img = Image.open("Datasets/BIRDS450/images/train/AFRICAN OYSTER CATCHER/003.jpg")
    prediction = predict(img, index_to_class_label_dict, k=5)
    top_prediction = prediction[0][0]
    available_images = all_image_files.get(
        'train').get(top_prediction.upper())
    examples_of_species = np.random.choice(available_images, size=3)
    files_to_get = []

    for im_name in examples_of_species:
        path = os.path.join("train",top_prediction.upper(),im_name)
        files_to_get.append(path)
    images = load_files(keys=files_to_get)

    st.title("Here is the image you've selected")
    resized_image = img.resize((400, 400))
    st.image(resized_image)
    st.title("Here are the five most likely bird species")
    df = pd.DataFrame(data=np.zeros((5, 2)),
                      columns=['Species', 'Confidence Level'],
                      index=np.linspace(1, 5, 5, dtype=int))

    for idx, p in enumerate(prediction):
        link = 'https://en.wikipedia.org/wiki/' + \
            p[0].lower().replace(' ', '_')
        df.iloc[idx,
                0] = f'<a href="{link}" target="_blank">{p[0].title()}</a>'
        df.iloc[idx, 1] = p[1]
    st.write(df.to_html(escape=False), unsafe_allow_html=True)
    st.title(f"Here are three other images of the {prediction[0][0]}")

    st.image(images)
