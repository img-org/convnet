# Tests simplest request
#
# curl http://2c36-86-245-206-22.ngrok.io/items/1
#
# or in python:
#
# response = requests.get(
#     "http://2c36-86-245-206-22.ngrok.io/items/1"
#     )
# response.text

import json

import numpy as np
import requests
import tensorflow as tf
from src.convnet.nodes.prep import normalize, reshape


def run(img):

    # PARAMS
    # ----
    # Server HOST
    HOST = "http://localhost"

    # Server PORT
    PORT = 8501

    # model and its path
    MODEL_NAME = "model"

    # model version
    VERSION = 1

    # labels
    CLASSES = [
        "T-shirt/top",
        "trouser",
        "pullover",
        "Dress",
        "coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    # Data
    # ----
    # simulate data to predict
    # (X_train, _), (
    #     X_test,
    #     _,
    # ) = tf.keras.datasets.fashion_mnist.load_data()

    # preprocess
    _, X_test = normalize(img, img)
    _, X_test = reshape(
        X_test, X_test, height=28, width=28, channel=1
    )
    data = X_test[0:3].tolist()

    # Build and submit predict request to model service
    # -------------------------------------------------
    # url,
    url = f"{HOST}:{PORT}/v{VERSION}/models/{MODEL_NAME}:predict"

    # payload (data)
    payload = json.dumps(
        {
            "signature_name": "serving_default",
            "instances": data,
        }
    )
    # header
    headers = {"content-type": "applicaton/json"}

    # post request and get response
    response = requests.post(
        url, data=payload, headers=headers
    )
    response_txt = json.loads(response.text)
    predictions = response_txt["predictions"]
    loc_most_likely = np.argmax(predictions[0])
    predicted = CLASSES[loc_most_likely]
    return predicted
