
import numpy as np
from src.convnet.nodes.prep import normalize, reshape
from min_tfs_client.requests import TensorServingClient
from min_tfs_client.tensors import tensor_proto_to_ndarray
import os 


def run(img):

    # known labels
    CLASSES = ["T-shirt/top", "trouser", "pullover", "Dress", "coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    # preprocess image
    img = img.astype(np.float32)
    _, img = normalize(img, img)
    _, img = reshape(
        img, img, height=28, width=28, channel=1
    )

    # get prediction
    client = TensorServingClient(host=os.getenv("SERVING_SERVICE", "127.0.0.1"), port=8500, credentials=None)    
    response = client.predict_request(model_name="model", model_version=1, input_dict={"conv2d_input": img})
    class_proba = tensor_proto_to_ndarray(response.outputs["dense_1"])
    loc_most_likely = np.argmax(class_proba)
    predicted = CLASSES[loc_most_likely]
    return predicted
