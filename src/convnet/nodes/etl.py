from io import BytesIO

import numpy as np
from PIL import Image


def load_dataset(name="fashion_mnist"):
    """Import train and test dataset

    Args:
        name (str): [description]
    """

    (X_train, Y_train), (X_test, Y_test) = eval(
        f"tf.keras.datasets.{name}.load_data()"
    )
    return (X_train, Y_train), (X_test, Y_test)


def read_convert_image(file, height, width):
    loaded_image = Image.open(BytesIO(file))
    image_to_convert = np.asarray(
        loaded_image.resize((height, width))
    )[..., :3]
    image_to_convert = np.expand_dims(image_to_convert, 0)
    return np.float32(image_to_convert)
