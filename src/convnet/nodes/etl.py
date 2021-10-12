import tensorflow as tf


def load_dataset(name="fashion_mnist"):
    """Import train and test dataset

    Args:
        name (str): [description]
    """

    (X_train, Y_train), (X_test, Y_test) = eval(
        f"tf.keras.datasets.{name}.load_data()"
    )
    return (X_train, Y_train), (X_test, Y_test)
