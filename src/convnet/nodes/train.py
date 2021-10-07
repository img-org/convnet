
import tensorflow as tf
import os 
import shutil

# define function nodes
def build_model():
    """Build the convnet model

    Returns:
        [type]: [description]
    """
    cnn = tf.keras.models.Sequential()
    cnn.add(tf.keras.layers.Conv2D(32, (3,3), 
                                    activation = "relu",
                                    input_shape = (28,28,1)))
    cnn.add(tf.keras.layers.MaxPooling2D(2,2))
    cnn.add(tf.keras.layers.Conv2D(64, (3,3), activation = "relu"))
    cnn.add(tf.keras.layers.Flatten())
    cnn.add(tf.keras.layers.Dense(64, activation = "relu"))
    cnn.add(tf.keras.layers.Dense(10, activation = "softmax"))
    cnn.summary()
    return cnn

def save_model(cnn, save_path:str):
    """Save model

    Args:
        cnn ([type]): [description]
        save_path ([type]): [description]
    """
    # clean up path
    if os.path.exists(save_path):
        print("\nalready saved a model, cleaning up\n")
        shutil.rmtree(save_path)

    # disable eager execution to save model
    if tf.executing_eagerly():
        tf.compat.v1.disable_eager_execution()
        
    tf.compat.v1.saved_model.simple_save(
        tf.compat.v1.keras.backend.get_session(),
        save_path,
        inputs = {"input_image": cnn.input},
        outputs = {i.name:i for i in cnn.outputs}
    )