
from src.convnet.nodes.etl import load_dataset
from src.convnet.pipes import prep
from src.convnet.nodes.train import build_model, save_model
from typing import Dict, Any
import tensorflow as tf
import os 

def run(params:Dict[str, Any]):
    
    # define set of image classes
    LABELS = params["LABELS"]
    EPOCHS = params["EPOCHS"]
    MODEL_PATH = params["MODEL_PATH"]
    MODEL_VERS = params["VERSION"]
    MODEL_LOSS = params["MODEL_LOSS"]
    METRICS = params["METRICS"] 

    # etl
    (X_train, Y_train), (X_test, Y_test) = load_dataset(name="fashion_mnist")

    # preprocessing
    (X_train, X_test) = prep.run(X_train, X_test)

    # training
    model = build_model()
    model.compile(
        optimizer=tf.compat.v1.train.AdamOptimizer(),
        loss=MODEL_LOSS,
        metrics=METRICS
        )
    model.fit(X_train, Y_train, epochs=EPOCHS)
    
    # inspect saved model
    os.system("saved_model_cli show --dir {save_path} --all")

    # evaluate
    test_loss, test_acc = model.evaluate(X_test, Y_test)
    print(f"\nTest accuracy: {test_acc}")

    # save model
    save_path = os.path.join(MODEL_PATH, str(MODEL_VERS))
    save_model(model, save_path)