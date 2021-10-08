
from src.convnet.nodes.etl import load_dataset
from src.convnet.pipes import prep
from src.convnet.nodes.train import build_model
from typing import Dict, Any
import tensorflow as tf

def run(params:Dict[str, Any]):
    
    # define set of image classes
    LABELS = params["LABELS"]
    EPOCHS = params["EPOCHS"]

    # etl
    (X_train, Y_train), (X_test, Y_test) = load_dataset(name="fashion_mnist")

    # preprocessing
    (X_train, X_test) = prep.run(X_train, X_test)

    # training
    model = build_model()
    model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])
    model.fit(X_train, Y_train, epochs=EPOCHS)
    test_loss, test_acc = model.evaluate(X_test, Y_test)
    print(f"\nTest accuracy: {test_acc}")
