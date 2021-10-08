
from src.convnet.nodes.prep import normalize, reshape

def run(X_train, X_test):

    # normalize data
    X_train, X_test = normalize(X_train, X_test)
    print(X_train.shape)

    # reshape train dataset to (60000, 28,28,1)
    X_train, X_test = reshape(X_train, X_test, height=28, width=28, channel=1)
    return (X_train, X_test)
