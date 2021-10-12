import matplotlib.pyplot as plt
import numpy as np


# define function nodes
def normalize(X_train, X_test):
    """Ceil image intensity to 1.0
    Args:
    Return:
    """
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    return X_train, X_test


def reshape(
    X_train, X_test, height: int, width: int, channel: int
):
    """Reshape image dataset
    Args:
    Return:
    """
    X_train = X_train.reshape(
        X_train.shape[0], height, width, channel
    )
    X_test = X_test.reshape(
        X_test.shape[0], height, width, channel
    )
    return X_train, X_test


def to_gray(img: np.ndarray):
    """Convert to grayscale that matches human perception

    Args:
        img (np.ndarray): (height,height,n_channels) array

    Returns:
        [np.ndarray]: (height,height,1) grayscale array
    """
    return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])


def plot_img_sample(X_train, Y_train):
    """Plot a sample of images
    Args:
    Return:
    """
    W_grid = 4
    L_grid = 4
    _, axes = plt.subplots(L_grid, W_grid, figsize=(15, 15))
    axes = axes.ravel()
    n_training = len(X_train)
    for i in np.arange(0, L_grid * W_grid):
        index = np.random.randint(0, n_training)
        axes[i].imshow(X_train[index].reshape(28, 28))
        axes[i].set_title(Y_train[index])
        axes[i].axis("off")
    plt.subplots_adjust(hspace=0.4)
