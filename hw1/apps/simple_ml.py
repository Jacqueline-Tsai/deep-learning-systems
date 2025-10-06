"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filesname, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    with gzip.open(image_filesname, "rb") as f_img:
        magic, num_images, rows, cols = struct.unpack(">IIII", f_img.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid image magic number: {magic}")
        img_bytes = f_img.read(num_images * rows * cols)
        X = np.frombuffer(img_bytes, dtype=np.uint8).reshape(num_images, rows * cols)
        X = (X.astype(np.float32) / 255.0)

    with gzip.open(label_filename, "rb") as f_lbl:
        magic, num_labels = struct.unpack(">II", f_lbl.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid label magic number: {magic}")
        y_bytes = f_lbl.read(num_labels)
        y = np.frombuffer(y_bytes, dtype=np.uint8)

    if num_images != num_labels:
        raise ValueError(f"Image/label count mismatch: {num_images} vs {num_labels}")

    return X, y


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    batch_size = Z.shape[0]
    logsumexp = ndl.log(ndl.exp(Z).sum(axes=(1,)))
    correct_logits = (y_one_hot * Z).sum()
    return (logsumexp.sum() - correct_logits) / batch_size

def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    num_iter = (y.size + batch - 1) // batch

    for i in range(num_iter):
        start = i * batch
        end   = (i + 1) * batch

        # Batch tensors
        x_b = ndl.Tensor(X[start:end, :])
        logits = ndl.relu(x_b.matmul(W1)).matmul(W2)

        # One-hot labels (same logic, fixed size per loop)
        y_b = y[start:end]
        y_one_hot = np.zeros((batch, y.max() + 1))
        y_one_hot[np.arange(batch), y_b] = 1
        y_one_hot = ndl.Tensor(y_one_hot)

        # Loss + backward
        loss = softmax_loss(logits, y_one_hot)
        loss.backward()

        # SGD update (recreate tensors to detach from graph)
        W1 = ndl.Tensor(W1.realize_cached_data() - lr * W1.grad.realize_cached_data())
        W2 = ndl.Tensor(W2.realize_cached_data() - lr * W2.grad.realize_cached_data())

    return W1, W2


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
