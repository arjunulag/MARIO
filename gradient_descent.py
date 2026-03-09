import numpy as np
from gradient import compute_gradients


def gradient_descent(model, x, y, lr=0.01, epochs=1000):
    """
    Train *model* on (x, y) using vanilla gradient descent.

    Parameters
    ----------
    model  : Parameter_init instance
    x      : np.ndarray, input data  — (N, in) or (in,)
    y      : np.ndarray, targets     — (N, out) or (out,)
    lr     : float, learning rate
    epochs : int, number of full passes

    Returns
    -------
    losses : list[float], loss at each epoch
    """
    losses = []

    for epoch in range(epochs):
        grads, loss = compute_gradients(model, x, y)
        losses.append(loss)

        for layer, g in zip(model.layers, grads):
            if layer["type"] == "linear":
                layer["W"] -= lr * g["dW"]
                layer["b"] -= lr * g["db"]

    return losses
