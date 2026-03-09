import numpy as np
from gradient import compute_gradients


def rmsprop(model, x, y, lr=0.001, beta=0.9, epsilon=1e-8, epochs=1000):
    """
    Train *model* on (x, y) using RMSProp.

    Maintains an exponential moving average of squared gradients per
    parameter, then scales the update by 1/sqrt(v + eps) so that
    parameters with large gradients get smaller steps and vice versa.

    Parameters
    ----------
    model   : Parameter_init instance
    x       : np.ndarray, input data  — (N, in) or (in,)
    y       : np.ndarray, targets     — (N, out) or (out,)
    lr      : float, learning rate
    beta    : float, decay rate for the moving average of squared gradients
    epsilon : float, small constant to avoid division by zero
    epochs  : int, number of full passes

    Returns
    -------
    losses : list[float], loss at each epoch
    """
    cache = []
    for layer in model.layers:
        if layer["type"] == "linear":
            cache.append({
                "v_W": np.zeros_like(layer["W"]),
                "v_b": np.zeros_like(layer["b"]),
            })
        else:
            cache.append({})

    losses = []

    for epoch in range(epochs):
        grads, loss = compute_gradients(model, x, y)
        losses.append(loss)

        for layer, g, c in zip(model.layers, grads, cache):
            if layer["type"] == "linear":
                c["v_W"] = beta * c["v_W"] + (1 - beta) * g["dW"] ** 2
                c["v_b"] = beta * c["v_b"] + (1 - beta) * g["db"] ** 2

                layer["W"] -= lr * g["dW"] / (np.sqrt(c["v_W"]) + epsilon)
                layer["b"] -= lr * g["db"] / (np.sqrt(c["v_b"]) + epsilon)

    return losses
