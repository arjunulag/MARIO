import numpy as np
from gradient import compute_gradients


def momentum(model, x, y, lr=0.01, beta=0.9, epochs=1000):
    """
    Train *model* on (x, y) using gradient descent with momentum.

    Maintains an exponential moving average of gradients (first moment)
    so the optimizer builds up velocity in consistent directions and
    dampens oscillation.

    Parameters
    ----------
    model  : Parameter_init instance
    x      : np.ndarray, input data  — (N, in) or (in,)
    y      : np.ndarray, targets     — (N, out) or (out,)
    lr     : float, learning rate
    beta   : float, momentum decay rate
    epochs : int, number of full passes

    Returns
    -------
    losses : list[float], loss at each epoch
    """
    cache = []
    for layer in model.layers:
        if layer["type"] == "linear":
            cache.append({
                "m_W": np.zeros_like(layer["W"]),
                "m_b": np.zeros_like(layer["b"]),
            })
        else:
            cache.append({})

    losses = []

    for epoch in range(epochs):
        grads, loss = compute_gradients(model, x, y)
        losses.append(loss)

        for layer, g, c in zip(model.layers, grads, cache):
            if layer["type"] == "linear":
                c["m_W"] = beta * c["m_W"] + (1 - beta) * g["dW"]
                c["m_b"] = beta * c["m_b"] + (1 - beta) * g["db"]

                layer["W"] -= lr * c["m_W"]
                layer["b"] -= lr * c["m_b"]

    return losses
