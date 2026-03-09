import numpy as np
from gradient import compute_gradients


def adam(model, x, y, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, epochs=1000):
    """
    Train *model* on (x, y) using Adam (Adaptive Moment Estimation).

    Combines momentum (first moment m) with RMSProp (second moment v),
    plus bias correction to account for zero-initialization of both.

    Parameters
    ----------
    model   : Parameter_init instance
    x       : np.ndarray, input data  — (N, in) or (in,)
    y       : np.ndarray, targets     — (N, out) or (out,)
    lr      : float, learning rate
    beta1   : float, decay rate for first moment  (momentum)
    beta2   : float, decay rate for second moment (RMSProp)
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
                "m_W": np.zeros_like(layer["W"]),
                "m_b": np.zeros_like(layer["b"]),
                "v_W": np.zeros_like(layer["W"]),
                "v_b": np.zeros_like(layer["b"]),
            })
        else:
            cache.append({})

    losses = []

    for epoch in range(epochs):
        grads, loss = compute_gradients(model, x, y)
        losses.append(loss)
        t = epoch + 1

        for layer, g, c in zip(model.layers, grads, cache):
            if layer["type"] == "linear":
                # momentum (first moment)
                c["m_W"] = beta1 * c["m_W"] + (1 - beta1) * g["dW"]
                c["m_b"] = beta1 * c["m_b"] + (1 - beta1) * g["db"]

                # RMSProp (second moment)
                c["v_W"] = beta2 * c["v_W"] + (1 - beta2) * g["dW"] ** 2
                c["v_b"] = beta2 * c["v_b"] + (1 - beta2) * g["db"] ** 2

                # bias correction
                m_W_hat = c["m_W"] / (1 - beta1 ** t)
                m_b_hat = c["m_b"] / (1 - beta1 ** t)
                v_W_hat = c["v_W"] / (1 - beta2 ** t)
                v_b_hat = c["v_b"] / (1 - beta2 ** t)

                layer["W"] -= lr * m_W_hat / (np.sqrt(v_W_hat) + epsilon)
                layer["b"] -= lr * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

    return losses
