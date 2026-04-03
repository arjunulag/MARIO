import numpy as np
from mse_vector import mse_vector, mse_derivative


def compute_gradients(model, x, y):
    """
    Full backward pass: computes dL/dW and dL/db for every linear layer.

    Parameters
    ----------
    model : Parameter_init instance (from functions.py)
    x     : np.ndarray, input data — shape (N, in) or (in,)
    y     : np.ndarray, target — shape (N, out) or (out,)

    Returns
    -------
    grads : list[dict]
        Same length as model.layers.  For linear layers the dict has
        keys "dW" and "db"; for activation layers the dict is empty.
    loss  : float
        MSE loss value for this forward pass.
    """
    single = x.ndim == 1
    if single:
        x = x[np.newaxis, :]
        y = y[np.newaxis, :]

    activations = [x]
    pre_activations = []

    a = x
    for layer in model.layers:
        if layer["type"] == "linear":
            z = a @ layer["W"] + layer["b"]
            pre_activations.append(z)
            a = z
        elif layer["type"] == "relu":
            alpha = layer.get("alpha", 0.01)
            pre_activations.append(a)
            a = np.where(a > 0, a, alpha * a)
        elif layer["type"] == "sigmoid":
            pre_activations.append(a)
            a = 1.0 / (1.0 + np.exp(-a))
        activations.append(a)

    y_hat = a
    loss = mse_vector(y_hat, y)
    da = mse_derivative(y_hat, y)

    grads = [None] * len(model.layers)

    act_idx = len(activations) - 1
    pre_idx = len(pre_activations) - 1

    for i in reversed(range(len(model.layers))):
        layer = model.layers[i]

        if layer["type"] == "linear":
            a_prev = activations[act_idx - 1]
            dW = a_prev.T @ da
            db = np.sum(da, axis=0)
            grads[i] = {"dW": dW, "db": db}
            da = da @ layer["W"].T
            act_idx -= 1
            pre_idx -= 1

        elif layer["type"] == "relu":
            alpha = layer.get("alpha", 0.01)
            z = pre_activations[pre_idx]
            da = da * np.where(z > 0, 1.0, alpha)
            act_idx -= 1
            pre_idx -= 1

        elif layer["type"] == "sigmoid":
            sig = activations[act_idx]
            da = da * sig * (1.0 - sig)
            act_idx -= 1
            pre_idx -= 1

        if grads[i] is None:
            grads[i] = {}

    return grads, loss
