import numpy as np

def mse_vector(y_hat: np.ndarray, y: np.ndarray) -> float:
    """
    y_hat: (d,) or (N, d)
    y:     (d,) or (N, d)
    returns: scalar MSE
    """
    if y_hat.shape != y.shape:
        raise ValueError(f"Shape mismatch: {y_hat.shape} vs {y.shape}")

    if y_hat.ndim == 1:
        d = y_hat.shape[0]
        e = y_hat - y
        return (e @ e) / d

    if y_hat.ndim == 2:
        N, d = y_hat.shape
        return np.sum((y_hat - y) ** 2) / (N * d)

    raise ValueError(f"Expected 1D or 2D arrays, got {y_hat.ndim}D")


def mse_derivative(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Gradient of MSE with respect to y_hat.

    y_hat: (d,) or (N, d)
    y:     (d,) or (N, d)
    returns: same shape as y_hat
        dL/dy_hat = (2/d) * (y_hat - y)       for 1D
        dL/dy_hat = (2/(N*d)) * (y_hat - y)   for 2D batch
    """
    if y_hat.shape != y.shape:
        raise ValueError(f"Shape mismatch: {y_hat.shape} vs {y.shape}")

    if y_hat.ndim == 1:
        d = y_hat.shape[0]
        return (2.0 / d) * (y_hat - y)

    if y_hat.ndim == 2:
        N, d = y_hat.shape
        return (2.0 / (N * d)) * (y_hat - y)

    raise ValueError(f"Expected 1D or 2D arrays, got {y_hat.ndim}D")