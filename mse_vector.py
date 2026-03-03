import numpy as np

def mse_vector(y_hat: np.ndarray, y: np.ndarray) -> float:
    """
    y_hat: (d,)
    y:     (d,)
    returns: scalar MSE
    """
    if y_hat.shape != y.shape:
        raise ValueError(f"Shape mismatch: {y_hat.shape} vs {y.shape}")

    if y_hat.ndim != 1:
        raise ValueError("Expected 1D vectors")

    d = y_hat.shape[0]
    e = y_hat - y

    return (e @ e) / d