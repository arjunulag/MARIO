"""
Numerical gradient checker for the from-scratch backprop in `gradient.py`.

Hand-written backward passes are easy to get subtly wrong — the canonical
example in this repo (see CLAUDE.md) was a Leaky-ReLU `alpha` mismatch between
the forward and backward passes, which silently corrupts every gradient that
flows through a negative pre-activation.

This module catches that whole class of bug by comparing the analytical
gradients from `compute_gradients` against a finite-difference estimate of the
same gradients. If the two disagree by more than a small tolerance, the
backward pass is wrong somewhere.

Usage
-----
    # As a script — runs a few representative networks and reports rel. error
    python gradient_check.py

    # As a library
    from gradient_check import gradient_check
    max_err = gradient_check(model, x, y)
    assert max_err < 1e-5
"""

import numpy as np

from functions import Parameter_init
from gradient import compute_gradients
from mse_vector import mse_vector


def numerical_gradient(model, x, y, eps=1e-5):
    """
    Estimate dL/dW and dL/db for every linear layer using central finite
    differences:  df/dp ~= (f(p + eps) - f(p - eps)) / (2 * eps).

    Returns a list aligned with `model.layers`; linear layers get a dict with
    "dW"/"db" keys, everything else gets an empty dict.
    """
    grads = [None] * len(model.layers)

    for i, layer in enumerate(model.layers):
        if layer["type"] != "linear":
            grads[i] = {}
            continue

        dW = np.zeros_like(layer["W"])
        db = np.zeros_like(layer["b"])

        # Perturb each weight independently.
        it = np.nditer(layer["W"], flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            original = layer["W"][idx]

            layer["W"][idx] = original + eps
            loss_plus = mse_vector(model.forward(x), _match(y, model.forward(x)))

            layer["W"][idx] = original - eps
            loss_minus = mse_vector(model.forward(x), _match(y, model.forward(x)))

            layer["W"][idx] = original  # restore
            dW[idx] = (loss_plus - loss_minus) / (2 * eps)
            it.iternext()

        # Perturb each bias independently.
        for j in range(layer["b"].shape[0]):
            original = layer["b"][j]

            layer["b"][j] = original + eps
            loss_plus = mse_vector(model.forward(x), _match(y, model.forward(x)))

            layer["b"][j] = original - eps
            loss_minus = mse_vector(model.forward(x), _match(y, model.forward(x)))

            layer["b"][j] = original  # restore
            db[j] = (loss_plus - loss_minus) / (2 * eps)

        grads[i] = {"dW": dW, "db": db}

    return grads


def _match(y, y_hat):
    """`mse_vector` requires matching shapes; broadcast a 1D target to 2D."""
    if y.shape == y_hat.shape:
        return y
    if y.ndim == 1 and y_hat.ndim == 2 and y_hat.shape[0] == 1:
        return y[np.newaxis, :]
    return y


def _rel_error(a, b, eps=1e-12):
    """Symmetric relative error, robust when both values are near zero."""
    return np.abs(a - b) / (np.abs(a) + np.abs(b) + eps)


def gradient_check(model, x, y, eps=1e-5):
    """
    Compare analytical gradients (`compute_gradients`) to a numerical estimate.

    Returns the maximum relative error across every dW and db in the network.
    A correct backward pass with eps=1e-5 typically yields < 1e-6.
    """
    analytic, _ = compute_gradients(model, x, y)
    numeric = numerical_gradient(model, x, y, eps=eps)

    max_err = 0.0
    for layer_a, layer_n in zip(analytic, numeric):
        if "dW" not in layer_a:
            continue
        for key in ("dW", "db"):
            err = np.max(_rel_error(layer_a[key], layer_n[key]))
            max_err = max(max_err, float(err))
    return max_err


def _demo():
    """Run the checker on a few representative architectures."""
    np.random.seed(0)

    configs = {
        "linear only": [
            {"type": "linear", "in": 4, "out": 3},
        ],
        "relu MLP": [
            {"type": "linear", "in": 4, "out": 6, "activation_hint": "relu"},
            {"type": "relu"},
            {"type": "linear", "in": 6, "out": 2, "activation_hint": "relu"},
        ],
        "sigmoid MLP": [
            {"type": "linear", "in": 3, "out": 5, "activation_hint": "sigmoid"},
            {"type": "sigmoid"},
            {"type": "linear", "in": 5, "out": 2, "activation_hint": "sigmoid"},
        ],
        "mixed deep": [
            {"type": "linear", "in": 5, "out": 8, "activation_hint": "relu"},
            {"type": "relu"},
            {"type": "linear", "in": 8, "out": 4, "activation_hint": "sigmoid"},
            {"type": "sigmoid"},
            {"type": "linear", "in": 4, "out": 3},
        ],
    }

    print(f"{'network':<14} {'in->out':<10} {'max rel error':<16} verdict")
    print("-" * 55)
    all_ok = True
    for name, config in configs.items():
        model = Parameter_init(config)
        in_dim = config[0]["in"]
        out_dim = next(c["out"] for c in reversed(config) if c.get("type") == "linear")

        # Batch of inputs deliberately spanning negative values so the
        # Leaky-ReLU negative branch is exercised.
        x = np.random.randn(4, in_dim)
        y = np.random.randn(4, out_dim)

        err = gradient_check(model, x, y)
        ok = err < 1e-5
        all_ok = all_ok and ok
        verdict = "PASS" if ok else "FAIL"
        print(f"{name:<14} {f'{in_dim}->{out_dim}':<10} {err:<16.3e} {verdict}")

    print("-" * 55)
    print("ALL PASS" if all_ok else "SOME CHECKS FAILED")
    return all_ok


if __name__ == "__main__":
    import sys

    sys.exit(0 if _demo() else 1)
