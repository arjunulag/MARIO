"""
Pytest wiring for the numerical gradient checker.

These tests are the safety net for `gradient.py`'s hand-written backward pass.
If anyone breaks the analytical gradients — for example by reintroducing the
Leaky-ReLU `alpha` mismatch between forward and backward — the relative error
against finite differences blows up and these tests fail loudly.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from functions import Parameter_init  # noqa: E402
from gradient import compute_gradients  # noqa: E402
from gradient_check import gradient_check, numerical_gradient  # noqa: E402

TOL = 1e-5


def _make(config, seed=0):
    np.random.seed(seed)
    model = Parameter_init(config)
    in_dim = config[0]["in"]
    out_dim = next(c["out"] for c in reversed(config) if c.get("type") == "linear")
    x = np.random.randn(4, in_dim)
    y = np.random.randn(4, out_dim)
    return model, x, y


CONFIGS = {
    "linear_only": [
        {"type": "linear", "in": 4, "out": 3},
    ],
    "relu_mlp": [
        {"type": "linear", "in": 4, "out": 6, "activation_hint": "relu"},
        {"type": "relu"},
        {"type": "linear", "in": 6, "out": 2, "activation_hint": "relu"},
    ],
    "sigmoid_mlp": [
        {"type": "linear", "in": 3, "out": 5, "activation_hint": "sigmoid"},
        {"type": "sigmoid"},
        {"type": "linear", "in": 5, "out": 2, "activation_hint": "sigmoid"},
    ],
    "mixed_deep": [
        {"type": "linear", "in": 5, "out": 8, "activation_hint": "relu"},
        {"type": "relu"},
        {"type": "linear", "in": 8, "out": 4, "activation_hint": "sigmoid"},
        {"type": "sigmoid"},
        {"type": "linear", "in": 4, "out": 3},
    ],
}


@pytest.mark.parametrize("name", list(CONFIGS.keys()))
def test_gradients_match_finite_differences(name):
    model, x, y = _make(CONFIGS[name])
    max_err = gradient_check(model, x, y)
    assert max_err < TOL, f"{name}: analytical vs numerical gradients diverge ({max_err:.2e})"


def test_custom_relu_alpha_is_respected():
    """
    A non-default Leaky-ReLU alpha must flow through both passes. This is the
    exact regression CLAUDE.md warns about: if the backward pass hardcodes a
    different alpha than the forward pass, this check fails.
    """
    config = [
        {"type": "linear", "in": 3, "out": 4, "activation_hint": "relu"},
        {"type": "relu", "alpha": 0.2},
        {"type": "linear", "in": 4, "out": 2},
    ]
    model, x, y = _make(config)
    # Force negative pre-activations so the alpha branch actually matters.
    x = -np.abs(x)
    assert gradient_check(model, x, y) < TOL


def test_numerical_gradient_does_not_mutate_parameters():
    """The finite-difference probe must restore every parameter it perturbs."""
    model, x, y = _make(CONFIGS["relu_mlp"])
    before = [
        (layer["W"].copy(), layer["b"].copy())
        for layer in model.layers
        if layer["type"] == "linear"
    ]
    numerical_gradient(model, x, y)
    after = [
        (layer["W"], layer["b"])
        for layer in model.layers
        if layer["type"] == "linear"
    ]
    for (w0, b0), (w1, b1) in zip(before, after):
        assert np.array_equal(w0, w1)
        assert np.array_equal(b0, b1)


def test_single_sample_input():
    """compute_gradients accepts a 1D sample; the checker should too."""
    np.random.seed(1)
    config = [
        {"type": "linear", "in": 4, "out": 4, "activation_hint": "relu"},
        {"type": "relu"},
        {"type": "linear", "in": 4, "out": 2},
    ]
    model = Parameter_init(config)
    x = np.random.randn(4)
    y = np.random.randn(2)
    # Sanity: the analytical pass runs on a 1D sample without error.
    grads, loss = compute_gradients(model, x, y)
    assert np.isfinite(loss)
    assert any("dW" in g for g in grads)
