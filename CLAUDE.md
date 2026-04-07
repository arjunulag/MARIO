# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

From-scratch machine learning library built entirely on NumPy — no PyTorch/TensorFlow. The repo implements neural networks, backpropagation, optimizers, and a DQN reinforcement learning agent, culminating in a 4D CartPole environment with a 3D OpenGL renderer.

The README describes a Super Mario Bros PPO agent, but the actual codebase is the from-scratch NumPy ML stack described below.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Train the 4D CartPole DQN agent (headless, 80k episodes default)
python train_cartpole4d.py

# Train with rendering (shows every 50 episodes)
python train_cartpole4d.py train --render

# Demo a trained agent
python train_cartpole4d.py demo --weights best_cartpole4d.npy

# Run tests
python -m pytest tests/

# Run a single test
python -m pytest tests/test_parameter_init.py
```

## Architecture

### Neural Network Core (no external ML libraries)

- **`functions.py`** — `Parameter_init` class: builds an MLP from a config list of dicts. Each layer is either `{"type": "linear", "in": N, "out": M}` or `{"type": "relu"}` / `{"type": "sigmoid"}`. Layers are stored as plain dicts with `"W"`, `"b"` keys. Weight init uses He (relu) or Xavier (sigmoid) based on `activation_hint`. Also contains `forward_verbose()` for debugging.
- **`gradient.py`** — `compute_gradients(model, x, y)`: full manual backprop through the layer list. Returns `(grads, loss)` where grads is a list of `{"dW", "db"}` dicts aligned with `model.layers`. Uses MSE loss from `mse_vector.py`.
- **`mse_vector.py`** — MSE loss and its derivative, supporting both 1D and batched 2D inputs.

### Optimizers (all use `compute_gradients`)

- **`gradient_descent.py`** — vanilla SGD
- **`momentum.py`** — SGD with momentum
- **`rmsprop.py`** — RMSProp
- **`adam.py`** — Adam optimizer

### DQN Reinforcement Learning

- **`dqn_agent.py`** — `DQNAgent` class: experience replay, epsilon-greedy, target network with periodic hard sync, inline Adam with gradient clipping. Uses `Parameter_init` for both Q-network and target network.
- **`cartpole4d_env.py`** — `CartPole4DEnv` (Gymnasium): cart moves on 3 spatial axes (X, Z, W) with pole that can tilt in 3 angular DOFs. 12-dim state, 6 discrete actions (push +/- on each axis).
- **`train_cartpole4d.py`** — training harness with CLI (train/demo subcommands).

### Rendering

- **`cartpole3d_renderer.py`** / **`cartpole4d_renderer.py`** — OpenGL + pygame dual-view renderer (3D scene for X/Y/Z + 2D panel for W axis).

## Key Design Decisions

- Leaky ReLU with alpha=0.01 is the default activation (not standard ReLU). The gradient code in `gradient.py` must match this alpha — a prior bug was caused by alpha mismatch between forward and backward passes.
- The DQN agent has its own inline Adam optimizer (not using `adam.py`) because it needs per-step updates with gradient clipping, not epoch-based training.
- Model weights are saved/loaded as `.npy` files with `allow_pickle=True`.
- Recent hyperparameter tuning: learning rate reduced to 0.0001, buffer size increased to 100k, target sync interval increased to 1000 steps — these changes fixed a diverging model issue.
