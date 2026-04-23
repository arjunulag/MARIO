<h1 align="center">From-Scratch NumPy ML Library</h1>

<p align="center">
  <strong>Neural networks, optimizers, and a DQN agent — built entirely on NumPy, no PyTorch or TensorFlow.</strong>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#training">Training</a> &bull;
  <a href="#architecture">Architecture</a> &bull;
  <a href="#project-structure">Project Structure</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/NumPy-2.0%2B-013243?logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/algorithm-DQN-green" alt="DQN">
  <img src="https://img.shields.io/badge/env-Gymnasium-red" alt="Gymnasium">
  <img src="https://img.shields.io/badge/renderer-OpenGL-5586A4?logo=opengl&logoColor=white" alt="OpenGL">
</p>

---

## Overview

This repository implements the core pieces of a deep learning stack from the ground up using only NumPy: an MLP builder with manual forward passes, full backpropagation, four optimizers (SGD, Momentum, RMSProp, Adam), and a DQN reinforcement learning agent. Everything culminates in a **4D CartPole** environment — a cart that moves on three spatial axes (X, Z, W) with a pole that can tilt in three angular DOFs — rendered live via a dual-view OpenGL + pygame setup.

**Highlights:**

- Neural networks built layer-by-layer from config dicts, weights stored as plain NumPy arrays
- Manual backpropagation through linear + leaky ReLU / sigmoid layers (MSE loss)
- Four interchangeable optimizers sharing the same `compute_gradients` interface
- DQN agent with experience replay, epsilon-greedy, target network, and inline Adam with gradient clipping
- 12-dim / 6-action 4D CartPole Gymnasium environment
- 3D OpenGL renderer with a 2D side panel for the W axis

## Quick Start

### 1. Clone and set up

```bash
git clone https://github.com/arjunulag/MARIO.git
cd MARIO

python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Train the 4D CartPole DQN agent

```bash
# Headless training
python train_cartpole4d.py

# Train with rendering every 50 episodes
python train_cartpole4d.py train --render
```

### 3. Watch a trained agent

```bash
python train_cartpole4d.py demo --weights best_cartpole4d.npy
```

## Training

### Commands

```bash
# Default training run
python train_cartpole4d.py

# Explicit train subcommand with live rendering
python train_cartpole4d.py train --render

# Demo mode — load weights and play
python train_cartpole4d.py demo --weights best_cartpole4d.npy
```

### Key Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| Learning rate | `0.0001` | Reduced to prevent divergence |
| Replay buffer | `100,000` | Sized up after divergence issues |
| Target network sync | `1,000 steps` | Hard sync, not Polyak |
| Activation | Leaky ReLU (α=0.01) | Forward and backward must match α |
| Optimizer (DQN) | Inline Adam + grad clipping | Not the epoch-based `adam.py` |
| Loss | MSE | See `mse_vector.py` |

### Tests

```bash
# Full suite
python -m pytest tests/

# Single file
python -m pytest tests/test_parameter_init.py
```

## Architecture

### Neural Network Core

Built in `functions.py`, `gradient.py`, and `mse_vector.py` — no external ML libraries.

| Module | Purpose |
|---|---|
| `functions.py` | `Parameter_init` builds an MLP from a list of dicts (`{"type": "linear", "in": N, "out": M}` / `{"type": "relu"}` / `{"type": "sigmoid"}`). Uses He init for relu, Xavier for sigmoid. |
| `gradient.py` | `compute_gradients(model, x, y)` — manual backprop, returns grads aligned with `model.layers` plus loss. |
| `mse_vector.py` | MSE loss and derivative, supports 1D and batched 2D inputs. |

### Optimizers

All optimizers consume gradients from `compute_gradients` and apply updates in-place on the layer dicts.

| File | Algorithm |
|---|---|
| `gradient_descent.py` | Vanilla SGD |
| `momentum.py` | SGD with momentum |
| `rmsprop.py` | RMSProp |
| `adam.py` | Adam |

### DQN Reinforcement Learning

| File | Role |
|---|---|
| `dqn_agent.py` | `DQNAgent` — experience replay, ε-greedy, target network, inline Adam with gradient clipping. Both Q-network and target network are `Parameter_init` MLPs. |
| `cartpole4d_env.py` | `CartPole4DEnv` Gymnasium environment: 12-dim state, 6 discrete actions (±push on X, Z, W). |
| `train_cartpole4d.py` | CLI training harness with `train` / `demo` subcommands. |

### Rendering

| File | Purpose |
|---|---|
| `cartpole3d_renderer.py` | 3D OpenGL scene for the X/Y/Z dimensions |
| `cartpole4d_renderer.py` | Adds a 2D side panel visualising the W axis |

### 4D CartPole Environment

The cart lives in 4D space but is rendered as a 3D scene plus a 2D strip:

- **State (12-dim):** positions and velocities along X/Z/W plus pole angles and angular velocities
- **Actions (6 discrete):** positive or negative push along each of the three spatial axes
- **Reward:** standard CartPole-style — survive longer, score higher

## Project Structure

```
MARIO/
├── functions.py                 # Parameter_init: MLP builder, forward pass
├── gradient.py                  # compute_gradients: manual backprop
├── mse_vector.py                # MSE loss + derivative
├── gradient_descent.py          # Vanilla SGD
├── momentum.py                  # SGD with momentum
├── rmsprop.py                   # RMSProp
├── adam.py                      # Adam optimizer
├── dqn_agent.py                 # DQN agent (replay, target net, inline Adam)
├── cartpole4d_env.py            # 4D CartPole Gymnasium environment
├── cartpole3d_env.py            # 3D CartPole environment
├── cartpole3d_renderer.py       # OpenGL renderer (3D scene)
├── cartpole4d_renderer.py       # OpenGL renderer with W-axis panel
├── train_cartpole4d.py          # Training CLI (train / demo subcommands)
├── demo.py                      # Plays a trained agent from saved weights
├── hyperparam_search.py         # Hyperparameter sweep helper
├── linear_regression.py         # Linear regression example
├── housing_predict.py           # Boston Housing regression example
├── Housing.csv                  # Housing dataset
├── requirements.txt             # NumPy, Gymnasium, pygame, PyOpenGL
├── tests/                       # pytest suite
└── CLAUDE.md                    # Guidance for Claude Code
```

## Design Notes

- **Leaky ReLU α = 0.01** is the default activation. The backward pass in `gradient.py` must use the same α as the forward pass — a prior bug was caused by an α mismatch.
- **DQN uses its own inline Adam** (not `adam.py`) because it needs per-step updates with gradient clipping, not epoch-based training.
- **Weights** are saved and loaded as `.npy` files with `allow_pickle=True`.
- Tuning history: the original hyperparameters diverged. Fix: lower LR to 1e-4, grow replay buffer to 100k, raise target sync to 1000 steps.

## License

See [LICENSE](LICENSE).
