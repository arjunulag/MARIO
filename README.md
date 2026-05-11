<h1 align="center">From-Scratch NumPy ML Library</h1>

<p align="center">
  <strong>MLPs, a CNN with full autodiff, a transformer, four optimizers, and a DQN agent — all built on NumPy. No PyTorch, no TensorFlow.</strong>
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
  <img src="https://img.shields.io/badge/autodiff-Tensor-purple" alt="Autodiff">
  <img src="https://img.shields.io/badge/env-Gymnasium-red" alt="Gymnasium">
  <img src="https://img.shields.io/badge/renderer-OpenGL-5586A4?logo=opengl&logoColor=white" alt="OpenGL">
</p>

---

## Overview

This repository implements the core pieces of a deep learning stack from the ground up using only NumPy:

- An MLP builder with manual forward passes and full backpropagation
- A reverse-mode autodiff `Tensor` class supporting `conv2d`, `maxpool2d`, `relu`, `linear`, `flatten`, and `softmax_cross_entropy`
- A CNN forward/backward pipeline for processing Super Mario Bros frames (4×84×84 stacked grayscale)
- A from-scratch transformer (configurable heads/layers/d_model, pre- or post-norm, causal masking)
- Four interchangeable optimizers (SGD, Momentum, RMSProp, Adam)
- A DQN reinforcement learning agent with experience replay, ε-greedy, target network, and inline Adam with gradient clipping
- A **4D CartPole** Gymnasium environment with a dual-view OpenGL + pygame renderer

**Highlights:**

- Two parallel modeling stacks: the static MLP layer-dict approach (`Parameter_init` + `compute_gradients`) and the dynamic autodiff graph approach (`Tensor`)
- Manual backprop for conv2d, maxpool2d, linear, leaky ReLU, flatten, and softmax cross-entropy — every gradient written by hand
- Mario frame preprocessing pipeline (grayscale → 84×84 → stack of 4) feeding a CNN head
- 12-dim / 6-action 4D CartPole environment, rendered as a 3D OpenGL scene plus a 2D W-axis side panel

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

### 4. Run the CNN autodiff demo

```bash
# Trains a small conv→relu→maxpool→linear→softmax-CE network for 20 steps
python cnn_autodiff_backprop.py
```

### 5. Optional: web playground

The `web_mario/` directory hosts a small FastAPI backend plus a static frontend (see `web_mario/instructions.md`).

## Training

### Commands

```bash
# Default DQN training run
python train_cartpole4d.py

# Explicit train subcommand with live rendering
python train_cartpole4d.py train --render

# Demo mode — load weights and play
python train_cartpole4d.py demo --weights best_cartpole4d.npy

# Standalone CNN autodiff sanity check
python cnn_autodiff_backprop.py
```

### Key Hyperparameters (DQN)

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

### Neural Network Core — Static MLP stack

Built in `functions.py`, `gradient.py`, and `mse_vector.py` — no external ML libraries.

| Module | Purpose |
|---|---|
| `functions.py` | `Parameter_init` builds an MLP from a list of dicts (`{"type": "linear", "in": N, "out": M}` / `{"type": "relu"}` / `{"type": "sigmoid"}`). Uses He init for relu, Xavier for sigmoid. |
| `gradient.py` | `compute_gradients(model, x, y)` — manual backprop, returns grads aligned with `model.layers` plus loss. |
| `mse_vector.py` | MSE loss and derivative, supports 1D and batched 2D inputs. |
| `linear+activation.py` | Standalone linear + activation forward with a cache for backprop, used as a teaching reference. |

### Autodiff Tensor — Dynamic graph stack

A reverse-mode autodiff engine built around a small `Tensor` class that records its parents on every op and replays a topological backward pass.

| Module | Purpose |
|---|---|
| `Tensor.py` | Minimal `Tensor` with `conv2d`, `relu` (leaky, α=0.01), `linear`, `flatten`, and `backward()`. |
| `cnn_autodiff_backprop.py` | Extended `Tensor` adding `maxpool2d`, conv2d padding/bias, and `softmax_cross_entropy`. Contains a 20-step training demo on a synthetic 1×8×8 input. |
| `CNN_network.py` | `build_weights()` + `forward()` — a two-conv-layer + two-linear-layer CNN producing a 64-dim feature vector from a 4×84×84 frame stack. |

### Mario Environment & Frame Pipeline

| File | Purpose |
|---|---|
| `mario_env.py` | Gymnasium-compatible Super Mario Bros environment factory (`gym_super_mario_bros` + `nes_py`), `SIMPLE_MOVEMENT` action space, frame skip/stack hooks, reward shaping. |
| `preprocessFrames.py` | Grayscale → resize-to-84×84 → normalize-to-[0,1]; `build_initial_stack()` returns a length-4 deque + stacked array. |
| `cnnMarioViewer.py` | Glue script: builds the Mario env, sets up SIMPLE_MOVEMENT, primes the frame stack, and runs the CNN forward pass. |
| `race.py` | Human-vs-ghost race mode — keyboard control through pygame with a stub for replaying saved trajectories. |

### Transformer

| File | Purpose |
|---|---|
| `transformer.py` | From-scratch transformer with `TransformerConfig` (vocab size, d_model, heads, layers, d_ff, max seq len, causal masking, pre/post-norm, FFN activation, tied embeddings, seed). |

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
| `cartpole3d_env.py` | 3D precursor environment used during bring-up. |
| `train_cartpole4d.py` | CLI training harness with `train` / `demo` subcommands. |
| `demo.py` | Loads weights and plays back a trained agent. |
| `hyperparam_search.py` | Hyperparameter sweep helper. |

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

### Web

| Path | Purpose |
|---|---|
| `web/` | Static HTML/CSS/JS playground (`index.html`, `play.html`, `about.html`). |
| `web_mario/backend/` | FastAPI/uvicorn server (`server.py`). |
| `web_mario/frontend/` | Static frontend served on port 3000. See `web_mario/instructions.md`. |

## Project Structure

```
MARIO/
├── functions.py                 # Parameter_init: MLP builder, forward pass
├── gradient.py                  # compute_gradients: manual backprop (MLP)
├── mse_vector.py                # MSE loss + derivative
├── linear+activation.py         # Standalone linear+activation forward+cache
├── Tensor.py                    # Autodiff Tensor: conv2d, relu, linear, flatten
├── cnn_autodiff_backprop.py     # Extended Tensor (maxpool, padded conv, softmax-CE) + demo
├── CNN_network.py               # CNN head for 4×84×84 Mario frames
├── transformer.py               # From-scratch transformer + TransformerConfig
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
├── mario_env.py                 # Gymnasium-compatible Super Mario Bros env factory
├── preprocessFrames.py          # Grayscale + 84×84 resize + 4-frame stack
├── cnnMarioViewer.py            # Mario env + CNN forward pass driver
├── race.py                      # Human-vs-ghost pygame race mode
├── linear_regression.py         # Linear regression example
├── housing_predict.py           # Boston Housing regression example
├── Housing.csv                  # Housing dataset
├── key_formulas.txt             # Cheat-sheet of derivations
├── requirements.txt             # NumPy, matplotlib, Gymnasium, pygame, PyOpenGL
├── web/                         # Static HTML/CSS/JS playground
├── web_mario/                   # FastAPI backend + static frontend
├── tests/                       # pytest suite
└── CLAUDE.md                    # Guidance for Claude Code
```

## Design Notes

- **Two modeling stacks coexist.** The static MLP stack (`Parameter_init` + `compute_gradients`) is used by the DQN agent; the dynamic autodiff stack (`Tensor`) powers the CNN that consumes Mario frames. They don't share code on purpose — the static stack is faster for fixed MLPs, the autodiff stack scales to arbitrary graphs.
- **Leaky ReLU α = 0.01** is the default activation everywhere. Forward and backward passes must use the same α — a prior bug came from an α mismatch in `gradient.py`.
- **DQN uses its own inline Adam** (not `adam.py`) because it needs per-step updates with gradient clipping, not epoch-based training.
- **Weights** are saved and loaded as `.npy` files with `allow_pickle=True`.
- **Tuning history:** the original DQN hyperparameters diverged. Fix: lower LR to 1e-4, grow replay buffer to 100k, raise target sync to 1000 steps.
- **CNN input contract:** the Mario CNN expects a 4×84×84 float array (four stacked grayscale frames normalized to [0,1]) and produces a 64-dim feature vector — not Q-values directly.

## License

See [LICENSE](LICENSE).
