<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/en/a/a9/MarioNSMBUDeluxe.png" alt="Mario" width="120">
</p>

<h1 align="center">Super Mario Bros AI Agent</h1>

<p align="center">
  <strong>A reinforcement learning agent that learns to beat Super Mario Bros using Proximal Policy Optimization</strong>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#training">Training</a> &bull;
  <a href="#architecture">Architecture</a> &bull;
  <a href="#results">Results</a> &bull;
  <a href="#google-colab">Google Colab</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/algorithm-PPO-green" alt="PPO">
  <img src="https://img.shields.io/badge/env-gym--super--mario--bros-red" alt="Environment">
  <img src="https://img.shields.io/badge/SB3-Stable%20Baselines%203-orange" alt="Stable Baselines 3">
</p>

---

## Overview

This project trains a deep reinforcement learning agent to play and beat **Super Mario Bros Level 1-1** from raw pixel input. The agent uses a convolutional neural network to process stacked game frames and outputs movement actions via PPO, a state-of-the-art policy gradient method.

**Highlights:**

- Trains from scratch using only pixel observations (no hand-crafted features)
- Reaches 80%+ level completion rate after ~5M timesteps
- Multiple CNN architectures: Standard, Large, and Attention-based
- Parallel environment training for 4-8x speedup
- Best-run finder that evaluates 10,000 stochastic episodes and exports the fastest clear as MP4
- Full Google Colab support for free GPU training

## Quick Start

### 1. Clone and set up

```bash
git clone https://github.com/<your-username>/Super-Mario-Training.git
cd Super-Mario-Training

python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Train

```bash
python train.py
```

### 3. Watch it play

```bash
python play.py ./checkpoints/best_model.zip
```

## Training

### Basic Usage

```bash
# Default: 5M steps, 8 parallel envs
python train.py

# Quick test run
python train.py --timesteps 100000

# Resume from checkpoint
python train.py --resume ./checkpoints/mario_ppo_1000000_steps.zip

# Force GPU
python train.py --device cuda
```

### All Options

| Flag | Default | Description |
|---|---|---|
| `--timesteps` | `5000000` | Total training steps |
| `--n-envs` | `8` | Parallel environments |
| `--lr` | `2.5e-4` | Learning rate |
| `--batch-size` | `256` | Minibatch size |
| `--n-steps` | `512` | Steps per rollout |
| `--n-epochs` | `4` | PPO update epochs |
| `--gamma` | `0.99` | Discount factor |
| `--gae-lambda` | `0.95` | GAE lambda |
| `--clip-range` | `0.2` | PPO clipping range |
| `--ent-coef` | `0.01` | Entropy bonus (exploration) |
| `--model-type` | `standard` | Architecture: `standard`, `large`, `attention` |
| `--device` | `auto` | `cuda`, `cpu`, or `auto` |
| `--resume` | &mdash; | Checkpoint path to resume from |

### Monitor with TensorBoard

```bash
tensorboard --logdir ./logs
```

Open [http://localhost:6006](http://localhost:6006) to view training curves including reward, episode length, x-position, and level completions.

## Evaluation

### Watch the Agent

```bash
# Play 5 episodes
python play.py ./checkpoints/best_model.zip

# Slower playback
python play.py ./checkpoints/best_model.zip --episodes 10 --fps 20

# Record MP4
python play.py ./checkpoints/best_model.zip --record
```

### Benchmark (Headless)

```bash
python play.py ./checkpoints/best_model.zip --benchmark --benchmark-episodes 100
```

### Find the Best Run

Run 10,000 stochastic episodes and automatically save the fastest level completion as an MP4:

```bash
python best_run.py

# With live rendering
python best_run_visual.py

# Custom options
python best_run.py --episodes 5000 --model ./checkpoints/my_model --fps 15
```

### Train and Watch

Combines training with periodic visual evaluation. Tracks the fastest clear and saves it when done:

```bash
python train_and_watch.py
python train_and_watch.py --timesteps 2000000 --eval-episodes 20
```

## Architecture

### Observation Pipeline

Raw 240x256 RGB frames go through a preprocessing pipeline before reaching the network:

```
Raw Frame (240×256×3 RGB)
  → Frame Skip (repeat action for 4 frames)
  → Reward Shaping (progress bonus, death penalty)
  → Grayscale (240×256×1)
  → Resize (84×84×1)
  → Frame Stack (84×84×4)
  → Normalize (pixel values → [0, 1])
```

### CNN Models

| Model | Description | Use Case |
|---|---|---|
| **Standard** | Nature DQN-style: 3 conv layers → 512-d FC | Default, good speed/performance balance |
| **Large** | 4 conv layers + batch norm + dropout → 1024 → 512 FC | Long training runs, harder levels |
| **Attention** | 3 conv layers + spatial attention gate → 512-d FC | Experimental, focuses on enemies/gaps |

All models feed into shared policy (256→256) and value (256→256) heads.

### Action Space

The agent chooses from 7 simplified actions each step:

| Index | Action |
|---|---|
| 0 | No-op |
| 1 | Right |
| 2 | Right + Jump |
| 3 | Right + Run |
| 4 | Right + Jump + Run |
| 5 | Jump |
| 6 | Left |

### Reward Function

| Component | Value | Purpose |
|---|---|---|
| Base game reward | varies | Coins, score, enemies |
| Forward progress | +0.1 / pixel | Encourage rightward movement |
| Flag reached | +100 | Level completion bonus |
| Death | −50 | Penalize losing a life |

## Results

Expected performance with default hyperparameters:

| Timesteps | Behavior | Approx. X-Position |
|---|---|---|
| 500K | Learns basic movement | ~500 |
| 1M | Clears first obstacles | ~1000 |
| 2M | Reaches mid-level consistently | ~2000 |
| 3M | Starts completing the level (~50%) | ~3200+ |
| 5M+ | Consistent completions (80%+) | Flag |

## Google Colab

A ready-to-run notebook is included for training on a free GPU:

1. Upload `Mario_Training_Colab.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Set runtime to **GPU** (`Runtime → Change runtime type → GPU`)
3. Run all cells &mdash; training takes roughly 1 hour for a level-beating agent

The Colab notebook handles all dependency installation automatically.

## Project Structure

```
Super Mario Training/
├── train.py                 # Main training script (PPO + callbacks)
├── play.py                  # Watch or benchmark a trained agent
├── model.py                 # CNN architectures (Standard, Large, Attention)
├── wrappers.py              # Env preprocessing (frame stack, grayscale, rewards)
├── best_run.py              # Find fastest clear over 10K stochastic runs → MP4
├── best_run_visual.py       # Same as above with live rendering
├── train_and_watch.py       # Train with periodic visual evaluation
├── colab_setup.py           # Dependency installer for Google Colab
├── Mario_Training_Colab.ipynb  # Colab notebook
├── requirements.txt         # Python dependencies
├── checkpoints/             # Saved model weights (created during training)
├── logs/                    # TensorBoard logs (created during training)
└── recordings/              # Exported gameplay videos
```

## Troubleshooting

<details>
<summary><strong>NumPy 2.0 compatibility error</strong></summary>

`nes-py` requires NumPy < 2.0. Downgrade with:

```bash
pip install "numpy>=1.24.0,<2.0.0"
```

</details>

<details>
<summary><strong>Out of memory</strong></summary>

Reduce the number of parallel environments:

```bash
python train.py --n-envs 4
```

</details>

<details>
<summary><strong>Agent not learning</strong></summary>

Try increasing exploration or lowering the learning rate:

```bash
python train.py --ent-coef 0.02 --lr 1e-4
```

</details>

<details>
<summary><strong>Slow training on CPU</strong></summary>

Use a GPU if available, or reduce environment count:

```bash
python train.py --device cuda
```

</details>

## References

- Schulman et al. &mdash; [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (2017)
- [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros) &mdash; OpenAI Gym environment
- [Stable Baselines 3](https://stable-baselines3.readthedocs.io/) &mdash; RL algorithm implementations

## License

This project is for educational purposes. Super Mario Bros is a trademark of Nintendo.
