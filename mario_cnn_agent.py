"""Bridge: from-scratch autodiff CNN <-> Super Mario Bros env.

Pulls the wrapped Gymnasium env from `mario_env.make_mario_env` (gives
(4, 84, 84) uint8 stacked frames) and runs it through the autodiff
`Tensor` ops in `cnn_autodiff_backprop.py`.

Two modes:

    python mario_cnn_agent.py play  --weights mario_cnn.npy
    python mario_cnn_agent.py train --episodes 50 --max-steps 200

Training is REINFORCE: roll out one episode, compute discounted returns,
then per transition seed the softmax-cross-entropy loss gradient with the
(normalised) return G_t and run the autodiff backward to accumulate
parameter gradients across the episode. Adam takes one step per episode.
"""

from __future__ import annotations

import argparse
import time
import numpy as np

from cnn_autodiff_backprop import Tensor, he_init
from mario_env import MarioEnvConfig, make_mario_env


# ── network ──────────────────────────────────────────────────────────────

IN_CHANNELS = 4  # FrameStack=4 over grayscale

# Conv1: 8 filters 8x8 stride 4   (4,84,84) -> (8,20,20)
# Conv2: 16 filters 4x4 stride 2  (8,20,20) -> (16,9,9) = 1296
# FC1: 1296 -> 128
# FC2: 128 -> n_actions
HIDDEN_FC = 128


def build_cnn(n_actions: int) -> dict[str, Tensor]:
    p = {
        "K1":  Tensor(he_init((8, IN_CHANNELS, 8, 8), IN_CHANNELS * 8 * 8)),
        "b1":  Tensor(np.zeros(8)),
        "K2":  Tensor(he_init((16, 8, 4, 4), 8 * 4 * 4)),
        "b2":  Tensor(np.zeros(16)),
        "W1":  Tensor(he_init((HIDDEN_FC, 1296), 1296)),
        "bf1": Tensor(np.zeros(HIDDEN_FC)),
        "W2":  Tensor(he_init((n_actions, HIDDEN_FC), HIDDEN_FC)),
        "bf2": Tensor(np.zeros(n_actions)),
    }
    return p


def forward(state: np.ndarray, params: dict[str, Tensor]) -> Tensor:
    """state: (4, 84, 84) float32 in [0,1]. Returns logits Tensor shape (n_actions,)."""
    x = Tensor(state)
    x = x.conv2d(params["K1"], params["b1"], stride=4).relu()
    x = x.conv2d(params["K2"], params["b2"], stride=2).relu()
    x = x.flatten()
    x = x.linear(params["W1"], params["bf1"]).relu()
    return x.linear(params["W2"], params["bf2"])


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def preprocess_obs(obs: np.ndarray) -> np.ndarray:
    s = np.asarray(obs, dtype=np.float32)
    if s.max() > 1.5:
        s = s / 255.0
    return s


# ── seeded backward (lets us inject G_t as the loss-gradient seed) ───────

def seeded_backward(out: Tensor, seed: float) -> None:
    topo: list[Tensor] = []
    visited: set[int] = set()

    def build(node: Tensor) -> None:
        if id(node) not in visited:
            visited.add(id(node))
            for parent in node._parents:
                build(parent)
            topo.append(node)

    build(out)
    out.grad = np.asarray(seed, dtype=np.float32).reshape(out.data.shape)
    for node in reversed(topo):
        node._backward()


# ── tiny inline Adam over the param dict ─────────────────────────────────

class Adam:
    def __init__(self, params: dict[str, Tensor], lr: float = 1e-4,
                 b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.params = params
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(p.data) for k, p in params.items()}
        self.v = {k: np.zeros_like(p.data) for k, p in params.items()}
        self.t = 0

    def step(self, grad_clip: float | None = 1.0) -> None:
        self.t += 1
        for k, p in self.params.items():
            g = p.grad
            if grad_clip is not None:
                g = np.clip(g, -grad_clip, grad_clip)
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            p.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)

    def zero_grad(self) -> None:
        for p in self.params.values():
            p.grad = np.zeros_like(p.data)


# ── persistence ──────────────────────────────────────────────────────────

def save_weights(params: dict[str, Tensor], path: str) -> None:
    np.save(path, {k: p.data for k, p in params.items()}, allow_pickle=True)


def load_weights(params: dict[str, Tensor], path: str) -> None:
    data = np.load(path, allow_pickle=True).item()
    for k, p in params.items():
        if k in data:
            p.data = np.asarray(data[k], dtype=np.float32)


# ── modes ────────────────────────────────────────────────────────────────

def play(weights: str | None, episodes: int, render: bool, max_steps: int) -> None:
    cfg = MarioEnvConfig(render_mode="human" if render else None)
    env = make_mario_env(cfg)
    n_actions = env.action_space.n
    params = build_cnn(n_actions)
    if weights:
        load_weights(params, weights)
        print(f"loaded weights from {weights}")

    for ep in range(episodes):
        obs, _ = env.reset()
        total, x_pos = 0.0, 0
        for step in range(max_steps):
            state = preprocess_obs(obs)
            logits = forward(state, params)
            probs = softmax(logits.data)
            action = int(np.random.choice(len(probs), p=probs))
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            x_pos = info.get("x_pos", x_pos)
            if terminated or truncated:
                break
        print(f"episode {ep:03d}  reward={total:8.2f}  steps={step+1:4d}  x_pos={x_pos}")
    env.close()


def train(episodes: int, max_steps: int, lr: float, gamma: float,
          save_path: str, render_every: int) -> None:
    cfg = MarioEnvConfig(render_mode=None)
    env = make_mario_env(cfg)
    n_actions = env.action_space.n
    params = build_cnn(n_actions)
    opt = Adam(params, lr=lr)

    best = -np.inf
    for ep in range(episodes):
        if render_every and ep % render_every == 0:
            env.close()
            env = make_mario_env(MarioEnvConfig(render_mode="human"))

        obs, _ = env.reset()
        traj: list[tuple[np.ndarray, int, float]] = []
        t0 = time.time()
        for step in range(max_steps):
            state = preprocess_obs(obs)
            logits = forward(state, params)
            probs = softmax(logits.data)
            action = int(np.random.choice(len(probs), p=probs))
            obs, reward, terminated, truncated, _ = env.step(action)
            traj.append((state, action, float(reward)))
            if terminated or truncated:
                break

        # discounted returns, baseline-normalised
        rewards = np.array([r for _, _, r in traj], dtype=np.float32)
        G = np.zeros_like(rewards)
        running = 0.0
        for t in reversed(range(len(rewards))):
            running = rewards[t] + gamma * running
            G[t] = running
        if G.std() > 1e-8:
            G = (G - G.mean()) / (G.std() + 1e-8)

        # accumulate gradients across the episode, then one Adam step
        opt.zero_grad()
        for (s, a, _), Gt in zip(traj, G):
            logits = forward(s, params)
            loss = logits.softmax_cross_entropy(a)
            seeded_backward(loss, float(Gt))
        opt.step()

        total = float(rewards.sum())
        dt = time.time() - t0
        print(f"ep {ep:03d}  R={total:8.2f}  len={len(traj):4d}  dt={dt:5.1f}s  eps_done")
        if total > best:
            best = total
            save_weights(params, save_path)
            print(f"  saved new best to {save_path}")

        if render_every and ep % render_every == 0:
            env.close()
            env = make_mario_env(MarioEnvConfig(render_mode=None))

    env.close()


# ── CLI ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_play = sub.add_parser("play", help="rollout with current/loaded weights")
    p_play.add_argument("--weights", default=None)
    p_play.add_argument("--episodes", type=int, default=1)
    p_play.add_argument("--max-steps", type=int, default=2000)
    p_play.add_argument("--no-render", action="store_true")

    p_train = sub.add_parser("train", help="REINFORCE training")
    p_train.add_argument("--episodes", type=int, default=50)
    p_train.add_argument("--max-steps", type=int, default=200)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--gamma", type=float, default=0.99)
    p_train.add_argument("--save", default="mario_cnn.npy")
    p_train.add_argument("--render-every", type=int, default=0,
                         help="render every N episodes (0 = headless)")

    args = parser.parse_args()
    if args.cmd == "play":
        play(args.weights, args.episodes, not args.no_render, args.max_steps)
    else:
        train(args.episodes, args.max_steps, args.lr, args.gamma,
              args.save, args.render_every)


if __name__ == "__main__":
    main()
