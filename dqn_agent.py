"""
DQN Agent — from-scratch Deep Q-Network
========================================
Uses the existing Parameter_init (functions.py) for the neural network and
compute_gradients (gradient.py) for back-propagation.  No PyTorch / TensorFlow.

Implements:
  - Experience replay buffer
  - Epsilon-greedy exploration with decay
  - Target network with periodic hard sync
  - Inline Adam optimizer with gradient clipping
"""

import numpy as np
from functions import Parameter_init
from gradient import compute_gradients


class ReplayBuffer:
    """Fixed-size circular buffer of (s, a, r, s', done) transitions."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer   = []
        self.pos      = 0

    def push(self, state, action, reward, next_state, done):
        transition = (
            np.array(state, dtype=np.float64),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float64),
            float(done),
        )
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch   = [self.buffer[i] for i in indices]

        states      = np.array([t[0] for t in batch])
        actions     = np.array([t[1] for t in batch], dtype=np.int64)
        rewards     = np.array([t[2] for t in batch], dtype=np.float64)
        next_states = np.array([t[3] for t in batch])
        dones       = np.array([t[4] for t in batch], dtype=np.float64)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network agent built entirely on the from-scratch numpy MLP.

    Parameters
    ----------
    state_dim         : int   – observation size (12 for CartPole4D)
    action_dim        : int   – number of discrete actions (6 for CartPole4D)
    hidden            : list  – hidden-layer widths
    lr                : float – Adam learning rate
    gamma             : float – discount factor
    epsilon_start/end : float – ε-greedy range
    epsilon_decay     : float – multiplicative decay per training step
    buffer_size       : int   – replay buffer capacity
    batch_size        : int   – mini-batch size for each SGD step
    target_sync_every : int   – copy Q → Q_target every N training steps
    grad_clip         : float – element-wise gradient clamp magnitude
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: list = None,
        lr: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.9995,
        buffer_size: int = 50_000,
        batch_size: int = 64,
        target_sync_every: int = 500,
        grad_clip: float = 1.0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        adam_eps: float = 1e-8,
    ):
        if hidden is None:
            hidden = [128, 128]

        self.state_dim   = state_dim
        self.action_dim  = action_dim
        self.gamma       = gamma
        self.epsilon     = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size  = batch_size
        self.target_sync_every = target_sync_every
        self.lr          = lr
        self.beta1       = beta1
        self.beta2       = beta2
        self.adam_eps     = adam_eps
        self.grad_clip   = grad_clip

        config = self._make_config(state_dim, action_dim, hidden)
        self.q_net    = Parameter_init(config)
        self.q_target = Parameter_init(config)
        self._sync_target()

        self.buffer      = ReplayBuffer(buffer_size)
        self._adam_cache  = self._init_adam()
        self.train_steps = 0

    # ── network config builder ────────────────────────────────────────

    @staticmethod
    def _make_config(s_dim, a_dim, hidden):
        cfg  = []
        prev = s_dim
        for h in hidden:
            cfg.append({"type": "linear", "in": prev, "out": h,
                        "activation_hint": "relu"})
            cfg.append({"type": "relu"})
            prev = h
        cfg.append({"type": "linear", "in": prev, "out": a_dim})
        return cfg

    # ── Adam cache ────────────────────────────────────────────────────

    def _init_adam(self):
        cache = []
        for layer in self.q_net.layers:
            if layer["type"] == "linear":
                cache.append({
                    "m_W": np.zeros_like(layer["W"]),
                    "m_b": np.zeros_like(layer["b"]),
                    "v_W": np.zeros_like(layer["W"]),
                    "v_b": np.zeros_like(layer["b"]),
                })
            else:
                cache.append({})
        return cache

    # ── target network sync ──────────────────────────────────────────

    def _sync_target(self):
        for ql, tl in zip(self.q_net.layers, self.q_target.layers):
            if ql["type"] == "linear":
                tl["W"] = ql["W"].copy()
                tl["b"] = ql["b"].copy()

    # ── public API ───────────────────────────────────────────────────

    def select_action(self, state: np.ndarray) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        q = self.q_net.forward(np.asarray(state, dtype=np.float64))
        return int(np.argmax(q[0]))

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def train(self) -> float | None:
        """One mini-batch DQN update.  Returns loss or None if buffer too small."""
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = \
            self.buffer.sample(self.batch_size)

        # TD targets via target network
        q_next     = self.q_target.forward(next_states)
        max_q_next = np.max(q_next, axis=1)
        td_targets = rewards + self.gamma * max_q_next * (1.0 - dones)

        # Build full target matrix (zero gradient for non-selected actions)
        q_pred  = self.q_net.forward(states)
        targets = q_pred.copy()
        for i in range(self.batch_size):
            targets[i, actions[i]] = td_targets[i]

        # Back-propagation
        grads, loss = compute_gradients(self.q_net, states, targets)

        # Adam update with gradient clipping
        self.train_steps += 1
        t = self.train_steps
        for layer, g, c in zip(self.q_net.layers, grads, self._adam_cache):
            if layer["type"] != "linear" or "dW" not in g:
                continue

            dW = np.clip(g["dW"], -self.grad_clip, self.grad_clip)
            db = np.clip(g["db"], -self.grad_clip, self.grad_clip)

            c["m_W"] = self.beta1 * c["m_W"] + (1 - self.beta1) * dW
            c["m_b"] = self.beta1 * c["m_b"] + (1 - self.beta1) * db
            c["v_W"] = self.beta2 * c["v_W"] + (1 - self.beta2) * dW ** 2
            c["v_b"] = self.beta2 * c["v_b"] + (1 - self.beta2) * db ** 2

            mW_hat = c["m_W"] / (1 - self.beta1 ** t)
            mb_hat = c["m_b"] / (1 - self.beta1 ** t)
            vW_hat = c["v_W"] / (1 - self.beta2 ** t)
            vb_hat = c["v_b"] / (1 - self.beta2 ** t)

            layer["W"] -= self.lr * mW_hat / (np.sqrt(vW_hat) + self.adam_eps)
            layer["b"] -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.adam_eps)

        # Periodic target sync & epsilon decay
        if self.train_steps % self.target_sync_every == 0:
            self._sync_target()

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return loss

    # ── persistence ──────────────────────────────────────────────────

    def save(self, path: str):
        data = {
            "layers": [],
            "epsilon": self.epsilon,
            "train_steps": self.train_steps,
        }
        for layer in self.q_net.layers:
            if layer["type"] == "linear":
                data["layers"].append({
                    "W": layer["W"].tolist(),
                    "b": layer["b"].tolist(),
                })
        np.save(path, data, allow_pickle=True)

    def load(self, path: str):
        data = np.load(path, allow_pickle=True).item()
        idx = 0
        for layer in self.q_net.layers:
            if layer["type"] == "linear":
                layer["W"] = np.array(data["layers"][idx]["W"])
                layer["b"] = np.array(data["layers"][idx]["b"])
                idx += 1
        self._sync_target()
        self.epsilon     = data.get("epsilon", self.epsilon_end)
        self.train_steps = data.get("train_steps", 0)
        self._adam_cache = self._init_adam()
