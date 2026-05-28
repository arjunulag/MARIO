from collections import deque
import random

import numpy as np

from CNN_network import forward
from Tensor import Tensor


class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(
            (
                np.array(state, dtype=np.float32),
                int(action),
                float(reward),
                np.array(next_state, dtype=np.float32),
                bool(done),
            )
        )

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


def clone_cnn_weights(kernels, W1, b1, W2, b2):
    target_kernels = [Tensor(k.data.copy()) for k in kernels]
    return (
        target_kernels,
        Tensor(W1.data.copy()),
        Tensor(b1.data.copy()),
        Tensor(W2.data.copy()),
        Tensor(b2.data.copy()),
    )


def sync_cnn_weights(source, target):
    src_kernels, src_W1, src_b1, src_W2, src_b2 = source
    tgt_kernels, tgt_W1, tgt_b1, tgt_W2, tgt_b2 = target

    for src, tgt in zip(src_kernels, tgt_kernels):
        tgt.data[...] = src.data
    tgt_W1.data[...] = src_W1.data
    tgt_b1.data[...] = src_b1.data
    tgt_W2.data[...] = src_W2.data
    tgt_b2.data[...] = src_b2.data


def sync_transformer(source, target):
    target.token_emb[...] = source.token_emb
    target.pos_emb[...] = source.pos_emb
    target.out_bias[...] = source.out_bias
    if source.out_proj is not None:
        target.out_proj[...] = source.out_proj

    for src_block, tgt_block in zip(source.blocks, target.blocks):
        for name in ["W_q", "W_k", "W_v", "W_o", "b_q", "b_k", "b_v", "b_o"]:
            getattr(tgt_block.attn, name)[...] = getattr(src_block.attn, name)
        for src_layer, tgt_layer in zip(src_block.ffn.mlp.layers, tgt_block.ffn.mlp.layers):
            if "W" in src_layer:
                tgt_layer["W"][...] = src_layer["W"]
            if "b" in src_layer:
                tgt_layer["b"][...] = src_layer["b"]
        tgt_block.ln1.gamma[...] = src_block.ln1.gamma
        tgt_block.ln1.beta[...] = src_block.ln1.beta
        tgt_block.ln2.gamma[...] = src_block.ln2.gamma
        tgt_block.ln2.beta[...] = src_block.ln2.beta

    target.ln_f.gamma[...] = source.ln_f.gamma
    target.ln_f.beta[...] = source.ln_f.beta


def q_values(state, kernels, W1, b1, W2, b2, transformer):
    x = forward(state, kernels, W1, b1, W2, b2)
    logits = transformer.forward_from_embedding(x.data.reshape(1, 1, -1))
    return logits[0, 0]


def select_dqn_action(state, kernels, W1, b1, W2, b2, transformer, epsilon):
    if np.random.random() < epsilon:
        return int(np.random.randint(transformer.cfg.vocab_size))

    q = q_values(state, kernels, W1, b1, W2, b2, transformer)
    return int(np.argmax(q))


def _zero_cnn_grads(kernels, weights):
    for tensor in [*kernels, *weights]:
        tensor.grad = np.zeros_like(tensor.grad)


def _update_from_q_gradient(state, action, grad_q, kernels, W1, b1, W2, b2, transformer, lr):
    x = forward(state, kernels, W1, b1, W2, b2)
    features = x.data.reshape(1, 1, -1)
    _, cache = transformer.forward_from_embedding_with_cache(features)

    grad_logits = np.zeros((1, 1, transformer.cfg.vocab_size), dtype=np.float32)
    grad_logits[0, 0, action] = grad_q

    h = cache["h"]
    grad_features = transformer.input_grad(grad_logits, cache).reshape(-1)

    if transformer.out_proj is None:
        transformer.token_emb -= lr * np.einsum("btd,btv->vd", h, grad_logits)
    else:
        transformer.out_proj -= lr * np.einsum("btd,btv->dv", h, grad_logits)
    transformer.out_bias -= lr * grad_logits.sum(axis=(0, 1))

    x.backward(grad_features)
    for tensor in kernels:
        tensor.data -= lr * tensor.grad
    for tensor in [W1, b1, W2, b2]:
        tensor.data -= lr * tensor.grad
    _zero_cnn_grads(kernels, [W1, b1, W2, b2])


def train_dqn_batch(
    replay,
    kernels,
    W1,
    b1,
    W2,
    b2,
    transformer,
    target_kernels,
    target_W1,
    target_b1,
    target_W2,
    target_b2,
    target_transformer,
    batch_size=8,
    gamma=0.99,
    lr=0.0001,
    reward_clip=5.0,
):
    if len(replay) < batch_size:
        return None

    states, actions, rewards, next_states, dones = replay.sample(batch_size)
    losses = []

    for state, action, reward, next_state, done in zip(
        states, actions, rewards, next_states, dones
    ):
        clipped_reward = float(np.clip(reward, -reward_clip, reward_clip))
        current_q = q_values(state, kernels, W1, b1, W2, b2, transformer)[action]
        next_q = q_values(
            next_state,
            target_kernels,
            target_W1,
            target_b1,
            target_W2,
            target_b2,
            target_transformer,
        )
        target = clipped_reward
        if not done:
            target += gamma * float(np.max(next_q))

        error = current_q - target
        losses.append(0.5 * float(error * error))
        _update_from_q_gradient(
            state,
            action,
            error / batch_size,
            kernels,
            W1,
            b1,
            W2,
            b2,
            transformer,
            lr,
        )

    return float(np.mean(losses))


def train_step(state, kernels, W1, b1, W2, b2, transformer, epsilon=0.05):
    """Compatibility helper for quick action selection."""
    action = select_dqn_action(state, kernels, W1, b1, W2, b2, transformer, epsilon)
    features = forward(state, kernels, W1, b1, W2, b2)
    return action, features
