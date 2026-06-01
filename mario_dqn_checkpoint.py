"""Save / load CNN–Transformer–DQN checkpoints (NumPy, no PyTorch)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from CNN_network import build_weights
from Tensor import Tensor
from dqn_agent import MarioCNNTransformerDQNAgent
from transformer import Transformer, TransformerConfig


def _export_attention(attn) -> dict[str, np.ndarray]:
    names = ["W_q", "W_k", "W_v", "W_o", "b_q", "b_k", "b_v", "b_o"]
    return {name: np.array(getattr(attn, name), copy=True) for name in names}


def _import_attention(attn, state: dict[str, np.ndarray]) -> None:
    for name, value in state.items():
        getattr(attn, name)[...] = value


def _export_ffn(ffn) -> list[dict[str, np.ndarray]]:
    layers = []
    for layer in ffn.mlp.layers:
        entry: dict[str, np.ndarray] = {}
        if "W" in layer:
            entry["W"] = np.array(layer["W"], copy=True)
        if "b" in layer:
            entry["b"] = np.array(layer["b"], copy=True)
        layers.append(entry)
    return layers


def _import_ffn(ffn, layers: list[dict[str, np.ndarray]]) -> None:
    for layer, saved in zip(ffn.mlp.layers, layers):
        if "W" in saved:
            layer["W"][...] = saved["W"]
        if "b" in saved:
            layer["b"][...] = saved["b"]


def export_transformer(transformer: Transformer) -> dict[str, Any]:
    blocks = []
    for block in transformer.blocks:
        blocks.append(
            {
                "attn": _export_attention(block.attn),
                "ffn": _export_ffn(block.ffn),
                "ln1_gamma": np.array(block.ln1.gamma, copy=True),
                "ln1_beta": np.array(block.ln1.beta, copy=True),
                "ln2_gamma": np.array(block.ln2.gamma, copy=True),
                "ln2_beta": np.array(block.ln2.beta, copy=True),
            }
        )

    payload: dict[str, Any] = {
        "cfg": asdict(transformer.cfg),
        "token_emb": np.array(transformer.token_emb, copy=True),
        "pos_emb": np.array(transformer.pos_emb, copy=True),
        "out_bias": np.array(transformer.out_bias, copy=True),
        "out_proj": None
        if transformer.out_proj is None
        else np.array(transformer.out_proj, copy=True),
        "blocks": blocks,
        "ln_f_gamma": np.array(transformer.ln_f.gamma, copy=True),
        "ln_f_beta": np.array(transformer.ln_f.beta, copy=True),
    }
    return payload


def import_transformer(payload: dict[str, Any]) -> Transformer:
    cfg = TransformerConfig(**payload["cfg"])
    transformer = Transformer(cfg)
    transformer.token_emb[...] = payload["token_emb"]
    transformer.pos_emb[...] = payload["pos_emb"]
    transformer.out_bias[...] = payload["out_bias"]
    if payload["out_proj"] is not None:
        transformer.out_proj[...] = payload["out_proj"]

    for block, saved in zip(transformer.blocks, payload["blocks"]):
        _import_attention(block.attn, saved["attn"])
        _import_ffn(block.ffn, saved["ffn"])
        block.ln1.gamma[...] = saved["ln1_gamma"]
        block.ln1.beta[...] = saved["ln1_beta"]
        block.ln2.gamma[...] = saved["ln2_gamma"]
        block.ln2.beta[...] = saved["ln2_beta"]

    transformer.ln_f.gamma[...] = payload["ln_f_gamma"]
    transformer.ln_f.beta[...] = payload["ln_f_beta"]
    return transformer


def export_agent(agent: MarioCNNTransformerDQNAgent) -> dict[str, Any]:
    return {
        "version": 1,
        "kernels": [np.array(k.data, copy=True) for k in agent.kernels],
        "W1": np.array(agent.W1.data, copy=True),
        "b1": np.array(agent.b1.data, copy=True),
        "W2": np.array(agent.W2.data, copy=True),
        "b2": np.array(agent.b2.data, copy=True),
        "transformer": export_transformer(agent.transformer),
        "epsilon": agent.epsilon,
        "train_steps": agent.train_steps,
        "action_dim": agent.action_dim,
        "hyperparams": {
            "lr": agent.lr,
            "gamma": agent.gamma,
            "batch_size": agent.batch_size,
            "target_sync_every": agent.target_sync_every,
            "grad_clip": agent.grad_clip,
            "reward_clip": agent.reward_clip,
        },
    }


def save_agent(
    agent: MarioCNNTransformerDQNAgent,
    path: str | Path,
    *,
    meta: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = export_agent(agent)
    payload["meta"] = meta or {}
    np.save(path, payload, allow_pickle=True)


def load_agent(
    path: str | Path,
    *,
    epsilon: float | None = None,
) -> MarioCNNTransformerDQNAgent:
    path = Path(path)
    payload = np.load(path, allow_pickle=True).item()

    kernels = [Tensor(np.array(k, copy=True)) for k in payload["kernels"]]
    W1 = Tensor(np.array(payload["W1"], copy=True))
    b1 = Tensor(np.array(payload["b1"], copy=True))
    W2 = Tensor(np.array(payload["W2"], copy=True))
    b2 = Tensor(np.array(payload["b2"], copy=True))
    transformer = import_transformer(payload["transformer"])

    hp = payload.get("hyperparams", {})
    agent = MarioCNNTransformerDQNAgent(
        kernels,
        W1,
        b1,
        W2,
        b2,
        transformer,
        action_dim=int(payload["action_dim"]),
        lr=float(hp.get("lr", 0.0001)),
        gamma=float(hp.get("gamma", 0.99)),
        batch_size=int(hp.get("batch_size", 16)),
        target_sync_every=int(hp.get("target_sync_every", 1000)),
        grad_clip=float(hp.get("grad_clip", 1.0)),
        reward_clip=float(hp.get("reward_clip", 5.0)),
        epsilon_start=float(payload.get("epsilon", 0.05)),
        epsilon_end=float(payload.get("epsilon", 0.05)),
        epsilon_decay=1.0,
    )
    agent.train_steps = int(payload.get("train_steps", 0))
    if epsilon is not None:
        agent.epsilon = epsilon
    agent._sync_target()
    agent.meta = payload.get("meta", {})
    return agent


def build_fresh_agent(
    action_dim: int,
    *,
    d_model: int = 64,
    fast_transformer: bool = True,
    **agent_kwargs,
) -> MarioCNNTransformerDQNAgent:
    kernels, W1, b1, W2, b2 = build_weights(d_model=d_model)
    if fast_transformer:
        cfg = TransformerConfig(
            vocab_size=action_dim,
            d_model=d_model,
            num_heads=2,
            num_layers=1,
            d_ff=128,
            max_seq_len=8,
            causal=True,
            tie_embeddings=True,
        )
    else:
        cfg = TransformerConfig(vocab_size=action_dim, d_model=d_model)

    transformer = Transformer(cfg)
    defaults = {
        "lr": 0.0001,
        "gamma": 0.99,
        "buffer_size": 30_000,
        "batch_size": 16,
        "target_sync_every": 800,
    }
    defaults.update(agent_kwargs)
    return MarioCNNTransformerDQNAgent(
        kernels,
        W1,
        b1,
        W2,
        b2,
        transformer,
        action_dim=action_dim,
        **defaults,
    )


def save_manifest(path: str | Path, records: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def load_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        return {"levels": {}}
    return json.loads(path.read_text(encoding="utf-8"))
