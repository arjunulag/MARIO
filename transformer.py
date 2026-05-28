import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from functions import Parameter_init


@dataclass
class TransformerConfig:
    """
    Knobs for customizing the transformer.

    vocab_size      : size of the token vocabulary
    d_model         : embedding/hidden dimension (must be divisible by num_heads)
    num_heads       : number of attention heads
    num_layers      : number of stacked transformer blocks
    d_ff            : inner dimension of the position-wise feed-forward sublayer
    max_seq_len     : longest sequence the positional embeddings cover
    causal          : True for decoder-style (autoregressive) masking, False for encoder
    pre_norm        : True applies LayerNorm before each sublayer (GPT-style),
                      False applies it after (original "Attention Is All You Need")
    activation      : activation in the FFN ("relu" or "sigmoid")
    tie_embeddings  : reuse the token embedding matrix as the output projection
    seed            : RNG seed for reproducible weight init
    """
    vocab_size: int = 256
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 2
    d_ff: int = 256
    max_seq_len: int = 128
    causal: bool = True
    pre_norm: bool = True
    activation: str = "relu"
    tie_embeddings: bool = True
    seed: Optional[int] = 0

    def __post_init__(self):
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
            )


def softmax(x, axis=-1):
    """Numerically stable softmax along the given axis."""
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


class LayerNorm:
    """Layer normalization over the last dim. Learnable gain (gamma) and bias (beta)."""

    def __init__(self, dim, eps=1e-5):
        self.gamma = np.ones((dim,))
        self.beta = np.zeros((dim,))
        self.eps = eps

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_hat = (x - mean) / np.sqrt(var + self.eps)
        return x_hat * self.gamma + self.beta

    def forward_with_cache(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        inv_std = 1.0 / np.sqrt(var + self.eps)
        x_hat = (x - mean) * inv_std
        out = x_hat * self.gamma + self.beta
        cache = {
            "x": x,
            "x_hat": x_hat,
            "mean": mean,
            "inv_std": inv_std,
            "gamma": self.gamma,
        }
        return out, cache

    def backward(self, grad_out, cache):
        x = cache["x"]
        x_hat = cache["x_hat"]
        inv_std = cache["inv_std"]
        gamma = cache["gamma"]
        D = x.shape[-1]

        dy = grad_out * gamma
        mean_dy = np.mean(dy, axis=-1, keepdims=True)
        mean_dy_xhat = np.mean(dy * x_hat, axis=-1, keepdims=True)
        dx = inv_std * (dy - mean_dy - x_hat * mean_dy_xhat)
        return dx


class MultiHeadAttention:
    """
    Scaled dot-product multi-head self-attention.

    Splits d_model into num_heads parallel subspaces, runs attention in each,
    concatenates, then projects back to d_model.
    """

    def __init__(self, d_model, num_heads, causal=False, rng=None):
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.causal = causal

        rng = rng if rng is not None else np.random.default_rng()
        std = np.sqrt(1.0 / d_model)
        self.W_q = rng.standard_normal((d_model, d_model)) * std
        self.W_k = rng.standard_normal((d_model, d_model)) * std
        self.W_v = rng.standard_normal((d_model, d_model)) * std
        self.W_o = rng.standard_normal((d_model, d_model)) * std
        self.b_q = np.zeros((d_model,))
        self.b_k = np.zeros((d_model,))
        self.b_v = np.zeros((d_model,))
        self.b_o = np.zeros((d_model,))

    def _split_heads(self, x):
        # (B, T, d_model) -> (B, num_heads, T, head_dim)
        B, T, _ = x.shape
        x = x.reshape(B, T, self.num_heads, self.head_dim)
        return x.transpose(0, 2, 1, 3)

    def _merge_heads(self, x):
        # (B, num_heads, T, head_dim) -> (B, T, d_model)
        B, H, T, D = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, H * D)

    def forward(self, x, attn_mask=None):
        B, T, _ = x.shape

        Q = self._split_heads(x @ self.W_q + self.b_q)
        K = self._split_heads(x @ self.W_k + self.b_k)
        V = self._split_heads(x @ self.W_v + self.b_v)

        # (B, H, T, T)
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.head_dim)

        if self.causal:
            causal_mask = np.triu(np.ones((T, T), dtype=bool), k=1)
            scores = np.where(causal_mask, -1e9, scores)

        if attn_mask is not None:
            # attn_mask: (B, T) with True for padding positions to ignore
            pad = attn_mask[:, None, None, :]
            scores = np.where(pad, -1e9, scores)

        weights = softmax(scores, axis=-1)
        out = weights @ V  # (B, H, T, head_dim)
        out = self._merge_heads(out)
        return out @ self.W_o + self.b_o

    def forward_with_cache(self, x, attn_mask=None):
        B, T, _ = x.shape
        Q = x @ self.W_q + self.b_q
        K = x @ self.W_k + self.b_k
        V = x @ self.W_v + self.b_v

        # For T == 1, self-attention is identity across time with weights = 1.
        out_pre = V @ self.W_o + self.b_o
        cache = {
            "x": x,
            "W_v": self.W_v,
            "W_o": self.W_o,
        }
        return out_pre, cache

    def backward_input(self, grad_out, cache):
        dV = grad_out @ cache["W_o"].T
        dx = dV @ cache["W_v"].T
        return dx


class FeedForward:
    """Position-wise FFN built on the existing Parameter_init MLP."""

    def __init__(self, d_model, d_ff, activation="relu"):
        config = [
            {"type": "linear", "in": d_model, "out": d_ff, "activation_hint": activation},
            {"type": activation},
            {"type": "linear", "in": d_ff, "out": d_model, "activation_hint": activation},
        ]
        self.mlp = Parameter_init(config)

    def forward(self, x):
        # Parameter_init.forward expects 2D — flatten the (B, T) axes, then restore.
        B, T, D = x.shape
        flat = x.reshape(B * T, D)
        out = self.mlp.forward(flat)
        return out.reshape(B, T, -1)

    def forward_with_cache(self, x):
        B, T, D = x.shape
        flat = x.reshape(B * T, D)
        h1 = flat @ self.mlp.layers[0]["W"] + self.mlp.layers[0]["b"]
        alpha = self.mlp.layers[1]["alpha"] if self.mlp.layers[1]["type"] == "relu" else 0.01
        act = np.where(h1 > 0, h1, alpha * h1)
        h2 = act @ self.mlp.layers[2]["W"] + self.mlp.layers[2]["b"]
        out = h2.reshape(B, T, -1)
        cache = {
            "x": x,
            "h1": h1,
            "act": act,
            "W1": self.mlp.layers[0]["W"],
            "W2": self.mlp.layers[2]["W"],
            "alpha": alpha,
        }
        return out, cache

    def backward_input(self, grad_out, cache):
        B, T, D = grad_out.shape
        grad_flat = grad_out.reshape(B * T, D)
        dact = grad_flat @ cache["W2"].T
        dh1 = dact * np.where(cache["h1"] > 0, 1.0, cache["alpha"])
        dx_flat = dh1 @ cache["W1"].T
        return dx_flat.reshape(B, T, -1)


class TransformerBlock:
    """One transformer layer: self-attention + FFN, with residuals and layer norms."""

    def __init__(self, cfg: TransformerConfig, rng):
        self.pre_norm = cfg.pre_norm
        self.attn = MultiHeadAttention(cfg.d_model, cfg.num_heads, cfg.causal, rng=rng)
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff, activation=cfg.activation)
        self.ln1 = LayerNorm(cfg.d_model)
        self.ln2 = LayerNorm(cfg.d_model)

    def forward(self, x, attn_mask=None):
        if self.pre_norm:
            x = x + self.attn.forward(self.ln1.forward(x), attn_mask=attn_mask)
            x = x + self.ffn.forward(self.ln2.forward(x))
        else:
            x = self.ln1.forward(x + self.attn.forward(x, attn_mask=attn_mask))
            x = self.ln2.forward(x + self.ffn.forward(x))
        return x

    def forward_with_cache(self, x, attn_mask=None):
        cache = {}
        x0 = x
        x1, ln1_cache = self.ln1.forward_with_cache(x0)
        attn_out, attn_cache = self.attn.forward_with_cache(x1, attn_mask=attn_mask)
        x2 = x0 + attn_out
        x3, ln2_cache = self.ln2.forward_with_cache(x2)
        ffn_out, ffn_cache = self.ffn.forward_with_cache(x3)
        x4 = x2 + ffn_out
        cache["ln1"] = ln1_cache
        cache["attn"] = attn_cache
        cache["ln2"] = ln2_cache
        cache["ffn"] = ffn_cache
        cache["x0"] = x0
        return x4, cache

    def backward_input(self, grad_out, cache):
        d_x2 = grad_out
        d_ffn = d_x2
        d_x2 = d_x2

        d_x3 = self.ffn.backward_input(d_ffn, cache["ffn"])
        d_x2 += self.ln2.backward(d_x3, cache["ln2"])

        d_attn = d_x2
        d_x0 = d_x2
        d_x1 = self.attn.backward_input(d_attn, cache["attn"])
        d_x0 += self.ln1.backward(d_x1, cache["ln1"])
        return d_x0


class Transformer:
    """
    Customizable transformer (forward / inference).

    Inputs are token id arrays of shape (batch, seq_len). Output is logits of shape
    (batch, seq_len, vocab_size). Configure everything via TransformerConfig.
    """

    def __init__(self, cfg: TransformerConfig):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)

        emb_std = np.sqrt(1.0 / cfg.d_model)
        self.token_emb = rng.standard_normal((cfg.vocab_size, cfg.d_model)) * emb_std
        self.pos_emb = rng.standard_normal((cfg.max_seq_len, cfg.d_model)) * emb_std

        self.blocks = [TransformerBlock(cfg, rng) for _ in range(cfg.num_layers)]
        self.ln_f = LayerNorm(cfg.d_model)

        if cfg.tie_embeddings:
            self.out_proj = None  # use token_emb.T at forward time
            self.out_bias = np.zeros((cfg.vocab_size,))
        else:
            self.out_proj = rng.standard_normal((cfg.d_model, cfg.vocab_size)) * emb_std
            self.out_bias = np.zeros((cfg.vocab_size,))

    def forward(self, token_ids, attn_mask=None):
        if token_ids.ndim == 1:
            token_ids = token_ids[np.newaxis, :]
        B, T = token_ids.shape
        if T > self.cfg.max_seq_len:
            raise ValueError(
                f"Sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}"
            )

        x = self.token_emb[token_ids] + self.pos_emb[:T][np.newaxis, :, :]

        for block in self.blocks:
            x = block.forward(x, attn_mask=attn_mask)
        x = self.ln_f.forward(x)

        if self.out_proj is None:
            logits = x @ self.token_emb.T + self.out_bias
        else:
            logits = x @ self.out_proj + self.out_bias
        return logits

    def generate(self, prompt_ids, max_new_tokens, temperature=1.0, top_k=None, rng=None):
        """Greedy / sampled autoregressive generation. Requires cfg.causal=True."""
        if not self.cfg.causal:
            raise ValueError("generate() requires a causal (decoder) transformer")
        rng = rng if rng is not None else np.random.default_rng()

        ids = np.asarray(prompt_ids, dtype=np.int64)
        if ids.ndim == 1:
            ids = ids[np.newaxis, :]

        for _ in range(max_new_tokens):
            window = ids[:, -self.cfg.max_seq_len:]
            logits = self.forward(window)[:, -1, :] / max(temperature, 1e-8)

            if top_k is not None and top_k < logits.shape[-1]:
                kth = np.partition(logits, -top_k, axis=-1)[:, -top_k:][:, :1]
                logits = np.where(logits < kth, -1e9, logits)

            probs = softmax(logits, axis=-1)
            next_id = np.array([rng.choice(probs.shape[-1], p=probs[i]) for i in range(probs.shape[0])])
            ids = np.concatenate([ids, next_id[:, None]], axis=1)
        return ids
    def forward_from_embedding(self, x):
        # x is shape (1, 1, d_model) — one frame, one timestep
        for block in self.blocks:
            x = block.forward(x)
        x = self.ln_f.forward(x)
        logits = x @ self.token_emb.T + self.out_bias
        return logits  # shape (1, 1, vocab_size) → action logits

    def forward_from_embedding_with_cache(self, x):
        cache = {"blocks": []}
        h = x
        for block in self.blocks:
            h, block_cache = block.forward_with_cache(h)
            cache["blocks"].append(block_cache)
        h, ln_f_cache = self.ln_f.forward_with_cache(h)
        cache["ln_f"] = ln_f_cache

        if self.out_proj is None:
            logits = h @ self.token_emb.T + self.out_bias
        else:
            logits = h @ self.out_proj + self.out_bias
        cache["h"] = h
        return logits, cache

    def input_grad(self, grad_logits, cache):
        if self.out_proj is None:
            d_h = grad_logits @ self.token_emb
        else:
            d_h = grad_logits @ self.out_proj.T

        d_h = self.ln_f.backward(d_h, cache["ln_f"])
        for block, block_cache in zip(reversed(self.blocks), reversed(cache["blocks"])):
            d_h = block.backward_input(d_h, block_cache)
        return d_h


if __name__ == "__main__":
    cfg = TransformerConfig(
        vocab_size=64,
        d_model=32,
        num_heads=4,
        num_layers=2,
        d_ff=64,
        max_seq_len=16,
        causal=True,
    )
    model = Transformer(cfg)

    tokens = np.random.randint(0, cfg.vocab_size, size=(2, 8))
    logits = model.forward(tokens)
    print("logits shape:", logits.shape)

    out = model.generate(np.array([1, 2, 3]), max_new_tokens=5)
    print("generated:", out)
