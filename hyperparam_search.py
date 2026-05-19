"""
Adaptive two-stage random search for the 4D CartPole DQN agent.

Stage 1 explores a broad space with short trials, then stage 2 zooms around
the top-K configs with longer trials. Scoring metric is the mean reward over
the last `window` episodes (higher is better; stored as negative so the
existing min-sort convention keeps working).

Usage:
    python hyperparam_search.py                          # full default search
    python hyperparam_search.py --n1 10 --n2 5 \
        --budget1 150 --budget2 600                      # quick sweep
    python hyperparam_search.py --seed 7 --top-k 3
"""

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np

from cartpole4d_env import CartPole4DEnv
from dqn_agent import DQNAgent


# ── sampling primitives ──────────────────────────────────────────────────

def log_uniform(low, high):
    lo, hi = math.log(low), math.log(high)
    return math.exp(random.uniform(lo, hi))


def uniform(low, high):
    return random.uniform(low, high)


def random_config(space: dict) -> dict:
    cfg = {}
    for k, v in space.items():
        cfg[k] = v() if callable(v) else random.choice(v)
    return cfg


# ── DQN-specific search space ────────────────────────────────────────────
# Centered on the known-stable config in train_cartpole4d.py
# (lr=1e-4, buffer=100k, target_sync=1000) but explores around it.

SPACE_STAGE1 = {
    "lr":                lambda: log_uniform(3e-5, 1e-3),
    "gamma":             lambda: uniform(0.97, 0.997),
    "epsilon_decay":     lambda: uniform(0.9985, 0.9999),
    "grad_clip":         lambda: log_uniform(0.5, 10.0),
    "hidden":            [[64, 64], [128, 128], [256, 256],
                          [128, 128, 128], [256, 128]],
    "batch_size":        [32, 64, 128],
    "buffer_size":       [50_000, 100_000, 200_000],
    "target_sync_every": [500, 1000, 2000],
}

# Param categories: continuous-log, continuous-linear, discrete.
# Used by zoom_space to decide how to refine around top configs.
CONTINUOUS_LOG = {"lr", "grad_clip"}
CONTINUOUS_LIN = {"gamma", "epsilon_decay"}
DISCRETE       = {"hidden", "batch_size", "buffer_size", "target_sync_every"}


def zoom_space(best_cfgs, log_factor=3.0, lin_window=0.2):
    """Build a stage-2 sampler that draws near the top stage-1 configs."""

    def make_log_sampler(values):
        def sampler():
            c = random.choice(values)
            return log_uniform(c / log_factor, c * log_factor)
        return sampler

    def make_lin_sampler(values):
        def sampler():
            c = random.choice(values)
            # Clip to a sensible band for probabilities/decays.
            lo = max(0.0, c - lin_window * abs(c))
            hi = min(1.0, c + lin_window * abs(c))
            return uniform(lo, hi)
        return sampler

    space = {}
    for key in CONTINUOUS_LOG:
        space[key] = make_log_sampler([c[key] for c in best_cfgs])
    for key in CONTINUOUS_LIN:
        space[key] = make_lin_sampler([c[key] for c in best_cfgs])
    for key in DISCRETE:
        # dedupe lists by JSON serialization since lists aren't hashable
        seen = {json.dumps(c[key]): c[key] for c in best_cfgs}
        space[key] = list(seen.values())
    return space


# ── DQN trial runner ─────────────────────────────────────────────────────

def train_eval_dqn(cfg: dict, episodes: int, window: int = 100,
                   max_env_steps: int = 500, log_path: str | None = None,
                   verbose: bool = False) -> float:
    """
    Train a fresh DQN with `cfg` for `episodes` episodes.
    Returns negative mean reward over the last `window` episodes
    (negative so lower = better, matching the existing sort).
    """
    env = CartPole4DEnv(use_discrete=True)
    agent = DQNAgent(
        state_dim=12,
        action_dim=6,
        hidden=cfg["hidden"],
        lr=cfg["lr"],
        gamma=cfg["gamma"],
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=cfg["epsilon_decay"],
        buffer_size=cfg["buffer_size"],
        batch_size=cfg["batch_size"],
        target_sync_every=cfg["target_sync_every"],
        grad_clip=cfg["grad_clip"],
    )

    reward_history: list[float] = []
    log_f = log_w = None
    if log_path is not None:
        log_f = open(log_path, "w", newline="", buffering=1)
        log_w = csv.writer(log_f)
        log_w.writerow(["episode", "reward", "avg100", "steps",
                        "epsilon", "loss", "timestamp"])

    try:
        for ep in range(1, episodes + 1):
            obs, _ = env.reset()
            ep_reward = 0.0
            losses: list[float] = []
            done = False
            steps = 0

            while not done and steps < max_env_steps:
                action = agent.select_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                agent.store(obs, action, reward, next_obs, float(done))
                loss = agent.train()
                if loss is not None:
                    losses.append(loss)

                obs = next_obs
                ep_reward += reward
                steps += 1

            reward_history.append(ep_reward)
            avg = float(np.mean(reward_history[-window:]))
            avg_loss = float(np.mean(losses)) if losses else 0.0

            if log_w is not None:
                log_w.writerow([ep, ep_reward, avg, info["step"],
                                agent.epsilon, avg_loss, time.time()])
            if verbose and ep % 50 == 0:
                print(f"    ep {ep:4d}/{episodes}  avg{window} {avg:7.1f}  "
                      f"ε {agent.epsilon:.3f}")

            # Early stop: if we've clearly diverged (NaN) or are flat near zero
            # after enough warmup, abandon — saves search budget.
            if not math.isfinite(ep_reward):
                if verbose:
                    print("    aborting trial: non-finite reward")
                break
    finally:
        if log_f is not None:
            log_f.close()

    final_avg = float(np.mean(reward_history[-window:])) if reward_history else 0.0
    return -final_avg


# ── two-stage adaptive search ────────────────────────────────────────────

def adaptive_two_stage(space1, train_eval, n1, n2, budget1, budget2,
                       top_k, seed, log_dir: Path):
    random.seed(seed)
    np.random.seed(seed)

    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = log_dir / "summary.csv"
    summary_f = open(summary_path, "w", newline="", buffering=1)
    summary_w = csv.writer(summary_f)
    summary_w.writerow(["stage", "trial", "score", "avg100_reward",
                        "episodes", "config"])

    def run_stage(stage_name, space, n_trials, budget):
        results = []
        for i in range(n_trials):
            cfg = random_config(space)
            trial_id = f"{stage_name}_{i:03d}"
            trial_log = log_dir / f"{trial_id}.csv"
            t0 = time.time()
            score = train_eval(cfg, budget, log_path=str(trial_log))
            dt = time.time() - t0
            results.append((score, cfg))
            summary_w.writerow([stage_name, i, score, -score, budget,
                                json.dumps(cfg)])
            print(f"[{stage_name} {i+1}/{n_trials}] avg100={-score:7.1f}  "
                  f"({dt:5.1f}s)  cfg={cfg}")
        results.sort(key=lambda t: t[0])
        return results

    print(f"\n── Stage 1: broad search ({n1} trials × {budget1} eps) ──")
    results1 = run_stage("s1", space1, n1, budget1)
    top_cfgs = [cfg for _, cfg in results1[:top_k]]

    print(f"\n── Stage 2: zoom on top {top_k} ({n2} trials × {budget2} eps) ──")
    space2 = zoom_space(top_cfgs)
    results2 = run_stage("s2", space2, n2, budget2)

    summary_f.close()

    best_score, best_cfg = results2[0] if results2 else results1[0]
    return best_cfg, best_score, results1, results2


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Adaptive HP search for 4D CartPole DQN")
    p.add_argument("--n1",       type=int,   default=20,
                   help="Stage-1 trial count (broad search)")
    p.add_argument("--n2",       type=int,   default=8,
                   help="Stage-2 trial count (zoomed search)")
    p.add_argument("--budget1",  type=int,   default=200,
                   help="Episodes per stage-1 trial")
    p.add_argument("--budget2",  type=int,   default=800,
                   help="Episodes per stage-2 trial")
    p.add_argument("--top-k",    type=int,   default=5,
                   help="Stage-1 configs carried forward to stage-2 zoom")
    p.add_argument("--seed",     type=int,   default=0)
    p.add_argument("--log-dir",  type=str,   default="hp_search_logs")
    args = p.parse_args()

    log_dir = Path(args.log_dir)
    best_cfg, best_score, _r1, _r2 = adaptive_two_stage(
        space1=SPACE_STAGE1,
        train_eval=train_eval_dqn,
        n1=args.n1,
        n2=args.n2,
        budget1=args.budget1,
        budget2=args.budget2,
        top_k=args.top_k,
        seed=args.seed,
        log_dir=log_dir,
    )

    print("\n══ best config ══")
    print(json.dumps(best_cfg, indent=2))
    print(f"avg100 reward: {-best_score:.1f}")
    with open(log_dir / "best_config.json", "w") as f:
        json.dump({"config": best_cfg, "avg100_reward": -best_score}, f, indent=2)
    print(f"Saved to {log_dir / 'best_config.json'}")


if __name__ == "__main__":
    main()
