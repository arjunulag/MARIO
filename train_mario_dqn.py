"""
Train CNN -> Transformer -> DQN on the first five Super Mario Bros levels.

Saves the best checkpoint per level under weights/mario_dqn/best_{world}-{stage}.npy

Usage:
    python train_mario_dqn.py
    python train_mario_dqn.py --episodes 120 --frame-skip 4
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from mario_dqn_checkpoint import build_fresh_agent, load_manifest, save_agent, save_manifest
from mario_levels import (
    MarioLevel,
    episode_score,
    iter_first_five_levels,
    make_level_env,
)
from mario_training_utils import shape_mario_reward
from preprocessFrames import build_initial_stack, preprocess_frame


DEFAULT_WEIGHTS_DIR = Path("weights/mario_dqn")
DEFAULT_MANIFEST = DEFAULT_WEIGHTS_DIR / "manifest.json"

# Faster training defaults (frame skip + lighter transformer + less logging)
FRAME_SKIP = 4
MAX_EPISODE_STEPS = 350
TRAIN_EVERY = 4
GRAD_UPDATES_PER_TRAIN = 1
LEARN_START = 800
LOG_EVERY = 1

PROGRESS_REWARD_SCALE = 0.05
IDLE_PENALTY = -0.01


def env_step(env, action: int, frame_skip: int):
    """Repeat the same action for frame_skip frames; return last frame + totals."""
    total_reward = 0.0
    done = False
    info: dict = {}
    frame = None

    for _ in range(frame_skip):
        result = env.step(action)
        if len(result) == 5:
            frame, reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
        else:
            frame, reward, done, info = result
            done = bool(done)

        total_reward += float(reward)
        if done:
            break

    return frame, total_reward, done, info


def run_episode(
    env,
    agent,
    *,
    frame_skip: int,
    max_steps: int,
    learn: bool,
    learn_start: int,
    train_every: int,
    grad_updates: int,
):
    stack, state = build_initial_stack(env)
    done = False
    step = 0
    total_reward = 0.0
    shaped_total = 0.0
    losses: list[float] = []
    previous_x_pos = None
    last_info: dict = {}

    while not done and step < max_steps:
        action = agent.select_action(state)
        frame, reward, done, info = env_step(env, action, frame_skip)
        last_info = info if isinstance(info, dict) else {}

        next_frame = preprocess_frame(frame)
        stack.append(next_frame)
        next_state = np.stack(list(stack), axis=0)

        x_pos = last_info.get("x_pos")
        shaped_reward, _progress, previous_x_pos = shape_mario_reward(
            reward,
            x_pos,
            previous_x_pos,
            done,
            progress_reward_scale=PROGRESS_REWARD_SCALE,
            idle_penalty=IDLE_PENALTY,
        )

        if learn:
            agent.store(state, action, shaped_reward, next_state, done)
        state = next_state
        total_reward += reward
        shaped_total += shaped_reward
        step += 1

        if learn and len(agent.buffer) >= learn_start and step % train_every == 0:
            loss = agent.train(updates=grad_updates)
            if loss is not None:
                losses.append(loss)

    return {
        "steps": step,
        "total_reward": total_reward,
        "shaped_total": shaped_total,
        "losses": losses,
        "info": last_info,
        "x_pos": last_info.get("x_pos"),
        "flag_get": bool(last_info.get("flag_get", False)),
    }


def score_episode_result(episode_result: dict) -> float:
    return episode_score(
        x_pos=episode_result.get("x_pos"),
        flag_get=episode_result.get("flag_get", False),
        shaped_reward=episode_result.get("shaped_total", 0.0),
    )


def evaluate_greedy_agent(
    agent,
    level: MarioLevel,
    *,
    frame_skip: int,
    max_steps: int,
    episodes: int,
) -> dict:
    """Run no-learning, epsilon=0 rollouts and return the best result."""
    eval_episodes = max(1, int(episodes))
    previous_epsilon = agent.epsilon
    results = []

    agent.epsilon = 0.0
    try:
        for eval_idx in range(eval_episodes):
            env = make_level_env(level)
            try:
                result = run_episode(
                    env,
                    agent,
                    frame_skip=frame_skip,
                    max_steps=max_steps,
                    learn=False,
                    learn_start=0,
                    train_every=1,
                    grad_updates=0,
                )
            finally:
                env.close()

            result["score"] = score_episode_result(result)
            result["eval_episode"] = eval_idx
            results.append(result)
    finally:
        agent.epsilon = previous_epsilon

    best_result = max(results, key=lambda item: item["score"])
    return {
        "episodes": eval_episodes,
        "scores": [float(result["score"]) for result in results],
        "best": best_result,
        "mean_score": float(np.mean([result["score"] for result in results])),
    }


def maybe_save_best(
    agent,
    level: MarioLevel,
    episode_result: dict,
    episode_idx: int,
    best_scores: dict[str, float],
    weights_dir: Path,
    manifest_path: Path,
    manifest: dict,
    *,
    frame_skip: int,
    max_steps: int,
    eval_episodes: int,
) -> bool:
    training_score = score_episode_result(episode_result)
    key = level.key
    prev = best_scores.get(key, -1.0)
    if training_score <= prev:
        return False

    if eval_episodes > 0:
        eval_summary = evaluate_greedy_agent(
            agent,
            level,
            frame_skip=frame_skip,
            max_steps=max_steps,
            episodes=eval_episodes,
        )
        best_eval = eval_summary["best"]
        score = float(best_eval["score"])
    else:
        eval_summary = {
            "episodes": 0,
            "scores": [],
            "mean_score": training_score,
            "best": episode_result,
        }
        best_eval = episode_result
        score = training_score

    if score <= prev:
        print(
            f"  [checkpoint] training candidate for {key} reached "
            f"score={training_score:.1f}, but greedy score={score:.1f} "
            f"did not beat previous={prev:.1f}",
            flush=True,
        )
        return False

    best_scores[key] = score
    out_path = weights_dir / level.best_weights_name()
    meta = {
        "level": key,
        "env_id": level.env_id,
        "episode": episode_idx,
        "score": score,
        "score_type": "greedy_eval" if eval_episodes > 0 else "training_episode",
        "training_score": training_score,
        "x_pos": best_eval.get("x_pos"),
        "flag_get": best_eval.get("flag_get", False),
        "steps": best_eval.get("steps"),
        "shaped_reward": best_eval.get("shaped_total"),
        "eval_episodes": eval_summary["episodes"],
        "eval_scores": eval_summary["scores"],
        "eval_mean_score": eval_summary["mean_score"],
    }
    agent.save(out_path, meta=meta)
    manifest.setdefault("levels", {})[key] = {
        "weights": str(out_path.as_posix()),
        "score": score,
        **meta,
    }
    save_manifest(manifest_path, manifest)
    print(
        f"  [checkpoint] new greedy best for {key}: score={score:.1f} "
        f"train={training_score:.1f} x={meta['x_pos']} "
        f"flag={meta['flag_get']} -> {out_path}",
        flush=True,
    )
    return True


def train(
    *,
    episodes: int,
    weights_dir: Path,
    frame_skip: int,
    max_steps: int,
    resume: str | None,
    eval_episodes: int,
) -> None:
    weights_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = weights_dir / "manifest.json"
    manifest = load_manifest(manifest_path)
    best_scores = {
        key: float(entry.get("score", -1.0))
        for key, entry in manifest.get("levels", {}).items()
    }

    levels = [MarioLevel(world=1, stage=1)]
    probe_env = make_level_env(levels[0])
    n_actions = probe_env.action_space.n
    probe_env.close()

    if resume:
        from mario_dqn_checkpoint import load_agent

        agent = load_agent(resume)
        print(f"Resumed agent from {resume}", flush=True)
    else:
        agent = build_fresh_agent(n_actions, fast_transformer=True)
        print(
            f"Training on level: {levels[0].key}",
            flush=True,
        )

    print(
        f"Settings: episodes={episodes} frame_skip={frame_skip} "
        f"max_steps={max_steps} batch={agent.batch_size} "
        f"learn_start={LEARN_START} train_every={TRAIN_EVERY} "
        f"eval_episodes={eval_episodes}",
        flush=True,
    )

    t0 = time.time()
    for episode in range(episodes):
        level = levels[episode % len(levels)]
        env = make_level_env(level)
        try:
            result = run_episode(
                env,
                agent,
                frame_skip=frame_skip,
                max_steps=max_steps,
                learn=True,
                learn_start=LEARN_START,
                train_every=TRAIN_EVERY,
                grad_updates=GRAD_UPDATES_PER_TRAIN,
            )
        finally:
            env.close()

        maybe_save_best(
            agent,
            level,
            result,
            episode,
            best_scores,
            weights_dir,
            manifest_path,
            manifest,
            frame_skip=frame_skip,
            max_steps=max_steps,
            eval_episodes=eval_episodes,
        )

        avg_loss = float(np.mean(result["losses"])) if result["losses"] else 0.0
        if episode % LOG_EVERY == 0 or episode == episodes - 1:
            elapsed = time.time() - t0
            print(
                f"Ep {episode:4d} | {level.key} | steps {result['steps']:3d} | "
                f"raw {result['total_reward']:7.1f} | shaped {result['shaped_total']:7.1f} | "
                f"x {result.get('x_pos', '?')} | flag {result.get('flag_get')} | "
                f"eps {agent.epsilon:.3f} | replay {len(agent.buffer)} | "
                f"loss {avg_loss:.4f} | {elapsed:.0f}s",
                flush=True,
            )

    save_agent(
        agent,
        weights_dir / "latest.npy",
        meta={"episodes": episodes, "levels": [l.key for l in levels]},
    )
    print(f"Done. Latest weights: {weights_dir / 'latest.npy'}", flush=True)
    print(f"Per-level best: {weights_dir}/best_*.npy", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Mario CNN-Transformer-DQN (first 5 levels)")
    parser.add_argument("--episodes", type=int, default=150, help="Total training episodes")
    parser.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHTS_DIR)
    parser.add_argument("--frame-skip", type=int, default=FRAME_SKIP)
    parser.add_argument("--max-steps", type=int, default=MAX_EPISODE_STEPS)
    parser.add_argument("--resume", type=str, default=None, help="Path to latest.npy to resume")
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=1,
        help="Greedy epsilon=0 rollouts to verify a candidate before saving best weights",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train(
        episodes=args.episodes,
        weights_dir=args.weights_dir,
        frame_skip=args.frame_skip,
        max_steps=args.max_steps,
        resume=args.resume,
        eval_episodes=args.eval_episodes,
    )


if __name__ == "__main__":
    main()
