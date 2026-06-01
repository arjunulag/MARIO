"""
Watch the CNN -> Transformer -> DQN agent play each of the first five Mario levels
using that level's best saved weights.

Usage:
    python watch_mario_levels.py
    python watch_mario_levels.py --level 1-1
    python watch_mario_levels.py --all --episodes 3 --delay 0.02
    python watch_mario_levels.py --weights-dir weights/mario_dqn --no-render
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from mario_dqn_checkpoint import load_agent, load_manifest
from mario_levels import (
    FIRST_FIVE_LEVELS,
    MarioLevel,
    iter_first_five_levels,
    level_from_key,
    make_level_env,
)
from mario_training_utils import shape_mario_reward
from preprocessFrames import build_initial_stack, preprocess_frame
from train_mario_dqn import env_step


DEFAULT_WEIGHTS_DIR = Path("weights/mario_dqn")
DEFAULT_MANIFEST = DEFAULT_WEIGHTS_DIR / "manifest.json"


def resolve_weights_path(level: MarioLevel, weights_dir: Path, manifest: dict) -> Path | None:
    entry = manifest.get("levels", {}).get(level.key, {})
    if entry.get("weights"):
        candidate = Path(entry["weights"])
        if candidate.is_file():
            return candidate

    direct = weights_dir / level.best_weights_name()
    if direct.is_file():
        return direct
    return None


def play_level(
    level: MarioLevel,
    weights_path: Path,
    *,
    episodes: int,
    frame_skip: int,
    max_steps: int,
    render: bool,
    delay: float,
) -> list[dict]:
    agent = load_agent(weights_path, epsilon=0.0)
    results: list[dict] = []

    render_mode = "human" if render else None
    env = make_level_env(level, render_mode=render_mode)

    print(f"\n=== {level.key} ({level.env_id}) ===", flush=True)
    print(f"Weights: {weights_path}", flush=True)
    meta = getattr(agent, "meta", {})
    if meta:
        print(
            f"Saved score={meta.get('score')} x={meta.get('x_pos')} "
            f"flag={meta.get('flag_get')} (ep {meta.get('episode')})",
            flush=True,
        )

    try:
        for ep in range(episodes):
            stack, state = build_initial_stack(env)
            done = False
            step = 0
            total_reward = 0.0
            previous_x_pos = None
            last_info: dict = {}

            while not done and step < max_steps:
                action = int(np.argmax(agent.q_values(state)))
                frame, reward, done, info = env_step(env, action, frame_skip)
                last_info = info if isinstance(info, dict) else {}

                stack.append(preprocess_frame(frame))
                state = np.stack(list(stack), axis=0)
                total_reward += reward
                step += 1

                if delay > 0 and render:
                    time.sleep(delay)

            flag = bool(last_info.get("flag_get", False))
            x_pos = last_info.get("x_pos")
            results.append(
                {
                    "episode": ep,
                    "steps": step,
                    "reward": total_reward,
                    "x_pos": x_pos,
                    "flag_get": flag,
                }
            )
            print(
                f"  run {ep + 1}/{episodes}: steps={step} reward={total_reward:.1f} "
                f"x={x_pos} flag={flag}",
                flush=True,
            )
    finally:
        env.close()

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Play Mario levels with per-level best CNN-Transformer-DQN weights"
    )
    parser.add_argument(
        "--level",
        action="append",
        dest="levels",
        help="Level key like 1-1 (default: all first five)",
    )
    parser.add_argument("--all", action="store_true", help="Play every first-five level")
    parser.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHTS_DIR)
    parser.add_argument("--episodes", type=int, default=1, help="Runs per level")
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--delay", type=float, default=0.0, help="Sleep between steps when rendering")
    parser.add_argument("--no-render", action="store_true", help="Headless rollout (no window)")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = load_manifest(args.weights_dir / "manifest.json")

    if args.levels:
        levels = [level_from_key(k) for k in args.levels]
    else:
        levels = list(iter_first_five_levels())

    missing: list[str] = []
    for level in levels:
        weights_path = resolve_weights_path(level, args.weights_dir, manifest)
        if weights_path is None:
            missing.append(level.key)
            continue
        play_level(
            level,
            weights_path,
            episodes=args.episodes,
            frame_skip=args.frame_skip,
            max_steps=args.max_steps,
            render=not args.no_render,
            delay=args.delay,
        )

    if missing:
        print(
            "\nMissing weights for: " + ", ".join(missing),
            file=sys.stderr,
        )
        print(
            f"Train first: python train_mario_dqn.py  (expects {args.weights_dir}/best_*.npy)",
            file=sys.stderr,
        )
        sys.exit(1)

    print("\nAll requested levels finished.", flush=True)


if __name__ == "__main__":
    main()
