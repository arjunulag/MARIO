"""Generate web replay ghosts from trained Mario DQN weights.

The saved weights do not contain trajectories, so this script rolls each
checkpoint out with epsilon=0 and records the resulting Mario positions in the
same JSON shape that server.py already replays.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


BACKEND_DIR = Path(__file__).resolve().parent
WEB_MARIO_DIR = BACKEND_DIR.parent
PROJECT_ROOT = WEB_MARIO_DIR.parent
DEFAULT_WEIGHTS_DIR = PROJECT_ROOT / "weights" / "mario_dqn"
DEFAULT_MANIFEST = DEFAULT_WEIGHTS_DIR / "manifest.json"
DEFAULT_OUTPUT_DIR = BACKEND_DIR / "ghosts"
NES_FRAME_MS = 1000 / 60

for import_path in (PROJECT_ROOT, BACKEND_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))


def install_numpy_pickle_compat() -> None:
    """Let NumPy 1.x load checkpoints pickled by NumPy 2.x."""
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)

import gym_super_mario_bros  # noqa: E402
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT  # noqa: E402
from mario_dqn_checkpoint import load_agent, load_manifest  # noqa: E402
from mario_levels import MarioLevel  # noqa: E402
from nes_py.wrappers import JoypadSpace  # noqa: E402
from preprocessFrames import build_initial_stack, preprocess_frame  # noqa: E402
from server import (  # noqa: E402
    ACTION_NAMES,
    MARIO_GROUND_Y_POS,
    current_mario_screen_foot_y,
    current_mario_screen_x,
    format_duration_ms,
    mario_y_pos_to_y_pixel,
    sanitize_replay_steps,
)


def leaky_relu(value: np.ndarray) -> np.ndarray:
    return np.where(value > 0, value, 0.01 * value)


def conv2d_valid(state: np.ndarray, kernels: np.ndarray) -> np.ndarray:
    """Inference-only valid convolution matching Tensor.conv2d(stride=1)."""
    kh = kernels.shape[2]
    kw = kernels.shape[3]
    windows = sliding_window_view(state, (kh, kw), axis=(1, 2))
    output = np.tensordot(kernels, windows, axes=([1, 2, 3], [0, 3, 4]))
    return output.astype(np.float32, copy=False)


class FastMarioPolicy:
    """Vectorized inference wrapper for the saved from-scratch agent."""

    def __init__(self, weights_path: Path) -> None:
        install_numpy_pickle_compat()
        self.agent = load_agent(weights_path, epsilon=0.0)

    def q_values(self, state: np.ndarray) -> np.ndarray:
        x = np.asarray(state, dtype=np.float32)
        x = leaky_relu(conv2d_valid(x, self.agent.kernels[0].data))
        x = leaky_relu(conv2d_valid(x, self.agent.kernels[1].data))
        x = x.reshape(-1)
        x = leaky_relu(x @ self.agent.W1.data.T + self.agent.b1.data)
        x = x @ self.agent.W2.data.T + self.agent.b2.data
        logits = self.agent.transformer.forward_from_embedding(x.reshape(1, 1, -1))
        return logits[0, 0]

    def select_action(self, state: np.ndarray) -> int:
        return int(np.argmax(self.q_values(state)))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_env_id(env_id: str, fallback_key: str) -> MarioLevel:
    if env_id.startswith("SuperMarioBros-"):
        try:
            tail = env_id.removeprefix("SuperMarioBros-")
            world_s, stage_s, version_s = tail.split("-", 2)
            return MarioLevel(int(world_s), int(stage_s), int(version_s.removeprefix("v")))
        except (TypeError, ValueError):
            pass

    world_s, stage_s = fallback_key.split("-", 1)
    return MarioLevel(int(world_s), int(stage_s))


def level_key(level: MarioLevel) -> str:
    return f"{level.world}-{level.stage}"


def resolve_weights_path(entry: dict[str, Any], level: MarioLevel, weights_dir: Path) -> Path | None:
    manifest_path = entry.get("weights")
    if manifest_path:
        candidate = Path(str(manifest_path))
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        if candidate.is_file():
            return candidate

    direct = weights_dir / level.best_weights_name()
    if direct.is_file():
        return direct
    return None


def iter_manifest_levels(
    manifest: dict[str, Any],
    weights_dir: Path,
    requested_levels: set[str] | None,
) -> Iterable[tuple[MarioLevel, Path, dict[str, Any]]]:
    for key, entry in sorted(manifest.get("levels", {}).items()):
        if requested_levels and key not in requested_levels:
            continue

        if not isinstance(entry, dict):
            continue

        level = parse_env_id(str(entry.get("env_id", "")), key)
        weights_path = resolve_weights_path(entry, level, weights_dir)
        if weights_path is None:
            print(f"Skipping {key}: no weights file found.", file=sys.stderr)
            continue
        yield level, weights_path, entry


def step_env_once(env, action_idx: int):
    result = env.step(action_idx)
    if len(result) == 5:
        frame, reward, terminated, truncated, info = result
        return frame, float(reward), bool(terminated or truncated), info or {}
    frame, reward, done, info = result
    return frame, float(reward), bool(done), info or {}


def make_rollout_env(level: MarioLevel):
    env = gym_super_mario_bros.make(level.env_id)
    return JoypadSpace(env, SIMPLE_MOVEMENT)


def record_step(env, info: dict[str, Any], step_idx: int, action_idx: int) -> dict[str, Any]:
    x_pos = safe_float(info.get("x_pos"), 0.0)
    y_pos = safe_float(info.get("y_pos"), MARIO_GROUND_Y_POS)
    action_name = ACTION_NAMES[action_idx] if action_idx < len(ACTION_NAMES) else str(action_idx)

    return {
        "step": step_idx,
        "x_pos": x_pos,
        "world_x": x_pos,
        "screen_x": current_mario_screen_x(env, info),
        "screen_foot_y": current_mario_screen_foot_y(env, info),
        "y_pos": y_pos,
        "y_pixel": mario_y_pos_to_y_pixel(y_pos),
        "action": action_name,
        "action_idx": action_idx,
    }


def output_filename(level: MarioLevel) -> str:
    return f"AI_w{level.world}-{level.stage}-v{level.version}.json"


def relative_to_project(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def build_ghost_payload(
    *,
    level: MarioLevel,
    weights_path: Path,
    manifest_entry: dict[str, Any],
    frame_skip: int,
    max_decisions: int,
    decisions: int,
    reward: float,
    last_info: dict[str, Any],
    steps: list[dict[str, Any]],
) -> dict[str, Any]:
    duration_ms = int(len(steps) * NES_FRAME_MS)
    duration_text = format_duration_ms(duration_ms)
    x_pos = safe_float(last_info.get("x_pos"), 0.0)
    flag_get = bool(last_info.get("flag_get", False))
    ghost_id = f"AI-{level.world}-{level.stage}-v{level.version}"
    status_text = "flag" if flag_get else f"x {int(x_pos)}"

    meta: dict[str, Any] = {
        "ghost_id": ghost_id,
        "name": f"AI {level.world}-{level.stage} v{level.version} | {duration_text} | {status_text}",
        "description": f"Greedy DQN rollout generated from {weights_path.name}.",
        "env_id": level.env_id,
        "world": level.world,
        "stage": level.stage,
        "version": level.version,
        "source": "weights/mario_dqn",
        "weights": relative_to_project(weights_path),
        "model_score": manifest_entry.get("score"),
        "model_episode": manifest_entry.get("episode"),
        "model_x_pos": manifest_entry.get("x_pos"),
        "model_flag_get": manifest_entry.get("flag_get"),
        "duration_ms": duration_ms,
        "duration_text": duration_text,
        "rollout_decisions": decisions,
        "rollout_frames": len(steps),
        "frame_skip": frame_skip,
        "max_decisions": max_decisions,
        "reward": reward,
        "x_pos": x_pos,
        "flag_get": flag_get,
        "created_at": int(time.time()),
    }

    if flag_get:
        meta["finish_time_ms"] = duration_ms
        meta["finish_time_text"] = duration_text

    return {
        "meta": meta,
        "steps": sanitize_replay_steps(steps),
    }


def generate_level_ghost(
    *,
    level: MarioLevel,
    weights_path: Path,
    manifest_entry: dict[str, Any],
    output_dir: Path,
    frame_skip: int,
    max_decisions: int,
) -> Path:
    policy = FastMarioPolicy(weights_path)
    env = make_rollout_env(level)
    steps: list[dict[str, Any]] = []
    last_info: dict[str, Any] = {}
    total_reward = 0.0
    decisions = 0

    try:
        stack, state = build_initial_stack(env)
        done = False

        while not done and decisions < max_decisions:
            action_idx = policy.select_action(state)
            last_frame = None

            for _ in range(frame_skip):
                frame, reward, done, info = step_env_once(env, action_idx)
                last_frame = frame
                last_info = info
                total_reward += reward
                steps.append(record_step(env, last_info, len(steps) + 1, action_idx))
                if done:
                    break

            if last_frame is None:
                break

            stack.append(preprocess_frame(last_frame))
            state = np.stack(list(stack), axis=0)
            decisions += 1
    finally:
        env.close()

    if not steps:
        raise RuntimeError(f"No rollout steps recorded for {level_key(level)}.")

    payload = build_ghost_payload(
        level=level,
        weights_path=weights_path,
        manifest_entry=manifest_entry,
        frame_skip=frame_skip,
        max_decisions=max_decisions,
        decisions=decisions,
        reward=total_reward,
        last_info=last_info,
        steps=steps,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename(level)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate web_mario ghosts from Mario DQN weights.")
    parser.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHTS_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--level", action="append", dest="levels", help="Level key like 1-1. Repeat for more.")
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--max-decisions", type=int, default=400)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights_dir = args.weights_dir.resolve()
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    requested_levels = set(args.levels or []) or None

    manifest = load_manifest(manifest_path)
    generated: list[Path] = []

    for level, weights_path, entry in iter_manifest_levels(manifest, weights_dir, requested_levels):
        print(f"Generating {level.env_id} from {relative_to_project(weights_path)}...", flush=True)
        output_path = generate_level_ghost(
            level=level,
            weights_path=weights_path,
            manifest_entry=entry,
            output_dir=output_dir,
            frame_skip=max(1, int(args.frame_skip)),
            max_decisions=max(1, int(args.max_decisions)),
        )
        generated.append(output_path)
        print(f"  wrote {output_path.name}", flush=True)

    if not generated:
        selected = ", ".join(sorted(requested_levels)) if requested_levels else "manifest levels"
        raise SystemExit(f"No ghosts generated for {selected}.")

    print(f"Generated {len(generated)} weight ghost(s).", flush=True)


if __name__ == "__main__":
    main()
