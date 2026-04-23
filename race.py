"""
Human-vs-ghost race mode for the MARIO project.

What this file does now:
- Lets a user control Mario with the keyboard.
- Runs through the same simplified action space described in the repo README.
- Renders gameplay through pygame so a ghost overlay can be added later.
- Includes a small ghost interface and drawing hook, but does NOT implement model playback yet.

What you can add later:
- Load a saved ghost trajectory from JSON / NPZ.
- Feed ghost positions into GhostReplay.get_state(step_idx).
- Replace the placeholder ghost marker with a sprite overlay.

Suggested usage:
    python race.py
    python race.py --scale 3 --fps 60
    python race.py --world 1 --stage 1

Controls:
- Right Arrow / D: move right
- Left Arrow / A: move left
- Space / J / Z: jump
- Left Shift / K / X: run
- R: reset current run
- Esc or window close: quit

Notes:
- This script intentionally focuses on human control + future ghost integration.
- It does not load or run the model yet.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pygame

try:
    import gym_super_mario_bros
    from nes_py.wrappers import JoypadSpace
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
except ImportError as exc:  # pragma: no cover - import-time environment dependency
    raise SystemExit(
        "Missing Mario environment dependencies. Install requirements.txt first.\n"
        "You likely need gym-super-mario-bros and nes-py available."
    ) from exc


# README says the project uses a simplified 7-action space.
# SIMPLE_MOVEMENT from gym-super-mario-bros usually matches that design closely.
# We slice/validate defensively so this file stays aligned with the repo concept.
EXPECTED_ACTIONS = [
    "NOOP",
    "RIGHT",
    "RIGHT_JUMP",
    "RIGHT_RUN",
    "RIGHT_JUMP_RUN",
    "JUMP",
    "LEFT",
]


@dataclass
class GhostState:
    """Single ghost state for future overlay playback."""

    step: int
    world_x: float
    world_y: float
    screen_x: Optional[float] = None
    screen_y: Optional[float] = None
    visible: bool = True


class GhostReplay:
    """
    Placeholder interface for future ghost playback.

    Later, this can load model trajectory data and return a GhostState for the
    current step. For now it always returns None, which cleanly disables overlay.
    """

    def __init__(self) -> None:
        self.enabled = False

    def reset(self) -> None:
        """Reset ghost playback state at the start of a run."""
        return None

    def get_state(self, step_idx: int, info: Optional[dict] = None) -> Optional[GhostState]:
        """Return the ghost state for the current frame, or None if unavailable."""
        _ = step_idx, info
        return None


class HumanController:
    """Maps keyboard input to the repo's simplified Mario action space."""

    def __init__(self) -> None:
        self.pressed: Set[int] = set()

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            self.pressed.add(event.key)
        elif event.type == pygame.KEYUP:
            self.pressed.discard(event.key)

    def current_action_index(self) -> int:
        left = self._is_pressed(pygame.K_LEFT, pygame.K_a)
        right = self._is_pressed(pygame.K_RIGHT, pygame.K_d)
        jump = self._is_pressed(pygame.K_SPACE, pygame.K_j, pygame.K_z, pygame.K_UP, pygame.K_w)
        run = self._is_pressed(pygame.K_LSHIFT, pygame.K_RSHIFT, pygame.K_k, pygame.K_x)

        # Priority favors the rightward race flow. If both left and right are held,
        # treat it as right when jump/run are present, otherwise noop.
        if right and jump and run:
            return 4  # Right + Jump + Run
        if right and run:
            return 3  # Right + Run
        if right and jump:
            return 2  # Right + Jump
        if right:
            return 1  # Right
        if jump:
            return 5  # Jump
        if left:
            return 6  # Left
        return 0  # No-op

    def _is_pressed(self, *keys: int) -> bool:
        return any(key in self.pressed for key in keys)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Mario manually and prepare for ghost racing.")
    parser.add_argument("--world", type=int, default=1, help="World number (default: 1)")
    parser.add_argument("--stage", type=int, default=1, help="Stage number (default: 1)")
    parser.add_argument("--version", type=int, default=0, help="Environment version suffix (default: 0)")
    parser.add_argument("--fps", type=int, default=60, help="Display/update FPS cap (default: 60)")
    parser.add_argument("--scale", type=int, default=3, help="Window scale multiplier (default: 3)")
    parser.add_argument(
        "--show-hud",
        action="store_true",
        help="Show basic HUD with step count, action, and progress info.",
    )
    return parser.parse_args()


def build_action_space() -> Sequence[Sequence[str]]:
    """
    Return a 7-action layout aligned with the repo README.

    gym-super-mario-bros SIMPLE_MOVEMENT generally contains these actions in order:
        0 No-op
        1 Right
        2 Right + A
        3 Right + B
        4 Right + A + B
        5 A
        6 Left
    """
    if len(SIMPLE_MOVEMENT) < 7:
        raise RuntimeError(
            f"Expected SIMPLE_MOVEMENT to have at least 7 actions, got {len(SIMPLE_MOVEMENT)}"
        )
    return SIMPLE_MOVEMENT[:7]


def make_env(world: int, stage: int, version: int):
    env_id = f"SuperMarioBros-{world}-{stage}-v{version}"

    # We request rgb_array so we can draw our own ghost overlay in pygame.
    # That is the most flexible setup for adding translucent ghost rendering later.
    # Older gym versions (which this repo likely uses) do NOT support render_mode in constructor
    # So we create normally and request rgb_array during render()
    env = gym_super_mario_bros.make(env_id)
    env = JoypadSpace(env, build_action_space())
    return env, env_id


def safe_reset(env) -> Tuple[np.ndarray, dict]:
    reset_result = env.reset()
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        obs, info = reset_result
        return obs, info
    return reset_result, {}


def safe_step(env, action: int) -> Tuple[np.ndarray, float, bool, dict]:
    step_result = env.step(action)

    # Gymnasium API
    if isinstance(step_result, tuple) and len(step_result) == 5:
        obs, reward, terminated, truncated, info = step_result
        done = bool(terminated or truncated)
        return obs, float(reward), done, info

    # Older Gym API
    if isinstance(step_result, tuple) and len(step_result) == 4:
        obs, reward, done, info = step_result
        return obs, float(reward), bool(done), info

    raise RuntimeError("Unexpected env.step(...) return format.")


def get_frame(env) -> np.ndarray:
    frame = env.render(mode="rgb_array")
    if frame is None:
        raise RuntimeError(
            "env.render() returned None. Ensure the environment supports render_mode='rgb_array'."
        )
    return np.asarray(frame)


def extract_progress(info: Optional[dict]) -> Dict[str, Optional[float]]:
    info = info or {}
    return {
        "x_pos": _maybe_float(info.get("x_pos")),
        "y_pos": _maybe_float(info.get("y_pos")),
        "time": _maybe_float(info.get("time")),
        "score": _maybe_float(info.get("score")),
        "coins": _maybe_float(info.get("coins")),
        "flag_get": bool(info.get("flag_get", False)),
    }


def _maybe_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def frame_to_surface(frame: np.ndarray, scale: int) -> pygame.Surface:
    # gym-super-mario-bros returns H x W x C, while pygame expects W x H x C.
    frame = np.transpose(frame, (1, 0, 2))
    surface = pygame.surfarray.make_surface(frame)
    if scale != 1:
        width, height = surface.get_size()
        surface = pygame.transform.scale(surface, (width * scale, height * scale))
    return surface


def draw_ghost_overlay(
    screen: pygame.Surface,
    ghost_state: Optional[GhostState],
    scale: int,
) -> None:
    """
    Placeholder ghost renderer.

    For now it draws a translucent marker only if a GhostState exists.
    Since GhostReplay currently returns None, nothing is rendered yet.
    This hook is where a future model ghost should be drawn.
    """
    if ghost_state is None or not ghost_state.visible:
        return

    x = ghost_state.screen_x
    y = ghost_state.screen_y
    if x is None or y is None:
        return

    radius = max(6, 4 * scale)
    overlay = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    pygame.draw.circle(overlay, (100, 220, 255, 140), (radius, radius), radius)
    screen.blit(overlay, (int(x * scale) - radius, int(y * scale) - radius))


def draw_hud(
    screen: pygame.Surface,
    font: pygame.font.Font,
    env_id: str,
    step_idx: int,
    action_idx: int,
    progress: Dict[str, Optional[float]],
) -> None:
    lines = [
        f"Env: {env_id}",
        f"Step: {step_idx}",
        f"Action: {action_idx} ({EXPECTED_ACTIONS[action_idx]})",
        f"X: {format_metric(progress['x_pos'])}  Y: {format_metric(progress['y_pos'])}",
        f"Time: {format_metric(progress['time'])}  Score: {format_metric(progress['score'])}",
        f"Coins: {format_metric(progress['coins'])}  Flag: {progress['flag_get']}",
        "R = reset    Esc = quit",
    ]

    padding = 8
    line_height = font.get_linesize()
    panel_width = max(font.size(line)[0] for line in lines) + padding * 2
    panel_height = line_height * len(lines) + padding * 2

    panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 165))

    for i, line in enumerate(lines):
        text = font.render(line, True, (255, 255, 255))
        panel.blit(text, (padding, padding + i * line_height))

    screen.blit(panel, (10, 10))


def format_metric(value: Optional[float]) -> str:
    if value is None:
        return "-"
    if value.is_integer():
        return str(int(value))
    return f"{value:.2f}"


def run_race(args: argparse.Namespace) -> int:
    pygame.init()
    pygame.display.set_caption("Mario Race Mode")
    font = pygame.font.SysFont("consolas", 18)
    clock = pygame.time.Clock()

    env, env_id = make_env(args.world, args.stage, args.version)
    controller = HumanController()
    ghost = GhostReplay()

    try:
        _, info = safe_reset(env)
        ghost.reset()
        step_idx = 0
        done = False

        initial_frame = get_frame(env)
        frame_surface = frame_to_surface(initial_frame, args.scale)
        screen = pygame.display.set_mode(frame_surface.get_size())

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    continue
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                    continue
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    _, info = safe_reset(env)
                    ghost.reset()
                    step_idx = 0
                    done = False
                    continue
                controller.handle_event(event)

            if not running:
                break

            action_idx = 0 if done else controller.current_action_index()

            if not done:
                _, _, done, info = safe_step(env, action_idx)
                step_idx += 1

            frame = get_frame(env)
            frame_surface = frame_to_surface(frame, args.scale)
            screen.blit(frame_surface, (0, 0))

            ghost_state = ghost.get_state(step_idx, info)
            draw_ghost_overlay(screen, ghost_state, args.scale)

            if args.show_hud:
                draw_hud(
                    screen=screen,
                    font=font,
                    env_id=env_id,
                    step_idx=step_idx,
                    action_idx=action_idx,
                    progress=extract_progress(info),
                )

            pygame.display.flip()
            clock.tick(args.fps)

        return 0
    finally:
        env.close()
        pygame.quit()


if __name__ == "__main__":
    sys.exit(run_race(parse_args()))
