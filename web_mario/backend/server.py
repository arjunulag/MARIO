"""
Gym Mario web backend.

Run from web_gym_mario/backend:
    uvicorn server:app --reload --host 127.0.0.1 --port 8000

Then open the frontend at:
    http://127.0.0.1:3000
"""

from __future__ import annotations

import base64
from bisect import bisect_right
import io
import json
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import gym_super_mario_bros
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from PIL import Image, ImageDraw
from pydantic import BaseModel


app = FastAPI(title="Gym Mario Web Backend")

# Frame quality tuning:
# - Higher JPEG quality looks better but sends larger frames.
# - 85 is a good balance for local play.
# - Use 95 if quality matters more than lag.
JPEG_QUALITY = 85
GHOST_DIR = Path(__file__).resolve().parent / "ghosts"
GHOST_DIR.mkdir(exist_ok=True)
PLAYER_SCREEN_X = 120
PLAYER_SCREEN_Y_FALLBACK = 209
LEGACY_REPLAY_Y_POS_FALLBACK = 207
MARIO_GROUND_Y_POS = 79
MARIO_Y_PIXEL_GROUND = 255 - MARIO_GROUND_Y_POS
MARIO_Y_PIXEL_TO_FOOT_OFFSET = PLAYER_SCREEN_Y_FALLBACK - MARIO_Y_PIXEL_GROUND
MARIO_SPRITE_CENTER_X_OFFSET = 8
# Tweak these if your local emulator draws the replay a few pixels off.
# Negative X moves the ghost left; positive X moves it right.
GHOST_SCREEN_X_OFFSET = 0
GHOST_SCREEN_Y_OFFSET = 0
GHOST_SPRITE_SHEET = Path(__file__).resolve().parent / "mario_ghost_sprites.png"
GHOST_SPRITE_FRAME_WIDTH = 22
GHOST_SPRITE_FRAME_HEIGHT = 16
GHOST_SPRITE_SCALE = 1
_GHOST_SPRITES: Optional[List[Image.Image]] = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Keep this action space aligned with the previous race.py design.
ACTION_NAMES = [
    "NOOP",
    "RIGHT",
    "RIGHT_JUMP",
    "RIGHT_RUN",
    "RIGHT_JUMP_RUN",
    "JUMP",
    "LEFT",
]

ACTION_SPACE = SIMPLE_MOVEMENT[:7]


class InputState(BaseModel):
    right: bool = False
    left: bool = False
    jump: bool = False
    run: bool = False


class StepRequest(BaseModel):
    input: InputState = InputState()


class StartRequest(BaseModel):
    world: int = 1
    stage: int = 1
    version: int = 0
    ghost_id: str = "none"
    run_mode: str = "stable"


class FrameResponse(BaseModel):
    ok: bool
    frame: Optional[str] = None  # data:image/jpeg;base64,...
    step: int = 0
    done: bool = False
    action: str = "NOOP"
    info: Dict = {}
    message: str = ""


@dataclass
class GhostState:
    enabled: bool = False
    x: Optional[float] = None
    y: Optional[float] = None
    action_idx: int = 0
    facing_left: bool = False
    step_idx: int = 0
    airborne: bool = False


class GhostReplay:
    """Replay ghost system.

    Ghosts are saved as world-position trajectories. During playback we convert
    ghost world_x to screen_x relative to the current player's world x. This is
    what fixes the ghost being drawn in the wrong place as the camera scrolls.
    """

    def __init__(self) -> None:
        self.ghost_id = "none"
        self.steps: List[Dict] = []
        self.step_numbers: List[int] = []
        self._last_replay_world_x: Optional[float] = None
        self._last_replay_screen_x: Optional[float] = None

    @classmethod
    def list_ghosts(cls) -> List[Dict[str, str]]:
        ghosts = [
            {
                "id": "none",
                "name": "No ghost",
                "description": "Play without a ghost.",
            },
            {
                "id": "demo",
                "name": "Demo ghost",
                "description": "A built-in example replay ghost.",
            },
        ]

        for path in sorted(GHOST_DIR.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                meta = ghost_metadata(data)
                ghost_label = meta.get("ghost_id") or meta.get("id") or path.stem
                time_text = ghost_time_text(meta)
                level_text = ghost_level_text(meta)
                name = meta.get("name") or f"{ghost_label} | {time_text} | {level_text}"
                ghosts.append({
                    "id": f"file:{path.name}",
                    "name": name,
                    "description": meta.get("description", "Recorded player ghost."),
                })
            except Exception:
                continue

        return ghosts

    def set_ghost(self, ghost_id: str) -> None:
        self.ghost_id = ghost_id or "none"
        self.steps = []
        self.step_numbers = []
        self._last_replay_world_x = None
        self._last_replay_screen_x = None

        if self.ghost_id.startswith("file:"):
            filename = self.ghost_id.split(":", 1)[1]
            path = GHOST_DIR / filename
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                self.steps = sanitize_replay_steps(data.get("steps", []))
                self.step_numbers = [int(step.get("step", index + 1)) for index, step in enumerate(self.steps)]
            else:
                self.ghost_id = "none"

    def reset(self) -> None:
        self._last_replay_world_x = None
        self._last_replay_screen_x = None

    def get_state(self, step_idx: int, info: Dict, camera_x: Optional[float] = None) -> GhostState:
        current_x = safe_float(info.get("x_pos"), 0.0)
        if camera_x is None:
            camera_x = fallback_camera_x(current_x)

        if self.ghost_id == "demo":
            ghost_world_x = step_idx * 2.2
            ghost_y = PLAYER_SCREEN_Y_FALLBACK
            screen_x = world_x_to_screen_x(ghost_world_x, camera_x)
            action_idx = 4 if (step_idx // 12) % 2 == 0 else 3
            return GhostState(enabled=True, x=screen_x, y=ghost_y, action_idx=action_idx, step_idx=step_idx)

        if self.steps:
            step = self.steps[self._step_index_for(step_idx)]
            ghost_world_x = safe_float(step.get("x_pos"), 0.0)

            # Prevent bad/old recordings from snapping the ghost back to the start.
            # A real replay may move left a little, but it should not suddenly jump
            # hundreds of pixels backward in one frame.
            if self._last_replay_world_x is not None and ghost_world_x < self._last_replay_world_x - 120:
                ghost_world_x = self._last_replay_world_x
            self._last_replay_world_x = ghost_world_x

            ghost_y = safe_float(step.get("screen_foot_y"), PLAYER_SCREEN_Y_FALLBACK)
            ghost_y += GHOST_SCREEN_Y_OFFSET

            action_idx = int(safe_float(step.get("action_idx"), 0))
            screen_x = world_x_to_screen_x(ghost_world_x, camera_x)

            # Extra screen-space guard against one-frame glitches.
            if self._last_replay_screen_x is not None and abs(screen_x - self._last_replay_screen_x) > 160:
                screen_x = self._last_replay_screen_x
            self._last_replay_screen_x = screen_x

            return GhostState(
                enabled=True,
                x=screen_x,
                y=ghost_y,
                action_idx=action_idx,
                facing_left=action_idx == 6,
                step_idx=step_idx,
                airborne=ghost_y < PLAYER_SCREEN_Y_FALLBACK - 4,
            )

        return GhostState(enabled=False)

    def _step_index_for(self, step_idx: int) -> int:
        if not self.step_numbers:
            return min(max(0, step_idx - 1), len(self.steps) - 1)

        index = bisect_right(self.step_numbers, step_idx) - 1
        if index < 0:
            return 0
        return min(index, len(self.steps) - 1)


class MarioSession:
    def __init__(self) -> None:
        self.env = None
        self.env_id = ""
        self.step_idx = 0
        self.done = False
        self.last_info: Dict = {}
        self.last_frame: Optional[np.ndarray] = None
        self.ghost = GhostReplay()
        self.recorded_steps: List[Dict] = []
        self.current_level = (1, 1, 0)
        self.current_run_mode = "stable"
        self.last_action_idx = 0
        self.run_start_wall_time = 0.0
        self.lock = threading.Lock()

    def start(self, world: int = 1, stage: int = 1, version: int = 0, ghost_id: str = "none", run_mode: str = "stable") -> FrameResponse:
        """Create/switch to a level and return the first rendered frame."""
        with self.lock:
            self._start_locked(world, stage, version, ghost_id, run_mode)
            return self._response_locked(action_idx=0)

    def start_packet(self, world: int = 1, stage: int = 1, version: int = 0, ghost_id: str = "none", run_mode: str = "stable") -> bytes:
        with self.lock:
            self._start_locked(world, stage, version, ghost_id, run_mode)
            return self._packet_locked(action_idx=0)

    def reset(self) -> FrameResponse:
        with self.lock:
            self._reset_locked()
            return self._response_locked(action_idx=0)

    def reset_packet(self) -> bytes:
        with self.lock:
            self._reset_locked()
            return self._packet_locked(action_idx=0)

    def step(self, input_state: InputState) -> FrameResponse:
        with self.lock:
            if self.env is None:
                self._start_locked()

            action_idx = self._step_locked(input_state)
            return self._response_locked(action_idx=action_idx)

    def step_packet(self, input_state: InputState) -> bytes:
        with self.lock:
            if self.env is None:
                self._start_locked()

            action_idx = self._step_locked(input_state)
            return self._packet_locked(action_idx=action_idx)

    def _start_locked(self, world: int = 1, stage: int = 1, version: int = 0, ghost_id: str = "none", run_mode: str = "stable") -> None:
        if self.env is not None:
            self.env.close()
            self.env = None

        self.current_level = (world, stage, version)
        self.current_run_mode = sanitize_run_mode(run_mode)
        self.env_id = f"SuperMarioBros-{world}-{stage}-v{version}"
        env = gym_super_mario_bros.make(self.env_id)
        env = JoypadSpace(env, ACTION_SPACE)

        self.env = env
        self.step_idx = 0
        self.done = False
        self.last_info = {}
        self.last_frame = None
        self.recorded_steps = []
        self.last_action_idx = 0
        self.run_start_wall_time = time.time()
        self.ghost.set_ghost(ghost_id)
        self.ghost.reset()
        self._reset_env_locked()

    def _reset_locked(self) -> None:
        if self.env is None:
            self.env_id = "SuperMarioBros-1-1-v0"
            env = gym_super_mario_bros.make(self.env_id)
            env = JoypadSpace(env, ACTION_SPACE)
            self.env = env

        self.step_idx = 0
        self.done = False
        self.last_info = {}
        self.last_frame = None
        self.recorded_steps = []
        self.last_action_idx = 0
        self.run_start_wall_time = time.time()
        self.ghost.reset()
        self._reset_env_locked()

    def _step_locked(self, input_state: InputState) -> int:
        action_idx = input_to_action(input_state)

        if not self.done:
            obs, _, self.done, self.last_info = safe_step(self.env, action_idx)
            self.last_frame = np.asarray(obs).copy()
            self.step_idx += 1
            self.last_action_idx = action_idx
            self._record_step_locked(action_idx)

            if self.done and self.last_info.get("flag_get"):
                self._save_recording_locked()

        return action_idx

    def _reset_env_locked(self) -> None:
        reset_result = self.env.reset()
        if isinstance(reset_result, tuple):
            self.last_frame = np.asarray(reset_result[0]).copy()
            self.last_info = reset_result[1] or {}
        else:
            self.last_frame = np.asarray(reset_result).copy()
            self.last_info = {}

    def _record_step_locked(self, action_idx: int) -> None:
        # x_pos is world space. y_pos is inverted RAM distance from the bottom,
        # so store a normalized screen-foot coordinate for stable replay.
        x_pos = safe_float(self.last_info.get("x_pos"), 0.0)
        y_pos = safe_float(self.last_info.get("y_pos"), MARIO_GROUND_Y_POS)
        screen_foot_y = current_mario_screen_foot_y(self.env, self.last_info)
        screen_x = current_mario_screen_x(self.env, self.last_info)

        self.recorded_steps.append({
            "step": self.step_idx,
            "x_pos": x_pos,
            "world_x": x_pos,
            "screen_x": screen_x,
            "screen_foot_y": screen_foot_y,
            "y_pos": y_pos,
            "y_pixel": mario_y_pos_to_y_pixel(y_pos),
            "action": ACTION_NAMES[action_idx],
            "action_idx": action_idx,
        })

    def _save_recording_locked(self) -> None:
        if len(self.recorded_steps) < 10:
            return

        world, stage, version = self.current_level
        timestamp = int(time.time())
        next_id = get_next_ghost_number()
        ghost_id = f"G{next_id:03d}"
        finish_time_ms = max(0, int((time.time() - self.run_start_wall_time) * 1000))
        finish_time_text = format_duration_ms(finish_time_ms)
        filename = f"{ghost_id}_w{world}-{stage}-v{version}_{timestamp}.json"
        path = GHOST_DIR / filename
        data = {
            "meta": {
                "ghost_id": ghost_id,
                "name": f"{ghost_id} | {finish_time_text} | {world}-{stage} v{version}",
                "description": f"Recorded finish {ghost_id} for World {world}-{stage} v{version} in {finish_time_text}.",
                "finish_time_ms": finish_time_ms,
                "finish_time_text": finish_time_text,
                "world": world,
                "stage": stage,
                "version": version,
                "run_mode": self.current_run_mode,
                "created_at": timestamp,
            },
            "steps": self.recorded_steps,
        }
        path.write_text(json.dumps(data, indent=2))

    def _response_locked(self, action_idx: int) -> FrameResponse:
        if self.env is None:
            return FrameResponse(ok=False, message="Environment has not started.")

        frame, meta = self._frame_and_meta_locked(action_idx)
        frame_uri = frame_to_data_uri(frame)

        return FrameResponse(
            frame=frame_uri,
            **meta,
        )

    def _packet_locked(self, action_idx: int) -> bytes:
        if self.env is None:
            return build_frame_packet(np.zeros((240, 256, 3), dtype=np.uint8), {
                "ok": False,
                "step": self.step_idx,
                "done": self.done,
                "action": ACTION_NAMES[action_idx],
                "info": {},
                "message": "Environment has not started.",
            })

        frame, meta = self._frame_and_meta_locked(action_idx)
        return build_frame_packet(frame, meta)

    def _frame_and_meta_locked(self, action_idx: int) -> Tuple[np.ndarray, Dict]:
        frame = self.last_frame
        if frame is None:
            frame = get_frame(self.env)
        else:
            frame = frame.copy()
        camera_x = current_camera_x(self.env, self.last_info)
        ghost_state = self.ghost.get_state(self.step_idx, self.last_info, camera_x)
        frame = draw_ghost_overlay(frame, ghost_state)

        info = sanitize_info(self.last_info)
        info["env_id"] = self.env_id
        info["run_mode"] = self.current_run_mode
        info["camera_x"] = camera_x

        return frame, {
            "ok": True,
            "step": self.step_idx,
            "done": self.done,
            "action": ACTION_NAMES[action_idx],
            "info": info,
            "message": "",
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
            "channels": int(frame.shape[2]) if frame.ndim == 3 else 1,
        }


session = MarioSession()


def ghost_metadata(data: Dict) -> Dict:
    """Return ghost metadata, accepting both saved file shapes."""
    meta = data.get("meta")
    if isinstance(meta, dict):
        return {**data, **meta}
    return data


def ghost_time_text(meta: Dict) -> str:
    if meta.get("finish_time_text"):
        return str(meta["finish_time_text"])
    if meta.get("duration_text"):
        return str(meta["duration_text"])
    if meta.get("time"):
        return str(meta["time"])

    elapsed_ms = safe_float(meta.get("finish_time_ms"), None)
    if elapsed_ms is None:
        elapsed_ms = safe_float(meta.get("elapsed_ms"), None)
    if elapsed_ms is not None:
        return format_duration_ms(max(0, int(elapsed_ms)))

    return "--:--.---"


def ghost_level_text(meta: Dict) -> str:
    env_id = str(meta.get("env_id", ""))
    if env_id.startswith("SuperMarioBros-"):
        return env_id.replace("SuperMarioBros-", "").replace("-v", " v")

    world = meta.get("world", "?")
    stage = meta.get("stage", "?")
    version = meta.get("version", "?")
    return f"{world}-{stage} v{version}"


def normalize_action_idx(value) -> int:
    if isinstance(value, str):
        if value in ACTION_NAMES:
            return ACTION_NAMES.index(value)
        value = safe_float(value, 0)
    return max(0, min(len(ACTION_NAMES) - 1, int(safe_float(value, 0))))


def mario_y_pos_to_y_pixel(y_pos: float) -> float:
    return 255.0 - safe_float(y_pos, MARIO_GROUND_Y_POS)


def mario_y_pos_to_screen_foot_y(y_pos: float) -> float:
    y_pos = safe_float(y_pos, MARIO_GROUND_Y_POS)
    return PLAYER_SCREEN_Y_FALLBACK - (y_pos - MARIO_GROUND_Y_POS)


def mario_y_pixel_to_screen_foot_y(y_pixel: float) -> float:
    return safe_float(y_pixel, MARIO_Y_PIXEL_GROUND) + MARIO_Y_PIXEL_TO_FOOT_OFFSET


def infer_legacy_screen_y_mode(steps: List[Dict]) -> str:
    """Classify old `screen_y` fields before per-step normalization."""
    if any("screen_foot_y" in step or "y_pos" in step or "y_pixel" in step for step in steps):
        return "explicit"

    screen_y_values = [safe_float(step.get("screen_y"), None) for step in steps]
    screen_y_values = [value for value in screen_y_values if value is not None]
    if not screen_y_values:
        return "screen_foot"

    x_pos_count = sum("x_pos" in step for step in steps)
    world_x_count = sum("world_x" in step for step in steps)

    if x_pos_count and not world_x_count:
        return "y_pos"
    if world_x_count and not x_pos_count:
        return "y_pixel"

    groundish_count = sum(70 <= value <= 90 for value in screen_y_values)
    pixel_groundish_count = sum(168 <= value <= 184 for value in screen_y_values)
    if groundish_count > pixel_groundish_count:
        return "y_pos"
    if pixel_groundish_count > groundish_count:
        return "y_pixel"
    return "screen_foot"


def normalize_replay_screen_foot_y(raw: Dict, screen_y_mode: str = "screen_foot") -> float:
    screen_foot_y = safe_float(raw.get("screen_foot_y"), None)
    if screen_foot_y is not None:
        return screen_foot_y

    y_pos = safe_float(raw.get("y_pos"), None)
    if y_pos is not None:
        return mario_y_pos_to_screen_foot_y(y_pos)

    y_pixel = safe_float(raw.get("y_pixel"), None)
    if y_pixel is not None:
        return mario_y_pixel_to_screen_foot_y(y_pixel)

    screen_y = safe_float(raw.get("screen_y"), None)
    if screen_y is None:
        return PLAYER_SCREEN_Y_FALLBACK

    if screen_y_mode == "y_pos":
        return mario_y_pos_to_screen_foot_y(screen_y)
    if screen_y_mode == "y_pixel":
        return mario_y_pixel_to_screen_foot_y(screen_y)

    # Older hand-exported files used `screen_y` for SMB's RAM y pixel
    # (ground around 176), while early web_mario files used it for y_pos
    # (ground around 79). Distinguish them by the companion x field shape.
    if "world_x" in raw and "x_pos" not in raw:
        return mario_y_pixel_to_screen_foot_y(screen_y)
    if screen_y <= 160:
        return mario_y_pos_to_screen_foot_y(screen_y)
    return screen_y


def repair_legacy_y_pos_fallback_runs(steps: List[Dict]) -> None:
    """Smooth old recordings where invalid y_pos was clamped to 207."""
    index = 0
    while index < len(steps):
        screen_y = safe_float(steps[index].get("screen_y"), None)
        if screen_y != LEGACY_REPLAY_Y_POS_FALLBACK:
            index += 1
            continue

        run_start = index
        while (
            index < len(steps)
            and safe_float(steps[index].get("screen_y"), None) == LEGACY_REPLAY_Y_POS_FALLBACK
        ):
            index += 1
        run_end = index
        run_length = run_end - run_start

        if run_length < 3 or run_start == 0 or run_end >= len(steps):
            continue

        previous_y = safe_float(steps[run_start - 1].get("screen_y"), None)
        next_y = safe_float(steps[run_end].get("screen_y"), None)
        if previous_y is None or next_y is None:
            continue
        if previous_y < 225 or next_y < 225:
            continue

        for offset, step_index in enumerate(range(run_start, run_end), start=1):
            t = offset / (run_length + 1)
            interpolated_y = previous_y + (next_y - previous_y) * t
            steps[step_index]["screen_foot_y"] = mario_y_pos_to_screen_foot_y(interpolated_y)


def get_env_ram(env) -> Optional[np.ndarray]:
    raw_env = getattr(env, "unwrapped", None)
    ram = getattr(raw_env, "ram", None)
    if ram is None:
        return None
    return ram


def current_mario_screen_x(env, info: Dict) -> float:
    ram = get_env_ram(env)
    if ram is not None:
        return float((int(ram[0x86]) - int(ram[0x071c])) % 256)
    return safe_float(info.get("x_pos"), 0.0) - fallback_camera_x(safe_float(info.get("x_pos"), 0.0))


def current_camera_x(env, info: Dict) -> float:
    ram = get_env_ram(env)
    if ram is not None:
        world_x = float(int(ram[0x6d]) * 0x100 + int(ram[0x86]))
        return world_x - current_mario_screen_x(env, info)
    return fallback_camera_x(safe_float(info.get("x_pos"), 0.0))


def current_mario_screen_foot_y(env, info: Dict) -> float:
    ram = get_env_ram(env)
    if ram is not None:
        return mario_y_pixel_to_screen_foot_y(float(int(ram[0x03b8])))
    return mario_y_pos_to_screen_foot_y(safe_float(info.get("y_pos"), MARIO_GROUND_Y_POS))


def fallback_camera_x(current_player_world_x: float) -> float:
    return max(0.0, safe_float(current_player_world_x, 0.0) - PLAYER_SCREEN_X)


def ghost_number_from_text(value) -> Optional[int]:
    text = str(value or "")
    if len(text) < 2 or text[0].upper() != "G":
        return None

    digits = []
    for char in text[1:]:
        if not char.isdigit():
            break
        digits.append(char)

    if not digits:
        return None
    return int("".join(digits))


def sanitize_replay_steps(steps: List[Dict]) -> List[Dict]:
    """Sort and lightly clean replay steps to reduce random ghost teleports."""
    cleaned: List[Dict] = []
    screen_y_mode = infer_legacy_screen_y_mode(steps)
    for raw in steps:
        try:
            step_num = int(raw.get("step", len(cleaned)))
            x_pos = safe_float(raw.get("x_pos"), None)
            if x_pos is None:
                x_pos = safe_float(raw.get("world_x"), None)
            if x_pos is None or x_pos < 0:
                continue
            item = dict(raw)
            item["step"] = step_num
            item["x_pos"] = x_pos
            item["screen_foot_y"] = normalize_replay_screen_foot_y(raw, screen_y_mode)
            item["action_idx"] = normalize_action_idx(raw.get("action_idx", raw.get("action", 0)))
            cleaned.append(item)
        except Exception:
            continue
    cleaned.sort(key=lambda item: int(item.get("step", 0)))
    if screen_y_mode == "y_pos":
        repair_legacy_y_pos_fallback_runs(cleaned)
    return cleaned


def get_next_ghost_number() -> int:
    highest = 0
    for path in GHOST_DIR.glob("*.json"):
        number = ghost_number_from_text(path.stem)
        if number is None:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                meta = ghost_metadata(data)
                number = ghost_number_from_text(meta.get("ghost_id") or meta.get("id"))
            except Exception:
                number = None
        if number is not None:
            highest = max(highest, number)
    return highest + 1


def format_duration_ms(ms: int) -> str:
    minutes = ms // 60000
    seconds = (ms % 60000) // 1000
    millis = ms % 1000
    return f"{minutes:02d}:{seconds:02d}.{millis:03d}"


def input_to_action(input_state: InputState) -> int:
    right = input_state.right
    left = input_state.left
    jump = input_state.jump
    run = input_state.run

    if right and jump and run:
        return 4
    if right and run:
        return 3
    if right and jump:
        return 2
    if right:
        return 1
    if jump:
        return 5
    if left:
        return 6
    return 0


def safe_step(env, action: int):
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, reward, bool(terminated or truncated), info
    if len(result) == 4:
        obs, reward, done, info = result
        return obs, reward, bool(done), info
    raise RuntimeError("Unexpected env.step result format")


def get_frame(env) -> np.ndarray:
    try:
        frame = env.render(mode="rgb_array")
    except TypeError:
        frame = env.render()

    if frame is None:
        raise RuntimeError("env.render returned None")

    return np.asarray(frame).copy()


def draw_ghost_overlay(frame: np.ndarray, ghost: GhostState) -> np.ndarray:
    if not ghost.enabled or ghost.x is None or ghost.y is None:
        return frame

    image = Image.fromarray(frame).convert("RGBA")
    x = int(ghost.x)
    y = int(ghost.y)

    # If the ghost is far off-screen, do not draw it.
    if x < -40 or x > image.width + 40:
        return frame

    sprite = make_mario_ghost_sprite(
        ghost.action_idx,
        ghost.facing_left,
        getattr(ghost, "step_idx", 0),
        getattr(ghost, "airborne", False),
    )
    # x is the approximate center of Mario; y is the foot/bottom position.
    paste_x = x - sprite.width // 2
    paste_y = y - sprite.height
    image.alpha_composite(sprite, (paste_x, paste_y))
    return np.asarray(image.convert("RGB"))


def load_mario_ghost_sprites() -> List[Image.Image]:
    """Load real Small Mario frames extracted from the provided sprite sheet.

    The file mario_ghost_sprites.png should live next to server.py. It contains
    transparent 18x16 frames from the sprite sheet the user provided.
    """
    global _GHOST_SPRITES
    if _GHOST_SPRITES is not None:
        return _GHOST_SPRITES

    if not GHOST_SPRITE_SHEET.exists():
        raise FileNotFoundError(
            f"Missing {GHOST_SPRITE_SHEET.name}. Put it in the same folder as server.py."
        )

    sheet = Image.open(GHOST_SPRITE_SHEET).convert("RGBA")
    sprites: List[Image.Image] = []

    frame_count = sheet.width // GHOST_SPRITE_FRAME_WIDTH
    for index in range(frame_count):
        left = index * GHOST_SPRITE_FRAME_WIDTH
        frame = sheet.crop((
            left,
            0,
            left + GHOST_SPRITE_FRAME_WIDTH,
            GHOST_SPRITE_FRAME_HEIGHT,
        ))

        if GHOST_SPRITE_SCALE != 1:
            frame = frame.resize(
                (
                    frame.width * GHOST_SPRITE_SCALE,
                    frame.height * GHOST_SPRITE_SCALE,
                ),
                Image.Resampling.NEAREST,
            )

        sprites.append(frame)

    _GHOST_SPRITES = sprites
    return sprites


def make_mario_ghost_sprite(
    action_idx: int = 0,
    facing_left: bool = False,
    step_idx: int = 0,
    airborne: bool = False,
) -> Image.Image:
    """Return an actual Mario frame from the provided sprite sheet.

    It uses extracted pixel-art frames from mario_ghost_sprites.png, 
    which is generated from the provided sprite sheet image.
    """
    sprites = load_mario_ghost_sprites()

    jumping = airborne or action_idx in (2, 4, 5)
    running_or_walking = action_idx in (1, 2, 3, 4, 6)

    if jumping and len(sprites) > 5:
        frame_index = 5
    elif running_or_walking and len(sprites) > 4:
        walk_frames = [1, 2, 3]
        frame_index = walk_frames[(step_idx // 4) % len(walk_frames)]
    else:
        frame_index = 0

    sprite = sprites[min(frame_index, len(sprites) - 1)].copy()

    if facing_left or action_idx == 6:
        sprite = sprite.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    return sprite

def world_x_to_screen_x(ghost_world_x: float, camera_x: float) -> float:
    """Convert a world-space x position into the current screen x position.

    `x_pos` from gym is world-space. The rendered frame is screen-space, so
    replay world positions need to be offset by the current camera/viewport.
    """
    return ghost_world_x - camera_x + MARIO_SPRITE_CENTER_X_OFFSET + GHOST_SCREEN_X_OFFSET

def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def sanitize_run_mode(value: str) -> str:
    if value in {"competition", "stable", "laptop"}:
        return value
    return "stable"


def frame_to_data_uri(frame: np.ndarray) -> str:
    """Encode a frame for the browser.

    JPEG is faster/smaller than PNG for this HTTP polling approach.
    Quality 85 keeps pixel art cleaner while still being much lighter than PNG.
    """
    image = Image.fromarray(frame)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=False, subsampling=0)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_frame_packet(frame: np.ndarray, meta: Dict) -> bytes:
    rgb_frame = np.ascontiguousarray(frame, dtype=np.uint8)

    header = {
        **meta,
        "width": int(rgb_frame.shape[1]),
        "height": int(rgb_frame.shape[0]),
        "channels": int(rgb_frame.shape[2]) if rgb_frame.ndim == 3 else 1,
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return struct.pack("<I", len(header_bytes)) + header_bytes + rgb_frame.tobytes()


def sanitize_info(info: Dict) -> Dict:
    output = {}
    for key, value in (info or {}).items():
        if isinstance(value, np.generic):
            output[key] = value.item()
        elif isinstance(value, np.ndarray):
            output[key] = value.tolist()
        else:
            output[key] = value
    return output


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ghosts")
def list_ghosts():
    return {"ghosts": GhostReplay.list_ghosts()}


@app.post("/start", response_model=FrameResponse)
def start_game(request: StartRequest):
    return session.start(request.world, request.stage, request.version, request.ghost_id, request.run_mode)


@app.post("/reset", response_model=FrameResponse)
def reset_game():
    return session.reset()


@app.post("/step", response_model=FrameResponse)
def step_game(request: StepRequest):
    return session.step(request.input)


@app.websocket("/ws")
async def websocket_game(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            raw_message = await websocket.receive_text()
            request = json.loads(raw_message)
            request_type = request.get("type", "step")

            if request_type == "start":
                packet = session.start_packet(
                    int(request.get("world", 1)),
                    int(request.get("stage", 1)),
                    int(request.get("version", 0)),
                    str(request.get("ghost_id", "none")),
                    str(request.get("run_mode", "stable")),
                )
            elif request_type == "reset":
                packet = session.reset_packet()
            else:
                packet = session.step_packet(InputState(**request.get("input", {})))

            await websocket.send_bytes(packet)
    except WebSocketDisconnect:
        return
