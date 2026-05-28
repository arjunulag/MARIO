"""
Gym Mario web backend.

Run from web_gym_mario/backend:
    uvicorn server:app --reload --host 127.0.0.1 --port 8000

Then open the frontend at:
    http://127.0.0.1:3000
"""

from __future__ import annotations

import base64
import io
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gym_super_mario_bros
import numpy as np
from fastapi import FastAPI
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
PLAYER_SCREEN_X = 80
PLAYER_SCREEN_Y_FALLBACK = 207
# Tweak these if your local emulator draws the replay a few pixels off.
# Negative X moves the ghost left; positive X moves it right.
GHOST_SCREEN_X_OFFSET = -4
GHOST_SCREEN_Y_OFFSET = 0
GHOST_SPRITE_SHEET = Path(__file__).resolve().parent / "mario_ghost_sprites.png"
GHOST_SPRITE_FRAME_WIDTH = 18
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


class GhostReplay:
    """Replay ghost system.

    Ghosts are saved as world-position trajectories. During playback we convert
    ghost world_x to screen_x relative to the current player's world x. This is
    what fixes the ghost being drawn in the wrong place as the camera scrolls.
    """

    def __init__(self) -> None:
        self.ghost_id = "none"
        self.steps: List[Dict] = []
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
                data = json.loads(path.read_text())
                meta = data.get("meta", {})
                ghost_label = meta.get("ghost_id", path.stem)
                time_text = meta.get("finish_time_text") or meta.get("duration_text") or "--:--.---"
                level_text = f"{meta.get('world', '?')}-{meta.get('stage', '?')} v{meta.get('version', '?')}"
                ghosts.append({
                    "id": f"file:{path.name}",
                    "name": f"{ghost_label} | {time_text} | {level_text}",
                    "description": meta.get("description", "Recorded player ghost."),
                })
            except Exception:
                continue

        return ghosts

    def set_ghost(self, ghost_id: str) -> None:
        self.ghost_id = ghost_id or "none"
        self.steps = []
        self._last_replay_world_x = None
        self._last_replay_screen_x = None

        if self.ghost_id.startswith("file:"):
            filename = self.ghost_id.split(":", 1)[1]
            path = GHOST_DIR / filename
            if path.exists():
                data = json.loads(path.read_text())
                self.steps = sanitize_replay_steps(data.get("steps", []))
            else:
                self.ghost_id = "none"

    def reset(self) -> None:
        pass

    def get_state(self, step_idx: int, info: Dict) -> GhostState:
        current_x = safe_float(info.get("x_pos"), 0.0)

        if self.ghost_id == "demo":
            ghost_world_x = step_idx * 2.2
            ghost_y = PLAYER_SCREEN_Y_FALLBACK
            screen_x = world_x_to_screen_x(ghost_world_x, current_x)
            action_idx = 4 if (step_idx // 12) % 2 == 0 else 3
            return GhostState(enabled=True, x=screen_x, y=ghost_y, action_idx=action_idx, step_idx=step_idx)

        if self.steps:
            step = self.steps[min(step_idx, len(self.steps) - 1)]
            ghost_world_x = safe_float(step.get("x_pos"), 0.0)

            # Prevent bad/old recordings from snapping the ghost back to the start.
            # A real replay may move left a little, but it should not suddenly jump
            # hundreds of pixels backward in one frame.
            if self._last_replay_world_x is not None and ghost_world_x < self._last_replay_world_x - 120:
                ghost_world_x = self._last_replay_world_x
            self._last_replay_world_x = ghost_world_x

            # Treat y as a foot position. The SMB player sprite is 16px tall,
            # and the normal ground foot position is around y=207 in the 240px frame.
            raw_y = safe_float(step.get("screen_y"), PLAYER_SCREEN_Y_FALLBACK)
            ghost_y = raw_y if 150 <= raw_y <= 220 else PLAYER_SCREEN_Y_FALLBACK
            ghost_y += GHOST_SCREEN_Y_OFFSET

            action_idx = int(safe_float(step.get("action_idx"), 0))
            screen_x = world_x_to_screen_x(ghost_world_x, current_x)

            # Extra screen-space guard against one-frame glitches.
            if self._last_replay_screen_x is not None and abs(screen_x - self._last_replay_screen_x) > 160:
                screen_x = self._last_replay_screen_x
            self._last_replay_screen_x = screen_x

            return GhostState(enabled=True, x=screen_x, y=ghost_y, action_idx=action_idx, step_idx=step_idx)

        return GhostState(enabled=False)


class MarioSession:
    def __init__(self) -> None:
        self.env = None
        self.env_id = ""
        self.step_idx = 0
        self.done = False
        self.last_info: Dict = {}
        self.ghost = GhostReplay()
        self.recorded_steps: List[Dict] = []
        self.current_level = (1, 1, 0)
        self.last_action_idx = 0
        self.run_start_wall_time = 0.0
        self.lock = threading.Lock()

    def start(self, world: int = 1, stage: int = 1, version: int = 0, ghost_id: str = "none") -> FrameResponse:
        """Create/switch to a level and return the first rendered frame."""
        with self.lock:
            if self.env is not None:
                self.env.close()
                self.env = None

            self.current_level = (world, stage, version)
            self.env_id = f"SuperMarioBros-{world}-{stage}-v{version}"
            env = gym_super_mario_bros.make(self.env_id)
            env = JoypadSpace(env, ACTION_SPACE)

            self.env = env
            self.step_idx = 0
            self.done = False
            self.last_info = {}
            self.recorded_steps = []
            self.last_action_idx = 0
            self.run_start_wall_time = time.time()
            self.ghost.set_ghost(ghost_id)
            self.ghost.reset()
            self._reset_env_locked()

            return self._response_locked(action_idx=0)

    def reset(self) -> FrameResponse:
        with self.lock:
            if self.env is None:
                self.env_id = "SuperMarioBros-1-1-v0"
                env = gym_super_mario_bros.make(self.env_id)
                env = JoypadSpace(env, ACTION_SPACE)
                self.env = env

            self.step_idx = 0
            self.done = False
            self.last_info = {}
            self.recorded_steps = []
            self.last_action_idx = 0
            self.run_start_wall_time = time.time()
            self.ghost.reset()
            self._reset_env_locked()
            return self._response_locked(action_idx=0)

    def step(self, input_state: InputState) -> FrameResponse:
        with self.lock:
            if self.env is None:
                return self.start()

            action_idx = input_to_action(input_state)

            if not self.done:
                _, _, self.done, self.last_info = safe_step(self.env, action_idx)
                self.step_idx += 1
                self.last_action_idx = action_idx
                self._record_step_locked(action_idx)

                if self.done and self.last_info.get("flag_get"):
                    self._save_recording_locked()

            return self._response_locked(action_idx=action_idx)

    def _reset_env_locked(self) -> None:
        reset_result = self.env.reset()
        if isinstance(reset_result, tuple):
            self.last_info = reset_result[1] or {}
        else:
            self.last_info = {}

    def _record_step_locked(self, action_idx: int) -> None:
        # x_pos is world space. y_pos from this env is close enough for a
        # transparent overlay, but we clamp/fallback for safety.
        x_pos = safe_float(self.last_info.get("x_pos"), 0.0)
        y_pos = safe_float(self.last_info.get("y_pos"), PLAYER_SCREEN_Y_FALLBACK)
        if y_pos <= 0 or y_pos > 240:
            y_pos = PLAYER_SCREEN_Y_FALLBACK

        self.recorded_steps.append({
            "step": self.step_idx,
            "x_pos": x_pos,
            "screen_y": y_pos,
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
                "created_at": timestamp,
            },
            "steps": self.recorded_steps,
        }
        path.write_text(json.dumps(data, indent=2))

    def _response_locked(self, action_idx: int) -> FrameResponse:
        if self.env is None:
            return FrameResponse(ok=False, message="Environment has not started.")

        frame = get_frame(self.env)
        ghost_state = self.ghost.get_state(self.step_idx, self.last_info)
        frame = draw_ghost_overlay(frame, ghost_state)
        frame_uri = frame_to_data_uri(frame)

        info = sanitize_info(self.last_info)
        info["env_id"] = self.env_id

        return FrameResponse(
            ok=True,
            frame=frame_uri,
            step=self.step_idx,
            done=self.done,
            action=ACTION_NAMES[action_idx],
            info=info,
        )


session = MarioSession()


def sanitize_replay_steps(steps: List[Dict]) -> List[Dict]:
    """Sort and lightly clean replay steps to reduce random ghost teleports."""
    cleaned: List[Dict] = []
    for raw in steps:
        try:
            step_num = int(raw.get("step", len(cleaned)))
            x_pos = safe_float(raw.get("x_pos"), None)
            if x_pos is None or x_pos < 0:
                continue
            item = dict(raw)
            item["step"] = step_num
            item["x_pos"] = x_pos
            cleaned.append(item)
        except Exception:
            continue
    cleaned.sort(key=lambda item: int(item.get("step", 0)))
    return cleaned


def get_next_ghost_number() -> int:
    highest = 0
    for path in GHOST_DIR.glob("G*_w*.json"):
        stem = path.stem
        if len(stem) >= 4 and stem[0] == "G" and stem[1:4].isdigit():
            highest = max(highest, int(stem[1:4]))
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

    sprite = make_mario_ghost_sprite(ghost.action_idx, ghost.facing_left, getattr(ghost, "step_idx", 0))
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


def make_mario_ghost_sprite(action_idx: int = 0, facing_left: bool = False, step_idx: int = 0) -> Image.Image:
    """Return an actual Mario frame from the provided sprite sheet.

    This no longer draws a Mario-like placeholder. It uses extracted pixel-art
    frames from mario_ghost_sprites.png, which is generated from the provided
    sprite sheet image.
    """
    sprites = load_mario_ghost_sprites()

    jumping = action_idx in (2, 4, 5)
    running_or_walking = action_idx in (1, 2, 3, 4, 6)

    if jumping and len(sprites) > 5:
        frame_index = 5
    elif running_or_walking and len(sprites) > 4:
        walk_frames = [1, 2, 3, 4]
        frame_index = walk_frames[(step_idx // 4) % len(walk_frames)]
    else:
        frame_index = 0

    sprite = sprites[min(frame_index, len(sprites) - 1)].copy()

    if facing_left or action_idx == 6:
        sprite = sprite.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    return sprite

def world_x_to_screen_x(ghost_world_x: float, current_player_world_x: float) -> float:
    """Convert a world-space x position into the current screen x position.

    `x_pos` from gym is world-space. The rendered frame is screen-space.
    The camera is roughly 0 near the start, then follows Mario after he reaches
    a screen anchor. This estimate is much more accurate than assuming Mario is
    always at the anchor position.
    """
    camera_x = max(0.0, current_player_world_x - PLAYER_SCREEN_X)
    return ghost_world_x - camera_x + GHOST_SCREEN_X_OFFSET

def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


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
    return session.start(request.world, request.stage, request.version, request.ghost_id)


@app.post("/reset", response_model=FrameResponse)
def reset_game():
    return session.reset()


@app.post("/step", response_model=FrameResponse)
def step_game(request: StepRequest):
    return session.step(request.input)