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
import threading
from dataclasses import dataclass
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


class GhostReplay:
    """
    Placeholder for later model ghost integration.

    Later, this class can:
    - load a saved model trajectory
    - return ghost x/y for the current step
    - sync by frame number or by x-position
    """

    def reset(self) -> None:
        pass

    def get_state(self, step_idx: int, info: Dict) -> GhostState:
        _ = step_idx, info
        return GhostState(enabled=False)


class MarioSession:
    def __init__(self) -> None:
        self.env = None
        self.env_id = ""
        self.step_idx = 0
        self.done = False
        self.last_info: Dict = {}
        self.ghost = GhostReplay()
        self.lock = threading.Lock()

    def start(self, world: int = 1, stage: int = 1, version: int = 0) -> FrameResponse:
        """Create/switch to a level and return the first rendered frame."""
        with self.lock:
            if self.env is not None:
                self.env.close()
                self.env = None

            self.env_id = f"SuperMarioBros-{world}-{stage}-v{version}"
            env = gym_super_mario_bros.make(self.env_id)
            env = JoypadSpace(env, ACTION_SPACE)

            self.env = env
            self.step_idx = 0
            self.done = False
            self.last_info = {}
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

            return self._response_locked(action_idx=action_idx)

    def _reset_env_locked(self) -> None:
        reset_result = self.env.reset()
        if isinstance(reset_result, tuple):
            self.last_info = reset_result[1] or {}
        else:
            self.last_info = {}

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

    image = Image.fromarray(frame)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    radius = 8
    x = int(ghost.x)
    y = int(ghost.y)
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=(0, 255, 255, 140),
    )

    image = image.convert("RGBA")
    image.alpha_composite(overlay)
    return np.asarray(image.convert("RGB"))


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


@app.post("/start", response_model=FrameResponse)
def start_game(request: StartRequest):
    return session.start(request.world, request.stage, request.version)


@app.post("/reset", response_model=FrameResponse)
def reset_game():
    return session.reset()


@app.post("/step", response_model=FrameResponse)
def step_game(request: StepRequest):
    return session.step(request.input)