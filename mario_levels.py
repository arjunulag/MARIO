"""Super Mario Bros level IDs and scoring for CNN–Transformer–DQN training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

# First five levels in game order: 1-1 … 1-4, then 2-1.
FIRST_FIVE_LEVELS: tuple[tuple[int, int], ...] = (
    (1, 1),
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 1),
)

DEFAULT_ROM_VERSION = 3  # rectangle ROM (matches prior SuperMarioBros-v3 training)


@dataclass(frozen=True)
class MarioLevel:
    world: int
    stage: int
    version: int = DEFAULT_ROM_VERSION

    @property
    def key(self) -> str:
        return f"{self.world}-{self.stage}"

    @property
    def env_id(self) -> str:
        return f"SuperMarioBros-{self.world}-{self.stage}-v{self.version}"

    def best_weights_name(self) -> str:
        return f"best_{self.key}.npy"


def iter_first_five_levels(version: int = DEFAULT_ROM_VERSION) -> Iterable[MarioLevel]:
    for world, stage in FIRST_FIVE_LEVELS:
        yield MarioLevel(world, stage, version)


def level_from_key(key: str, version: int = DEFAULT_ROM_VERSION) -> MarioLevel:
    world_s, stage_s = key.split("-", 1)
    return MarioLevel(int(world_s), int(stage_s), version)


def make_level_env(
    level: MarioLevel,
    *,
    render_mode: str | None = None,
):
    """Create a Joypad-wrapped env for one stage."""
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    from nes_py.wrappers import JoypadSpace

    env = gym_super_mario_bros.make(level.env_id, render_mode=render_mode)
    return JoypadSpace(env, SIMPLE_MOVEMENT)


def episode_score(
    *,
    x_pos: int | float | None,
    flag_get: bool = False,
    shaped_reward: float = 0.0,
) -> float:
    """
    Higher is better. Flag clears dominate; otherwise max x position wins.
    """
    x = float(x_pos or 0)
    if flag_get:
        return 1_000_000.0 + x
    return x + 0.001 * float(shaped_reward)


def parse_level_args(level_keys: Sequence[str] | None) -> list[MarioLevel]:
    if not level_keys:
        return list(iter_first_five_levels())
    return [level_from_key(k) for k in level_keys]
