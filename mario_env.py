"""Mario environment factory for reinforcement learning.

This module builds a Gymnasium-compatible Super Mario Bros environment with:
- reduced acti([github.com](https://github.com/rickyegl/gymnasium-super-mario-bros?utm_source=chatgpt.com))annel-first output
- optional frame stacking
- simple reward shaping

Install:
    pip install gym-super-mario-bros gymnasium nes-py opencv-python numpy

Notes:
- Import gym_super_mario_bros before calling gym.make(), because the envs are
  registered at import time.
- The wrappers here follow Gymnasium-style observation wrappers.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque, Tuple, Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import GrayScaleObservation, ResizeObservation

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


@dataclass
class MarioEnvConfig:
    env_id: str = "SuperMarioBros-v0"
    render_mode: str | None = None
    skip_frames: int = 4
    stack_frames: int = 4
    resize_shape: int = 84
    movement: list[list[str]] | tuple[tuple[str, ...], ...] = tuple(map(tuple, SIMPLE_MOVEMENT))
    progress_reward_scale: float = 0.02
    death_penalty: float = -15.0
    flag_bonus: float = 50.0
    idle_penalty: float = -0.01


class SkipFrame(gym.Wrapper):
    """Repeat the same action for `skip` frames and accumulate reward."""

    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        self.skip = max(1, int(skip))

    def step(self, action: Any):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        obs = None

        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info


class ChannelFirst(gym.ObservationWrapper):
    """Convert observations from HWC/HW to CHW for PyTorch-friendly input."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        old_space = env.observation_space
        assert isinstance(old_space, spaces.Box)

        if len(old_space.shape) == 2:
            h, w = old_space.shape
            new_shape = (1, h, w)
        elif len(old_space.shape) == 3:
            h, w, c = old_space.shape
            new_shape = (c, h, w)
        else:
            raise ValueError(f"Unsupported observation shape: {old_space.shape}")

        self.observation_space = spaces.Box(
            low=old_space.low.min(),
            high=old_space.high.max(),
            shape=new_shape,
            dtype=old_space.dtype,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        if observation.ndim == 2:
            return np.expand_dims(observation, axis=0)
        return np.transpose(observation, (2, 0, 1))


class FrameStack(gym.Wrapper):
    """Stack the last N frames along the channel dimension."""

    def __init__(self, env: gym.Env, num_stack: int = 4):
        super().__init__(env)
        self.num_stack = max(1, int(num_stack))
        self.frames: Deque[np.ndarray] = deque(maxlen=self.num_stack)

        obs_space = env.observation_space
        assert isinstance(obs_space, spaces.Box)
        assert len(obs_space.shape) == 3, "Expected CHW observation before stacking"
        c, h, w = obs_space.shape

        self.observation_space = spaces.Box(
            low=obs_space.low.min(),
            high=obs_space.high.max(),
            shape=(c * self.num_stack, h, w),
            dtype=obs_space.dtype,
        )

    def _get_observation(self) -> np.ndarray:
        return np.concatenate(list(self.frames), axis=0)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames.clear()
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_observation(), info

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_observation(), reward, terminated, truncated, info


class MarioRewardShaping(gym.Wrapper):
    """Shape reward using forward progress and simple episode events."""

    def __init__(
        self,
        env: gym.Env,
        progress_reward_scale: float = 0.02,
        death_penalty: float = -15.0,
        flag_bonus: float = 50.0,
        idle_penalty: float = -0.01,
    ) -> None:
        super().__init__(env)
        self.progress_reward_scale = progress_reward_scale
        self.death_penalty = death_penalty
        self.flag_bonus = flag_bonus
        self.idle_penalty = idle_penalty
        self.prev_x_pos = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_x_pos = 0
        return obs, info

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)

        x_pos = int(info.get("x_pos", self.prev_x_pos))
        progress = x_pos - self.prev_x_pos
        self.prev_x_pos = x_pos

        shaped_reward = float(reward)
        shaped_reward += progress * self.progress_reward_scale

        if progress <= 0:
            shaped_reward += self.idle_penalty

        if terminated:
            if info.get("flag_get", False):
                shaped_reward += self.flag_bonus
            else:
                shaped_reward += self.death_penalty

        return obs, shaped_reward, terminated, truncated, info


def make_mario_env(config: MarioEnvConfig | None = None) -> gym.Env:
    """Create a wrapped Mario environment.

    Returns a Gymnasium environment with discrete actions and image observations
    shaped as (stack, height, width), e.g. (4, 84, 84).
    """
    config = config or MarioEnvConfig()

    env = gym_super_mario_bros.make(config.env_id, render_mode=config.render_mode)
    env = JoypadSpace(env, [list(a) for a in config.movement])
    env = SkipFrame(env, skip=config.skip_frames)
    env = MarioRewardShaping(
        env,
        progress_reward_scale=config.progress_reward_scale,
        death_penalty=config.death_penalty,
        flag_bonus=config.flag_bonus,
        idle_penalty=config.idle_penalty,
    )
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=config.resize_shape)
    env = ChannelFirst(env)

    if config.stack_frames > 1:
        env = FrameStack(env, num_stack=config.stack_frames)

    return env


if __name__ == "__main__":
    cfg = MarioEnvConfig(render_mode="human")
    env = make_mario_env(cfg)

    obs, info = env.reset()
    print("Observation shape:", obs.shape)
    print("Action space:", env.action_space)

    for step in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if step % 25 == 0:
            print(
                f"step={step} reward={reward:.3f} x_pos={info.get('x_pos')} "
                f"coins={info.get('coins')} time={info.get('time')}"
            )
        if terminated or truncated:
            obs, info = env.reset()

    env.close()
