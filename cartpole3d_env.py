"""
True 3D CartPole Gymnasium Environment
=======================================
The cart moves freely on a 2D frictionless surface (X and Z axes).
The pole can fall in any direction (two angular DOF: theta_x, theta_z).

State (8,):
    [0] x           - cart position X        (-x_threshold, x_threshold)
    [1] x_dot       - cart velocity X
    [2] z           - cart position Z        (-z_threshold, z_threshold)
    [3] z_dot       - cart velocity Z
    [4] theta_x     - pole tilt toward X     (-angle_threshold, angle_threshold)
    [5] theta_x_dot - pole angular vel X
    [6] theta_z     - pole tilt toward Z     (-angle_threshold, angle_threshold)
    [7] theta_z_dot - pole angular vel Z

Action:
    Discrete(4): 0=+X, 1=-X, 2=+Z, 3=-Z
    Continuous Box(2,): [Fx, Fz]

Install:
    pip install gymnasium numpy
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional
import json


class CartPole3DEnv(gym.Env):

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 50}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        use_discrete: bool = True,
        gravity: float = 9.8,
        masscart: float = 1.0,
        masspole: float = 0.1,
        length: float = 0.5,
        force_mag: float = 10.0,
        tau: float = 0.02,
        x_threshold: float = 4.8,
        z_threshold: float = 4.8,
        angle_threshold_deg: float = 24.0,
        max_episode_steps: int = 500,
    ):
        super().__init__()

        self.gravity          = gravity
        self.masscart         = masscart
        self.masspole         = masspole
        self.total_mass       = masscart + masspole
        self.length           = length
        self.polemass_length  = masspole * length
        self.force_mag        = force_mag
        self.tau              = tau
        self.x_threshold      = x_threshold
        self.z_threshold      = z_threshold
        self.angle_threshold  = np.radians(angle_threshold_deg)
        self.max_episode_steps = max_episode_steps
        self.use_discrete     = use_discrete
        self.render_mode      = render_mode

        self._step_count = 0
        self.state       = None

        # ── Action space ──────────────────────────────────────────────
        if use_discrete:
            # 4 directions: +X, -X, +Z, -Z
            self.action_space = spaces.Discrete(4)
        else:
            self.action_space = spaces.Box(
                low=np.float32(-force_mag),
                high=np.float32(force_mag),
                shape=(2,),   # [Fx, Fz]
                dtype=np.float32,
            )

        # ── Observation space ─────────────────────────────────────────
        inf = np.finfo(np.float32).max
        high = np.array([
            x_threshold * 2, inf,
            z_threshold * 2, inf,
            self.angle_threshold * 2, inf,
            self.angle_threshold * 2, inf,
        ], dtype=np.float32)

        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────
    # Physics
    # ──────────────────────────────────────────────────────────────────

    def _step_physics(self, fx: float, fz: float):
        x, x_dot, z, z_dot, th_x, th_x_dot, th_z, th_z_dot = self.state

        g   = self.gravity
        mp  = self.masspole
        mt  = self.total_mass
        l   = self.length
        mpl = self.polemass_length

        # ── X-axis: cart force Fx drives cart + couples into theta_x ──
        sin_x, cos_x = np.sin(th_x), np.cos(th_x)
        temp_x   = (fx + mpl * th_x_dot**2 * sin_x) / mt
        th_x_ddot = (g * sin_x - cos_x * temp_x) / (
            l * (4.0 / 3.0 - mp * cos_x**2 / mt)
        )
        x_ddot = temp_x - mpl * th_x_ddot * cos_x / mt

        # ── Z-axis: cart force Fz drives cart + couples into theta_z ──
        sin_z, cos_z = np.sin(th_z), np.cos(th_z)
        temp_z    = (fz + mpl * th_z_dot**2 * sin_z) / mt
        th_z_ddot = (g * sin_z - cos_z * temp_z) / (
            l * (4.0 / 3.0 - mp * cos_z**2 / mt)
        )
        z_ddot = temp_z - mpl * th_z_ddot * cos_z / mt

        # ── Semi-implicit Euler integration ───────────────────────────
        x_dot_new     = x_dot     + self.tau * x_ddot
        z_dot_new     = z_dot     + self.tau * z_ddot
        th_x_dot_new  = th_x_dot  + self.tau * th_x_ddot
        th_z_dot_new  = th_z_dot  + self.tau * th_z_ddot

        x_new    = x    + self.tau * x_dot_new
        z_new    = z    + self.tau * z_dot_new
        th_x_new = th_x + self.tau * th_x_dot_new
        th_z_new = th_z + self.tau * th_z_dot_new

        self.state = np.array([
            x_new, x_dot_new,
            z_new, z_dot_new,
            th_x_new, th_x_dot_new,
            th_z_new, th_z_dot_new,
        ], dtype=np.float64)

    # ──────────────────────────────────────────────────────────────────
    # Gymnasium API
    # ──────────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state       = self.np_random.uniform(low=-0.05, high=0.05, size=(8,))
        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        assert self.state is not None, "Call reset() before step()"

        if self.use_discrete:
            # 0=+X  1=-X  2=+Z  3=-Z
            fx = [self.force_mag, -self.force_mag, 0.0, 0.0][int(action)]
            fz = [0.0, 0.0, self.force_mag, -self.force_mag][int(action)]
        else:
            fx = float(np.clip(action[0], -self.force_mag, self.force_mag))
            fz = float(np.clip(action[1], -self.force_mag, self.force_mag))

        self._step_physics(fx, fz)
        self._step_count += 1

        obs        = self._get_obs()
        terminated = self._is_terminated()
        truncated  = self._step_count >= self.max_episode_steps

        # Shaped reward
        angle_penalty = self.state[4]**2 + self.state[6]**2
        cart_penalty  = (self.state[0] / self.x_threshold)**2 + \
                        (self.state[2] / self.z_threshold)**2
        reward = 1.0 - 0.1 * angle_penalty - 0.05 * cart_penalty

        info = {
            "step":        self._step_count,
            "x":           float(self.state[0]),
            "z":           float(self.state[2]),
            "theta_x_deg": float(np.degrees(self.state[4])),
            "theta_z_deg": float(np.degrees(self.state[6])),
        }
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        return self.state.astype(np.float32)

    def _is_terminated(self):
        x, _, z, _, th_x, _, th_z, _ = self.state
        return bool(
            abs(x)   > self.x_threshold
            or abs(z)   > self.z_threshold
            or abs(th_x) > self.angle_threshold
            or abs(th_z) > self.angle_threshold
        )

    def render(self):
        if self.render_mode in ("human", "ansi"):
            x, _, z, _, th_x, _, th_z, _ = self.state
            print(
                f"step={self._step_count:4d} | "
                f"x={x:+6.3f}  z={z:+6.3f} | "
                f"θx={np.degrees(th_x):+6.1f}°  "
                f"θz={np.degrees(th_z):+6.1f}°"
            )

    def export_trajectory(self, frames: list, path: str = "trajectory.json"):
        with open(path, "w") as f:
            json.dump({
                "metadata": {
                    "length": self.length,
                    "x_threshold": self.x_threshold,
                    "z_threshold": self.z_threshold,
                    "angle_threshold_rad": float(self.angle_threshold),
                    "tau": self.tau,
                },
                "frames": frames,
            }, f, indent=2)
        print(f"Trajectory saved → {path}  ({len(frames)} frames)")

    def close(self):
        pass


# ──────────────────────────────────────────────────────────────────────
# Smoke-test
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    env = CartPole3DEnv(render_mode="human", use_discrete=True)
    obs, _ = env.reset(seed=42)
    print("Obs space :", env.observation_space)
    print("Act space :", env.action_space)
    print(f"Init obs  : {obs}\n")

    total_reward = 0.0
    for _ in range(300):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        total_reward += reward
        if terminated or truncated:
            break

    print(f"\nDone at step {info['step']} | reward = {total_reward:.2f}")
    env.close()