"""
True 4D CartPole Gymnasium Environment
=======================================
The cart moves freely on a 3D hyper-surface (X, Z, W axes).
The pole can fall in any of three angular DOFs: theta_x, theta_z, theta_w.

State (12,):
    [0]  x           - cart position X        (-x_threshold, x_threshold)
    [1]  x_dot       - cart velocity X
    [2]  z           - cart position Z        (-z_threshold, z_threshold)
    [3]  z_dot       - cart velocity Z
    [4]  w           - cart position W        (-w_threshold, w_threshold)
    [5]  w_dot       - cart velocity W
    [6]  theta_x     - pole tilt toward X     (-angle_threshold, angle_threshold)
    [7]  theta_x_dot - pole angular vel X
    [8]  theta_z     - pole tilt toward Z     (-angle_threshold, angle_threshold)
    [9]  theta_z_dot - pole angular vel Z
    [10] theta_w     - pole tilt toward W     (-angle_threshold, angle_threshold)
    [11] theta_w_dot - pole angular vel W

Action:
    Discrete(6): 0=+X, 1=-X, 2=+Z, 3=-Z, 4=+W, 5=-W
    Continuous Box(3,): [Fx, Fz, Fw]

Install:
    pip install gymnasium numpy
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional
import json


class CartPole4DEnv(gym.Env):

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
        w_threshold: float = 4.8,
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
        self.w_threshold      = w_threshold
        self.angle_threshold  = np.radians(angle_threshold_deg)
        self.max_episode_steps = max_episode_steps
        self.use_discrete     = use_discrete
        self.render_mode      = render_mode

        self._step_count = 0
        self.state       = None

        if use_discrete:
            self.action_space = spaces.Discrete(6)
        else:
            self.action_space = spaces.Box(
                low=np.float32(-force_mag),
                high=np.float32(force_mag),
                shape=(3,),
                dtype=np.float32,
            )

        inf = np.finfo(np.float32).max
        high = np.array([
            x_threshold * 2, inf,
            z_threshold * 2, inf,
            w_threshold * 2, inf,
            self.angle_threshold * 2, inf,
            self.angle_threshold * 2, inf,
            self.angle_threshold * 2, inf,
        ], dtype=np.float32)

        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def _step_axis(self, force, theta, theta_dot):
        """Shared 2D inverted-pendulum physics for one (cart-axis, pole-angle) pair."""
        g   = self.gravity
        mp  = self.masspole
        mt  = self.total_mass
        l   = self.length
        mpl = self.polemass_length

        sin_t, cos_t = np.sin(theta), np.cos(theta)
        temp     = (force + mpl * theta_dot**2 * sin_t) / mt
        theta_dd = (g * sin_t - cos_t * temp) / (l * (4.0/3.0 - mp * cos_t**2 / mt))
        pos_dd   = temp - mpl * theta_dd * cos_t / mt

        return pos_dd, theta_dd

    def _step_physics(self, fx: float, fz: float, fw: float):
        (x, x_dot, z, z_dot, w, w_dot,
         th_x, th_x_dot, th_z, th_z_dot, th_w, th_w_dot) = self.state

        x_dd, th_x_dd = self._step_axis(fx, th_x, th_x_dot)
        z_dd, th_z_dd = self._step_axis(fz, th_z, th_z_dot)
        w_dd, th_w_dd = self._step_axis(fw, th_w, th_w_dot)

        tau = self.tau

        x_dot_n     = x_dot     + tau * x_dd
        z_dot_n     = z_dot     + tau * z_dd
        w_dot_n     = w_dot     + tau * w_dd
        th_x_dot_n  = th_x_dot  + tau * th_x_dd
        th_z_dot_n  = th_z_dot  + tau * th_z_dd
        th_w_dot_n  = th_w_dot  + tau * th_w_dd

        self.state = np.array([
            x    + tau * x_dot_n,     x_dot_n,
            z    + tau * z_dot_n,     z_dot_n,
            w    + tau * w_dot_n,     w_dot_n,
            th_x + tau * th_x_dot_n,  th_x_dot_n,
            th_z + tau * th_z_dot_n,  th_z_dot_n,
            th_w + tau * th_w_dot_n,  th_w_dot_n,
        ], dtype=np.float64)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state       = self.np_random.uniform(low=-0.05, high=0.05, size=(12,))
        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        assert self.state is not None, "Call reset() before step()"

        fm = self.force_mag
        if self.use_discrete:
            forces = [
                ( fm, 0., 0.), (-fm, 0., 0.),
                (0.,  fm, 0.), (0., -fm, 0.),
                (0., 0.,  fm), (0., 0., -fm),
            ]
            fx, fz, fw = forces[int(action)]
        else:
            fx = float(np.clip(action[0], -fm, fm))
            fz = float(np.clip(action[1], -fm, fm))
            fw = float(np.clip(action[2], -fm, fm))

        self._step_physics(fx, fz, fw)
        self._step_count += 1

        obs        = self._get_obs()
        terminated = self._is_terminated()
        truncated  = self._step_count >= self.max_episode_steps

        angle_penalty = self.state[6]**2 + self.state[8]**2 + self.state[10]**2
        cart_penalty  = ((self.state[0] / self.x_threshold)**2 +
                         (self.state[2] / self.z_threshold)**2 +
                         (self.state[4] / self.w_threshold)**2)
        reward = 1.0 - 0.1 * angle_penalty - 0.05 * cart_penalty

        info = {
            "step":        self._step_count,
            "x":           float(self.state[0]),
            "z":           float(self.state[2]),
            "w":           float(self.state[4]),
            "theta_x_deg": float(np.degrees(self.state[6])),
            "theta_z_deg": float(np.degrees(self.state[8])),
            "theta_w_deg": float(np.degrees(self.state[10])),
        }
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        return self.state.astype(np.float32)

    def _is_terminated(self):
        x, _, z, _, w, _, th_x, _, th_z, _, th_w, _ = self.state
        return bool(
            abs(x)    > self.x_threshold
            or abs(z)    > self.z_threshold
            or abs(w)    > self.w_threshold
            or abs(th_x) > self.angle_threshold
            or abs(th_z) > self.angle_threshold
            or abs(th_w) > self.angle_threshold
        )

    def render(self):
        if self.render_mode in ("human", "ansi"):
            s = self.state
            print(
                f"step={self._step_count:4d} | "
                f"x={s[0]:+6.3f}  z={s[2]:+6.3f}  w={s[4]:+6.3f} | "
                f"θx={np.degrees(s[6]):+6.1f}°  "
                f"θz={np.degrees(s[8]):+6.1f}°  "
                f"θw={np.degrees(s[10]):+6.1f}°"
            )

    def close(self):
        pass


if __name__ == "__main__":
    env = CartPole4DEnv(render_mode="human", use_discrete=True)
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
