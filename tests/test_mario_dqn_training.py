import tempfile
import unittest
from pathlib import Path
import sys
import types
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if "cv2" not in sys.modules:
    sys.modules["cv2"] = types.SimpleNamespace(
        COLOR_RGB2GRAY=0,
        INTER_AREA=0,
        cvtColor=lambda frame, code: frame.mean(axis=2).astype(np.uint8),
        resize=lambda frame, size, interpolation=None: np.zeros(
            (size[1], size[0]),
            dtype=np.float64,
        ),
    )

from mario_dqn_checkpoint import build_fresh_agent, load_agent, save_agent
from mario_levels import FIRST_FIVE_LEVELS, MarioLevel, episode_score, iter_first_five_levels
from train_mario_dqn import evaluate_greedy_agent, maybe_save_best, run_episode


class FakeReplayBuffer:
    def __len__(self):
        return 999


class FakeAgent:
    def __init__(self):
        self.buffer = FakeReplayBuffer()
        self.epsilon = 0.5
        self.epsilons_seen = []
        self.store_calls = 0
        self.train_calls = 0

    def select_action(self, state):
        self.epsilons_seen.append(self.epsilon)
        return 0

    def store(self, state, action, reward, next_state, done):
        self.store_calls += 1

    def train(self, updates=1):
        self.train_calls += 1
        return 0.25


class FakeSavingAgent(FakeAgent):
    def __init__(self):
        super().__init__()
        self.saved_path = None
        self.saved_meta = None

    def save(self, path, meta=None):
        self.saved_path = Path(path)
        self.saved_meta = meta
        self.saved_path.write_text("saved", encoding="utf-8")


class FakeMarioEnv:
    action_space = type("ActionSpace", (), {"n": 7})()

    def __init__(self, x_positions=(10, 20), flag_get=False):
        self.x_positions = list(x_positions)
        self.flag_get = flag_get
        self.step_idx = 0
        self.closed = False

    def reset(self):
        self.step_idx = 0
        return self._frame(), {}

    def step(self, action):
        x_pos = self.x_positions[min(self.step_idx, len(self.x_positions) - 1)]
        self.step_idx += 1
        done = self.step_idx >= len(self.x_positions)
        return self._frame(), 1.0, done, False, {"x_pos": x_pos, "flag_get": self.flag_get}

    def close(self):
        self.closed = True

    @staticmethod
    def _frame():
        return np.zeros((240, 256, 3), dtype=np.uint8)


class MarioDQNTrainingTests(unittest.TestCase):
    def test_first_five_levels_are_game_order(self):
        self.assertEqual(
            FIRST_FIVE_LEVELS,
            ((1, 1), (1, 2), (1, 3), (1, 4), (2, 1)),
        )
        keys = [level.key for level in iter_first_five_levels()]
        self.assertEqual(keys, ["1-1", "1-2", "1-3", "1-4", "2-1"])

    def test_level_env_ids(self):
        level = MarioLevel(1, 2, 3)
        self.assertEqual(level.env_id, "SuperMarioBros-1-2-v3")
        self.assertEqual(level.best_weights_name(), "best_1-2.npy")

    def test_episode_score_prioritizes_flag(self):
        no_flag = episode_score(x_pos=500, flag_get=False)
        with_flag = episode_score(x_pos=100, flag_get=True)
        self.assertGreater(with_flag, no_flag)

    def test_eval_episode_does_not_store_or_train(self):
        agent = FakeAgent()
        env = FakeMarioEnv()

        result = run_episode(
            env,
            agent,
            frame_skip=1,
            max_steps=10,
            learn=False,
            learn_start=0,
            train_every=1,
            grad_updates=1,
        )

        self.assertEqual(result["x_pos"], 20)
        self.assertEqual(agent.store_calls, 0)
        self.assertEqual(agent.train_calls, 0)

    def test_greedy_evaluation_restores_epsilon(self):
        agent = FakeAgent()
        envs = [FakeMarioEnv((10, 30)), FakeMarioEnv((10, 40))]

        with patch("train_mario_dqn.make_level_env", side_effect=envs):
            summary = evaluate_greedy_agent(
                agent,
                MarioLevel(1, 1),
                frame_skip=1,
                max_steps=10,
                episodes=2,
            )

        self.assertEqual(agent.epsilon, 0.5)
        self.assertEqual(agent.epsilons_seen, [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(summary["episodes"], 2)
        self.assertEqual(summary["best"]["x_pos"], 40)
        self.assertTrue(all(env.closed for env in envs))

    def test_best_checkpoint_uses_greedy_score(self):
        agent = FakeSavingAgent()
        level = MarioLevel(1, 1)
        training_result = {
            "steps": 5,
            "total_reward": 5.0,
            "shaped_total": 100.0,
            "x_pos": 100,
            "flag_get": False,
        }
        greedy_summary = {
            "episodes": 1,
            "scores": [60.0],
            "mean_score": 60.0,
            "best": {
                "score": 60.0,
                "steps": 4,
                "shaped_total": 50.0,
                "x_pos": 60,
                "flag_get": False,
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            weights_dir = Path(tmp)
            manifest_path = weights_dir / "manifest.json"
            manifest = {}

            with patch("train_mario_dqn.evaluate_greedy_agent", return_value=greedy_summary):
                saved = maybe_save_best(
                    agent,
                    level,
                    training_result,
                    3,
                    {"1-1": 50.0},
                    weights_dir,
                    manifest_path,
                    manifest,
                    frame_skip=1,
                    max_steps=10,
                    eval_episodes=1,
                )

            self.assertTrue(saved)
            self.assertEqual(agent.saved_path, weights_dir / "best_1-1.npy")
            self.assertEqual(agent.saved_meta["score"], 60.0)
            self.assertEqual(agent.saved_meta["score_type"], "greedy_eval")
            self.assertGreater(agent.saved_meta["training_score"], agent.saved_meta["score"])
            self.assertEqual(manifest["levels"]["1-1"]["score"], 60.0)

    def test_checkpoint_roundtrip(self):
        agent = build_fresh_agent(
            action_dim=7,
            fast_transformer=True,
            buffer_size=32,
            batch_size=4,
            epsilon_start=0.75,
            epsilon_end=0.12,
            epsilon_decay=0.91,
        )
        agent.epsilon = 0.44
        state = np.random.rand(4, 84, 84).astype(np.float32)
        before = agent.q_values(state).copy()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "roundtrip.npy"
            save_agent(agent, path, meta={"level": "1-1", "score": 42.0})
            resumed = load_agent(path)
            viewer = load_agent(path, epsilon=0.0)
            after = viewer.q_values(state)

        np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-5)
        self.assertEqual(resumed.epsilon, 0.44)
        self.assertEqual(resumed.epsilon_end, 0.12)
        self.assertEqual(resumed.epsilon_decay, 0.91)
        self.assertEqual(viewer.meta.get("level"), "1-1")
        self.assertEqual(viewer.epsilon, 0.0)
        self.assertEqual(viewer.epsilon_end, 0.0)
        self.assertEqual(viewer.epsilon_decay, 1.0)


if __name__ == "__main__":
    unittest.main()
