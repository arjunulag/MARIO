import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mario_dqn_checkpoint import build_fresh_agent, load_agent, save_agent
from mario_levels import FIRST_FIVE_LEVELS, MarioLevel, episode_score, iter_first_five_levels


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

    def test_checkpoint_roundtrip(self):
        agent = build_fresh_agent(action_dim=7, fast_transformer=True, buffer_size=32, batch_size=4)
        state = np.random.rand(4, 84, 84).astype(np.float32)
        before = agent.q_values(state).copy()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "roundtrip.npy"
            save_agent(agent, path, meta={"level": "1-1", "score": 42.0})
            loaded = load_agent(path, epsilon=0.0)
            after = loaded.q_values(state)

        np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-5)
        self.assertEqual(loaded.meta.get("level"), "1-1")
        self.assertEqual(loaded.epsilon, 0.0)


if __name__ == "__main__":
    unittest.main()
