import unittest
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CNN_network import forward
from Tensor import Tensor
from dqn_agent import MarioCNNTransformerDQNAgent, ReplayBuffer
from mario_training_utils import shape_mario_reward
from transformer import Transformer, TransformerConfig


def make_tiny_cnn(d_model=8):
    kernels = [
        Tensor(np.random.randn(2, 4, 3, 3) * 0.1),
        Tensor(np.random.randn(3, 2, 3, 3) * 0.1),
    ]
    flat_size = 3 * 8 * 8
    W1 = Tensor(np.random.randn(12, flat_size) * 0.1)
    b1 = Tensor(np.zeros(12))
    W2 = Tensor(np.random.randn(d_model, 12) * 0.1)
    b2 = Tensor(np.zeros(d_model))
    return kernels, W1, b1, W2, b2


def make_tiny_agent(action_dim=4, batch_size=4, target_sync_every=10):
    kernels, W1, b1, W2, b2 = make_tiny_cnn(d_model=8)
    cfg = TransformerConfig(
        vocab_size=action_dim,
        d_model=8,
        num_heads=2,
        num_layers=1,
        d_ff=16,
        max_seq_len=4,
    )
    transformer = Transformer(cfg)
    agent = MarioCNNTransformerDQNAgent(
        kernels,
        W1,
        b1,
        W2,
        b2,
        transformer,
        action_dim=action_dim,
        batch_size=batch_size,
        target_sync_every=target_sync_every,
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay=1.0,
        buffer_size=20,
    )
    return agent


class CNNTransformerDQNTests(unittest.TestCase):
    def test_cnn_forward_and_backward_produces_expected_embedding_shape(self):
        np.random.seed(0)
        kernels, W1, b1, W2, b2 = make_tiny_cnn(d_model=8)
        state = np.random.rand(4, 12, 12).astype(np.float32)

        embedding = forward(state, kernels, W1, b1, W2, b2)
        self.assertEqual(embedding.data.shape, (8,))

        embedding.backward(np.ones(8, dtype=np.float32))
        self.assertTrue(np.any(W2.grad != 0))
        self.assertTrue(np.any(kernels[0].grad != 0))

    def test_transformer_embedding_path_returns_q_values_and_input_gradient(self):
        np.random.seed(1)
        cfg = TransformerConfig(
            vocab_size=5,
            d_model=8,
            num_heads=2,
            num_layers=1,
            d_ff=16,
            max_seq_len=4,
        )
        transformer = Transformer(cfg)
        embedding = np.random.randn(1, 1, 8)

        logits, cache = transformer.forward_from_embedding_with_cache(embedding)
        self.assertEqual(logits.shape, (1, 1, 5))

        grad_logits = np.zeros_like(logits)
        grad_logits[0, 0, 2] = 1.0
        grad_embedding = transformer.input_grad(grad_logits, cache)
        self.assertEqual(grad_embedding.shape, embedding.shape)
        self.assertTrue(np.all(np.isfinite(grad_embedding)))

    def test_replay_buffer_preserves_transition_shapes(self):
        replay = ReplayBuffer(capacity=3)
        state = np.zeros((4, 12, 12))
        for i in range(4):
            replay.push(state + i, i, i * 0.5, state + i + 1, i == 3)

        self.assertEqual(len(replay), 3)
        states, actions, rewards, next_states, dones = replay.sample(2)
        self.assertEqual(states.shape, (2, 4, 12, 12))
        self.assertEqual(actions.shape, (2,))
        self.assertEqual(rewards.shape, (2,))
        self.assertEqual(next_states.shape, (2, 4, 12, 12))
        self.assertEqual(dones.shape, (2,))

    def test_mario_dqn_agent_selects_action_and_trains_cnn_transformer_q_head(self):
        np.random.seed(2)
        agent = make_tiny_agent(action_dim=4, batch_size=4)
        state = np.random.rand(4, 12, 12).astype(np.float32)
        next_state = np.random.rand(4, 12, 12).astype(np.float32)

        q = agent.q_values(state)
        self.assertEqual(q.shape, (4,))

        action = agent.select_action(state)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, 4)

        for i in range(4):
            agent.store(state, i % 4, float(i - 1), next_state, False)

        before_W2 = agent.W2.data.copy()
        before_token_emb = agent.transformer.token_emb.copy()
        loss = agent.train()

        self.assertIsNotNone(loss)
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(agent.train_steps, 1)
        self.assertTrue(np.any(agent.W2.data != before_W2))
        self.assertTrue(np.any(agent.transformer.token_emb != before_token_emb))

    def test_mario_dqn_target_network_syncs_on_schedule(self):
        np.random.seed(3)
        agent = make_tiny_agent(action_dim=4, batch_size=4, target_sync_every=1)
        state = np.random.rand(4, 12, 12).astype(np.float32)
        next_state = np.random.rand(4, 12, 12).astype(np.float32)

        for i in range(4):
            agent.store(state, i % 4, 1.0, next_state, False)

        agent.train()
        np.testing.assert_allclose(agent.W2.data, agent.target_W2.data)
        np.testing.assert_allclose(
            agent.transformer.token_emb, agent.target_transformer.token_emb
        )

    def test_reward_shaping_rewards_progress_and_penalizes_idle(self):
        shaped, progress, previous = shape_mario_reward(
            raw_reward=2.0,
            x_pos=526,
            previous_x_pos=524,
            done=False,
            progress_reward_scale=0.05,
            idle_penalty=-0.01,
        )
        self.assertEqual(progress, 2.0)
        self.assertEqual(previous, 526.0)
        self.assertAlmostEqual(shaped, 2.1)

        shaped, progress, previous = shape_mario_reward(
            raw_reward=0.0,
            x_pos=526,
            previous_x_pos=526,
            done=False,
            progress_reward_scale=0.05,
            idle_penalty=-0.01,
        )
        self.assertEqual(progress, 0.0)
        self.assertEqual(previous, 526.0)
        self.assertAlmostEqual(shaped, -0.01)


if __name__ == "__main__":
    unittest.main()
