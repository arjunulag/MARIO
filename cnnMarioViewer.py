import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import numpy as np

from CNN_network import build_weights
from dqn_agent import MarioCNNTransformerDQNAgent
from mario_training_utils import shape_mario_reward
from preprocessFrames import build_initial_stack, preprocess_frame
from transformer import Transformer, TransformerConfig


EPISODES = 500
BATCH_SIZE = 16
GAMMA = 0.99
LR = 0.0001
REPLAY_SIZE = 50000
LEARN_START = 1000
TARGET_SYNC_EVERY = 1000
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.99995
PROGRESS_REWARD_SCALE = 0.05
IDLE_PENALTY = -0.01


env = gym_super_mario_bros.make("SuperMarioBros-v3")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

kernels, W1, b1, W2, b2 = build_weights(d_model=64)
cfg = TransformerConfig(vocab_size=env.action_space.n, d_model=64)
transformer = Transformer(cfg)
agent = MarioCNNTransformerDQNAgent(
    kernels,
    W1,
    b1,
    W2,
    b2,
    transformer,
    action_dim=env.action_space.n,
    lr=LR,
    gamma=GAMMA,
    epsilon_start=EPSILON_START,
    epsilon_end=EPSILON_END,
    epsilon_decay=EPSILON_DECAY,
    buffer_size=REPLAY_SIZE,
    batch_size=BATCH_SIZE,
    target_sync_every=TARGET_SYNC_EVERY,
)
global_step = 0

for episode in range(EPISODES):
    stack, state = build_initial_stack(env)
    done = False
    total_reward = 0.0
    shaped_total_reward = 0.0
    losses = []
    step = 0
    previous_x_pos = None

    while not done:
        action = agent.select_action(state)

        result = env.step(action)
        if len(result) == 5:
            next_frame, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            next_frame, reward, done, info = result

        next_frame = preprocess_frame(next_frame)
        stack.append(next_frame)
        next_state = np.stack(list(stack), axis=0)

        x_pos = info.get("x_pos") if isinstance(info, dict) else None
        shaped_reward, progress, previous_x_pos = shape_mario_reward(
            reward,
            x_pos,
            previous_x_pos,
            done,
            progress_reward_scale=PROGRESS_REWARD_SCALE,
            idle_penalty=IDLE_PENALTY,
        )

        agent.store(state, action, shaped_reward, next_state, done)
        state = next_state
        total_reward += reward
        shaped_total_reward += shaped_reward
        step += 1
        global_step += 1

        if len(agent.buffer) >= LEARN_START:
            loss = agent.train()
            if loss is not None:
                losses.append(loss)

        if step == 1 or step % 25 == 0 or done:
            x_pos = info.get("x_pos", "?") if isinstance(info, dict) else "?"
            avg_loss = np.mean(losses[-25:]) if losses else 0.0
            print(
                f"Episode {episode} | step {step} | raw {reward:.2f} | "
                f"shaped {shaped_reward:.2f} | total {total_reward:.1f} | "
                f"shaped_total {shaped_total_reward:.1f} | x {x_pos} | "
                f"dx {progress:.1f} | action {action} | "
                f"eps {agent.epsilon:.3f} | replay {len(agent.buffer)} | "
                f"loss {avg_loss:.4f}",
                flush=True,
            )

    avg_loss = np.mean(losses) if losses else 0.0
    print(
        f"Episode {episode} done - reward: {total_reward:.1f} | "
        f"shaped_reward: {shaped_total_reward:.1f} | steps: {step} | "
        f"eps: {agent.epsilon:.3f} | loss: {avg_loss:.4f}",
        flush=True,
    )
