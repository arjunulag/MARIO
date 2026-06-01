import os
import pickle
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
BATCH_SIZE = 8
GAMMA = 0.99
LR = 0.0001
REPLAY_SIZE = 50000
LEARN_START = 1000
TARGET_SYNC_EVERY = 2000
TRAIN_EVERY = 20

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.9995

PROGRESS_REWARD_SCALE = 0.05
IDLE_PENALTY = -0.01

MAX_STUCK_STEPS = 180
MAX_STEPS_PER_EPISODE = 5000

SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)


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
best_score = float("-inf")

def tensor_to_data(x):
        return x.data if hasattr(x, "data") else x


def save_agent_weights(agent):
    return {
        "kernels": [tensor_to_data(k) for k in agent.kernels],
        "W1": tensor_to_data(agent.W1),
        "b1": tensor_to_data(agent.b1),
        "W2": tensor_to_data(agent.W2),
        "b2": tensor_to_data(agent.b2),

        "target_kernels": [
            tensor_to_data(k)
            for k in getattr(agent, "target_kernels", [])
        ],
        "target_W1": tensor_to_data(getattr(agent, "target_W1", None)),
        "target_b1": tensor_to_data(getattr(agent, "target_b1", None)),
        "target_W2": tensor_to_data(getattr(agent, "target_W2", None)),
        "target_b2": tensor_to_data(getattr(agent, "target_b2", None)),

        "epsilon": agent.epsilon,
    }

for episode in range(EPISODES):
    stack, state = build_initial_stack(env)

    done = False
    total_reward = 0.0
    shaped_total_reward = 0.0
    losses = []
    step = 0

    previous_x_pos = None
    best_x = 0
    stuck_steps = 0

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

        if x_pos is not None:
            if x_pos > best_x:
                best_x = x_pos
                stuck_steps = 0
            else:
                stuck_steps += 1

        if stuck_steps >= MAX_STUCK_STEPS:
            done = True

        if step >= MAX_STEPS_PER_EPISODE:
            done = True

        agent.store(state, action, shaped_reward, next_state, done)

        state = next_state
        total_reward += reward
        shaped_total_reward += shaped_reward
        step += 1
        global_step += 1

        if len(agent.buffer) >= LEARN_START and global_step % TRAIN_EVERY == 0:
            loss = agent.train()
            if loss is not None:
                losses.append(loss)

        if step == 1 or step % 250 == 0 or done:
            avg_loss = np.mean(losses[-25:]) if losses else 0.0

            print(
                f"Episode {episode} | step {step} | raw {reward:.2f} | "
                f"shaped {shaped_reward:.2f} | total {total_reward:.1f} | "
                f"shaped_total {shaped_total_reward:.1f} | x {x_pos} | "
                f"best_x {best_x} | stuck {stuck_steps} | "
                f"dx {progress:.1f} | action {action} | "
                f"eps {agent.epsilon:.3f} | replay {len(agent.buffer)} | "
                f"loss {avg_loss:.4f}",
                flush=True,
            )

    avg_loss = np.mean(losses) if losses else 0.0

    print(
        f"Episode {episode} done - reward: {total_reward:.1f} | "
        f"shaped_reward: {shaped_total_reward:.1f} | best_x: {best_x} | "
        f"steps: {step} | eps: {agent.epsilon:.3f} | loss: {avg_loss:.4f}",
        flush=True,
    )

    checkpoint = {
        "episode": episode,
        "global_step": global_step,
        "epsilon": agent.epsilon,
        "total_reward": total_reward,
        "shaped_total_reward": shaped_total_reward,
        "best_x": best_x,
        "loss": avg_loss,
        "weights": save_agent_weights(agent),
    }

    os.makedirs(SAVE_DIR, exist_ok=True)
    latest_path = os.path.join(SAVE_DIR, "latest_mario.pkl")

    
    with open(latest_path, "wb") as f:
        pickle.dump(checkpoint, f)

    performance = best_x

    if performance > best_score:
        best_score = performance
        checkpoint["best_score"] = best_score

        best_path = os.path.join(SAVE_DIR, "best_mario.pkl")

        with open(best_path, "wb") as f:
            pickle.dump(checkpoint, f)

        print(
            f"Saved new best model to {best_path} | best_x: {best_x}",
            flush=True,
        )