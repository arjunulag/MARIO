import pickle
import numpy as np
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from CNN_network import build_weights
from dqn_agent import MarioCNNTransformerDQNAgent
from preprocessFrames import build_initial_stack, preprocess_frame
from transformer import Transformer, TransformerConfig


CHECKPOINT_PATH = "checkpoints/best_mario.pkl"


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
)

with open(CHECKPOINT_PATH, "rb") as f:
    checkpoint = pickle.load(f)

weights = checkpoint["weights"]

for k, saved in zip(agent.kernels, weights["kernels"]):
    k.data = saved

agent.W1.data = weights["W1"]
agent.b1.data = weights["b1"]
agent.W2.data = weights["W2"]
agent.b2.data = weights["b2"]

agent.epsilon = 0.0

print("Loaded best checkpoint")
print("Episode:", checkpoint.get("episode"))
print("Best x:", checkpoint.get("best_x"))
print("Reward:", checkpoint.get("total_reward"))
print("Shaped reward:", checkpoint.get("shaped_total_reward"))

stack, state = build_initial_stack(env)

done = False

while not done:
    env.render()

    action = agent.select_action(state)

    result = env.step(action)

    if len(result) == 5:
        frame, reward, terminated, truncated, info = result
        done = terminated or truncated
    else:
        frame, reward, done, info = result

    frame = preprocess_frame(frame)
    stack.append(frame)
    state = np.stack(list(stack), axis=0)

env.close()