import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from preprocess import build_initial_stack, preprocess_frame
from network import build_weights, forward

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

n_actions = env.action_space.n
kernels, W1, b1, W2, b2 = build_weights(n_actions)
stack, state = build_initial_stack(env)