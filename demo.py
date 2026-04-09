from cartpole4d_renderer import CartPole4DRenderer   # named import
import argparse
import numpy as np
import matplotlib.pyplot as plt

from cartpole4d_env import CartPole4DEnv
from dqn_agent import DQNAgent


def demo(args):
    env   = CartPole4DEnv(use_discrete=True)
    agent = DQNAgent(state_dim=12, action_dim=6, hidden=[128, 128])
    agent.load(args.weights)
    agent.epsilon = 0.0

    renderer = CartPole4DRenderer(env)
    user_closed = False

    for ep in range(1, args.demo_episodes + 1):
        obs, _ = env.reset()
        renderer.reset_stats()
        ep_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward

            if not renderer.render(reward=reward):
                user_closed = True
                break

        if user_closed:
            break


def main():
    parser = argparse.ArgumentParser(description="Train / demo DQN on 4D CartPole")

    sub = parser.add_subparsers(dest="mode")
    sub.required = False

    d = sub.add_parser("demo", help="Run a trained agent with the renderer")
    d.add_argument("--weights",        type=str, default="best_cartpole4d.npy")
    d.add_argument("--demo-episodes",  type=int, default=10)

    args = parser.parse_args()

    # If no subcommand given, inject demo defaults so demo() always has its args
    if args.mode != "demo":
        args.mode           = "demo"
        args.weights        = "best_cartpole4d.npy"
        args.demo_episodes  = 10

    demo(args)


if __name__ == "__main__":
    main()