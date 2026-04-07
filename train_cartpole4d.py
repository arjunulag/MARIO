"""
Train a DQN agent on the 4D CartPole — entirely from scratch.
==============================================================
Uses:
  - CartPole4DEnv       (cartpole4d_env.py)     12-dim state, 6 actions
  - DQNAgent            (dqn_agent.py)           numpy MLP + backprop + Adam
  - CartPole4DRenderer  (cartpole3d_renderer.py) dual-view: 3D + W-axis panel

Run:
    python train_cartpole4d.py                   # headless training
    python train_cartpole4d.py --render           # render every 50 episodes
    python train_cartpole4d.py --demo             # demo with saved weights
"""

import argparse
import numpy as np
from cartpole4d_env import CartPole4DEnv
from dqn_agent import DQNAgent


def train(args):
    env   = CartPole4DEnv(use_discrete=True)
    agent = DQNAgent(
        state_dim=12,
        action_dim=6,
        hidden=[128, 128],
        lr=args.lr,                      # reduced from 0.001 to 0.0001
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.9995,
        buffer_size=100_000,             # increased from 50_000 to 100_000
        batch_size=64,
        target_sync_every=1000,          # increased from 500 to 1000
    )

    renderer = None
    if args.render:
        from cartpole3d_renderer import CartPole4DRenderer
        renderer = CartPole4DRenderer(env)

    best_reward   = -float("inf")
    reward_history = []

    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        ep_reward = 0.0
        losses    = []

        show = (renderer is not None and ep % args.render_every == 0)
        if show:
            renderer.reset_stats()

        done = False
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store(obs, action, reward, next_obs, float(done))
            loss = agent.train()
            if loss is not None:
                losses.append(loss)

            obs        = next_obs
            ep_reward += reward

            if show:
                if not renderer.render(reward=reward):
                    if renderer:
                        renderer.close()
                    return

        reward_history.append(ep_reward)
        avg_100 = np.mean(reward_history[-100:])

        if ep_reward > best_reward:
            best_reward = ep_reward
            agent.save("best_cartpole4d.npy")

        avg_loss = np.mean(losses) if losses else 0.0
        print(
            f"Ep {ep:5d}/{args.episodes} | "
            f"R {ep_reward:7.1f} | Avg100 {avg_100:7.1f} | "
            f"Steps {info['step']:3d} | "
            f"ε {agent.epsilon:.4f} | "
            f"Loss {avg_loss:.6f}"
        )

        if avg_100 >= args.solve_threshold and ep >= 100:
            print(f"\nSolved at episode {ep}  (avg100 = {avg_100:.1f})")
            agent.save("solved_cartpole4d.npy")
            break

    if renderer:
        renderer.close()

    print(f"\nBest episode reward: {best_reward:.1f}")
    print("Weights saved to best_cartpole4d.npy")


def demo(args):
    from cartpole3d_renderer import CartPole4DRenderer

    env   = CartPole4DEnv(use_discrete=True)
    agent = DQNAgent(state_dim=12, action_dim=6, hidden=[128, 128])
    agent.load(args.weights)
    agent.epsilon = 0.0

    renderer = CartPole4DRenderer(env)

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
                renderer.close()
                return

        print(f"Demo episode {ep} | Reward {ep_reward:.1f}")

    renderer.close()


def main():
    parser = argparse.ArgumentParser(description="Train / demo DQN on 4D CartPole")

    sub = parser.add_subparsers(dest="mode")
    sub.required = False

    # -- train ---------------------------------------------------------
    t = sub.add_parser("train", help="Train the agent")
    t.add_argument("--episodes",          type=int,   default=2000)
    t.add_argument("--lr",                type=float, default=0.0001)   # reduced from 0.001
    t.add_argument("--render",            action="store_true")
    t.add_argument("--render-every",      type=int,   default=50)
    t.add_argument("--solve-threshold",   type=float, default=400.0)

    # -- demo ----------------------------------------------------------
    d = sub.add_parser("demo", help="Run a trained agent with the renderer")
    d.add_argument("--weights",           type=str,   default="best_cartpole4d.npy")
    d.add_argument("--demo-episodes",     type=int,   default=10)

    args = parser.parse_args()

    if args.mode == "demo":
        demo(args)
    else:
        if args.mode is None:
            args.episodes        = 80000
            args.lr              = 0.0001   # reduced from 0.001
            args.render          = False
            args.render_every    = 50
            args.solve_threshold = 400.0
        train(args)


if __name__ == "__main__":
    main()