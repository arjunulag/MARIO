"""
Live training dashboard for the CartPole4D DQN agent.

Tails the CSV log written by train_cartpole4d.py and live-plots
reward, loss, epsilon, and episode length. Run it in a separate
terminal while training:

    python train_cartpole4d.py            # terminal 1
    python dashboard.py                   # terminal 2

Also works after training to view a finished run.

    python dashboard.py --log training_log.csv --refresh 2
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_log(path: Path) -> dict:
    cols = {"episode": [], "reward": [], "avg100": [],
            "steps": [], "epsilon": [], "loss": []}
    if not path.exists():
        return cols
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                cols["episode"].append(int(row["episode"]))
                cols["reward"].append(float(row["reward"]))
                cols["avg100"].append(float(row["avg100"]))
                cols["steps"].append(int(row["steps"]))
                cols["epsilon"].append(float(row["epsilon"]))
                cols["loss"].append(float(row["loss"]))
            except (ValueError, KeyError):
                # Skip partially-written or malformed rows.
                continue
    return cols


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log",     type=str,   default="training_log.csv",
                    help="Path to the CSV log produced by the trainer.")
    ap.add_argument("--refresh", type=float, default=2.0,
                    help="Seconds between refreshes.")
    args = ap.parse_args()

    path = Path(args.log)

    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    (ax_r, ax_l), (ax_e, ax_s) = axes

    print(f"Watching {path}  (close the window or Ctrl-C to exit)")
    try:
        while plt.fignum_exists(fig.number):
            cols = read_log(path)
            ep = cols["episode"]

            for ax in axes.flat:
                ax.cla()
                ax.grid(True, alpha=0.3)

            if ep:
                ax_r.plot(ep, cols["reward"], alpha=0.3, label="Episode")
                ax_r.plot(ep, cols["avg100"], linewidth=2, label="Avg100")
                ax_r.set_title(f"Reward  (avg100={cols['avg100'][-1]:.1f})")
                ax_r.set_xlabel("Episode"); ax_r.set_ylabel("Reward")
                ax_r.legend(loc="lower right")

                ax_l.plot(ep, cols["loss"], color="tab:red")
                ax_l.set_title(f"Avg loss  (latest={cols['loss'][-1]:.4f})")
                ax_l.set_xlabel("Episode"); ax_l.set_ylabel("Loss")
                if min(cols["loss"]) > 0:
                    ax_l.set_yscale("log")

                ax_e.plot(ep, cols["epsilon"], color="tab:green")
                ax_e.set_title(f"Epsilon  (latest={cols['epsilon'][-1]:.4f})")
                ax_e.set_xlabel("Episode"); ax_e.set_ylabel("ε")

                ax_s.plot(ep, cols["steps"], color="tab:purple", alpha=0.7)
                ax_s.set_title(f"Episode length  (latest={cols['steps'][-1]})")
                ax_s.set_xlabel("Episode"); ax_s.set_ylabel("Steps")

                fig.suptitle(
                    f"CartPole4D dashboard — {path.name} — {len(ep)} episodes"
                )
            else:
                fig.suptitle(f"Waiting for data at {path} ...")

            fig.tight_layout()
            plt.pause(args.refresh)
    except KeyboardInterrupt:
        pass

    plt.ioff()


if __name__ == "__main__":
    main()
