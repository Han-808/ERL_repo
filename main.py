"""
Entry point for ERL experiments.

Usage:
  python main.py                                        # both envs, 20 episodes
  python main.py --env frozen_lake
  python main.py --env sokoban --episodes 10
  python main.py --env both --model qwen3-14b --server http://192.168.1.5:30000/v1
"""

import argparse
import json

from environments.frozen_lake import FrozenLake
from environments.sokoban import Sokoban
from erl_pipeline import ERLPipeline


# ── table rendering ────────────────────────────────────────────────────────────

def _print_table(logs: list):
    """Print a fixed-width per-episode statistics table from the episode logs."""
    # Column widths (content only, borders added separately)
    W = {"ep": 7, "r1": 9, "r2": 9, "gated": 6, "mem": 14}

    def row(*cells, widths):
        return "│" + "│".join(
            f" {str(c).center(w)} " for c, w in zip(cells, widths.values())
        ) + "│"

    def divider(left, mid, right, fill="─"):
        segs = [fill * (w + 2) for w in W.values()]
        return left + mid.join(segs) + right

    header = row("Episode", "Reward 1", "Reward 2", "Gated", "Memory Size", widths=W)

    print(divider("┌", "┬", "┐"))
    print(header)
    print(divider("├", "┼", "┤"))
    for lg in logs:
        print(row(
            lg["episode"],
            lg["reward1"],
            lg["reward2"],
            "Yes" if lg["gated"] else "No",
            lg["memory_size"],
            widths=W,
        ))
    print(divider("└", "┴", "┘"))


# ── experiment runner ──────────────────────────────────────────────────────────

def run_experiment(env_name: str, n_episodes: int, model: str, server: str):
    if env_name == "frozen_lake":
        env = FrozenLake()
    elif env_name == "sokoban":
        env = Sokoban()
    else:
        raise ValueError(f"Unknown environment: {env_name!r}")

    pipeline = ERLPipeline(env, model=model, server_url=server)
    results  = pipeline.run(n_episodes)

    _print_table(results["logs"])

    output_path = f"results_{env_name}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved to {output_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ERL experiments.")
    parser.add_argument(
        "--env",
        choices=["frozen_lake", "sokoban", "both"],
        default="both",
        help="Which environment to run (default: both)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of episodes per environment (default: 20)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-8b",
        help="LM model name (default: qwen3-8b)",
    )
    parser.add_argument(
        "--server",
        type=str,
        default="http://LOCAL_SERVER/v1",
        help="SGLang/OpenAI-compatible server base URL",
    )
    args = parser.parse_args()

    envs = ["frozen_lake", "sokoban"] if args.env == "both" else [args.env]
    for env_name in envs:
        run_experiment(env_name, args.episodes, args.model, args.server)
