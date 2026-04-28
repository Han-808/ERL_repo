"""
Entry point for the ACE Notebook pipeline.

Usage:
    python main_ace_notebook.py --env frozen_lake --episodes 50
    python main_ace_notebook.py --env both --model qwen3-14b --server http://192.168.1.5:30000/v1
"""

import argparse
import json

from environments.frozen_lake import FrozenLake
from environments.sokoban import Sokoban
from ace_notebook_pipeline import ACENotebookPipeline


def _print_table(logs: list) -> None:
    """Print per-episode table: Episode | Reward | Operations Applied | Notebook Lines."""
    W = {"ep": 9, "rew": 8, "ops": 19, "nb": 16}

    def row(*cells, widths):
        return "│" + "│".join(
            f" {str(c).center(w)} " for c, w in zip(cells, widths.values())
        ) + "│"

    def divider(left, mid, right, fill="─"):
        segs = [fill * (w + 2) for w in W.values()]
        return left + mid.join(segs) + right

    print(divider("┌", "┬", "┐"))
    print(row("Episode", "Reward", "Operations Applied", "Notebook Lines", widths=W))
    print(divider("├", "┼", "┤"))
    for lg in logs:
        print(row(
            lg["episode"],
            lg["reward"],
            len(lg.get("operations", [])),
            lg.get("notebook_lines", ""),
            widths=W,
        ))
    print(divider("└", "┴", "┘"))


def run_experiment(
    env_name: str,
    n_episodes: int,
    model: str,
    server: str,
    disable_thinking: bool = False,
) -> None:
    if env_name == "frozen_lake":
        env = FrozenLake()
    elif env_name == "sokoban":
        env = Sokoban()
    else:
        raise ValueError(f"Unknown environment: {env_name!r}")

    pipeline = ACENotebookPipeline(
        env,
        model=model,
        server_url=server,
        disable_thinking=disable_thinking,
    )
    results = pipeline.run(n_episodes=n_episodes)

    _print_table(results["logs"])

    output_path = f"results_ace_notebook_{env_name}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ACE Notebook pipeline.")
    parser.add_argument(
        "--env",
        choices=["frozen_lake", "sokoban", "both"],
        default="both",
        help="Which environment to run (default: both).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help=(
            "Number of random environment instances K (default: 50). "
            "Each instance is a different randomly generated map/puzzle. "
            "The notebook accumulates experience across all K instances (online setting)."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-8b",
        help="LM model name served by the backend (default: qwen3-8b).",
    )
    parser.add_argument(
        "--server",
        type=str,
        default="http://LOCAL_SERVER/v1",
        help=(
            "Base URL of an OpenAI-API-compatible inference server "
            "(e.g. SGLang); must expose /v1/chat/completions."
        ),
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        default=False,
        help="Disable thinking mode for Qwen3/Qwen3.5 models (default: thinking enabled)",
    )
    args = parser.parse_args()

    envs = ["frozen_lake", "sokoban"] if args.env == "both" else [args.env]
    for env_name in envs:
        run_experiment(
            env_name,
            args.episodes,
            args.model,
            args.server,
            disable_thinking=args.disable_thinking,
        )


if __name__ == "__main__":
    main()
