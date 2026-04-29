"""
Unified entry point for every method in this repo.

Mirrors the structure of appworld-context-updater/run.py: a METHODS
registry maps a method name to a class (or factory lambda), and the
argparse CLI instantiates one method against one or more environments.

Usage examples:
    python run.py --method erl --env frozen_lake --episodes 20
    python run.py --method ace --env both --model qwen3-14b --server http://192.168.1.5:30000/v1
"""

import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import print_episode_table, results_path, write_results
from environments.frozen_lake import FrozenLake
from environments.sokoban import Sokoban
from methods.ace import ACEMethod
from methods.erl import ERLMethod
from methods.notebook_minimal import NotebookMinimalMethod


def _notebook_factory(initial_notebook):
    """Bind the initial_notebook variant into a callable matching METHODS."""
    def build(env, **kw):
        return NotebookMinimalMethod(
            env, initial_notebook=initial_notebook, **kw
        )
    return build


# Registry: method-name -> (class-or-factory, size-field, size-header)
METHODS = {
    "erl":                    (ERLMethod, "memory_size", "Memory Size"),
    "ace":                    (ACEMethod, "playbook_size", "Playbook Size"),
    "notebook_minimal":       (_notebook_factory("default"),
                               "notebook_size", "Notebook Lines"),
    "notebook_minimal_empty": (_notebook_factory("empty"),
                               "notebook_size", "Notebook Lines"),
}


ENVS = {
    "frozen_lake": FrozenLake,
    "sokoban": Sokoban,
}


def run_experiment(method_name: str, env_name: str, args) -> None:
    outputs_dir = Path(args.outputs_dir)
    os.environ["LLM_TRACE_PATH"] = str(
        outputs_dir / f"llm_calls_{method_name}_{env_name}.jsonl"
    )
    print(f"LM traces will be saved to {os.environ['LLM_TRACE_PATH']}")

    method_cls, size_field, size_header = METHODS[method_name]
    env = ENVS[env_name]()
    method = method_cls(
        env,
        model=args.model,
        server_url=args.server,
        disable_thinking=args.disable_thinking,
    )
    results = method.run(args.episodes)

    print_episode_table(
        results["logs"], size_field=size_field, size_header=size_header
    )

    out_path = results_path(args.outputs_dir, method_name, env_name)
    write_results(out_path, results)
    print(f"Saved to {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an experiment.")
    parser.add_argument(
        "--method", required=True, choices=sorted(METHODS),
        help="Method to run.",
    )
    parser.add_argument(
        "--env", choices=["frozen_lake", "sokoban", "both"], default="both",
        help="Which environment to run (default: both).",
    )
    parser.add_argument(
        "--episodes", type=int, default=20,
        help="Number of episodes per environment (default: 20).",
    )
    parser.add_argument(
        "--model", type=str, default="qwen3-8b",
        help="LM model name served by the backend (default: qwen3-8b).",
    )
    parser.add_argument(
        "--server", type=str, default="http://LOCAL_SERVER/v1",
        help=(
            "Base URL of an OpenAI-API-compatible inference server "
            "(e.g. SGLang); must expose /v1/chat/completions."
        ),
    )
    parser.add_argument(
        "--outputs-dir", dest="outputs_dir", default="./outputs",
        help="Where to write results_<method>_<env>.json (default: ./outputs).",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help=(
            "Disable Qwen3 thinking via SGLang chat_template_kwargs "
            "enable_thinking=False for every LM call."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    envs = ["frozen_lake", "sokoban"] if args.env == "both" else [args.env]
    for env_name in envs:
        run_experiment(args.method, env_name, args)


if __name__ == "__main__":
    main()
