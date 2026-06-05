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
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import print_episode_table, results_path, write_results
from environments.frozen_lake import FrozenLake
from environments.sokoban import Sokoban
from methods.ace import ACEMethod
from methods.ace_once import ACEOnceMethod, ACEOnceMiniGridMethod
from methods.erl import ERLMethod
from methods.notebook_minimal import (
    NotebookMinimalMethod,
    NotebookMinimalMiniGridMethod,
)
from methods.notebook_minimal_mechanism import (
    NotebookMinimalMechanismMethod,
    NotebookMinimalMechanismMiniGridMethod,
)
from methods.notebook_minimal_thinkahead import NotebookMinimalThinkAheadMethod


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
    "ace_once":               (ACEOnceMethod, "playbook_size", "Playbook Size"),
    "ace_once_minigrid":      (ACEOnceMiniGridMethod, "playbook_size", "Playbook Size"),
    "notebook_minimal":       (_notebook_factory("empty"),
                               "notebook_size", "Notebook Lines"),
    "notebook_minimal_empty": (_notebook_factory("empty"),
                               "notebook_size", "Notebook Lines"),
    "notebook_minimal_default": (_notebook_factory("default"),
                                 "notebook_size", "Notebook Lines"),
    "notebook_minimal_mechanism": (NotebookMinimalMechanismMethod,
                                   "notebook_size", "Notebook Lines"),
    "notebook_minimal_mechanism_minigrid": (
        NotebookMinimalMechanismMiniGridMethod,
        "notebook_size",
        "Notebook Lines",
    ),
    "notebook_minimal_minigrid": (NotebookMinimalMiniGridMethod,
                                  "notebook_size", "Notebook Lines"),
    "notebook_minimal_thinkahead": (NotebookMinimalThinkAheadMethod,
                                    "notebook_size", "Notebook Lines"),
}


ENVS = {
    "frozen_lake": FrozenLake,
    "sokoban": Sokoban,
}


def minigrid_output_label(env_id: str) -> str:
    label = env_id
    if label.startswith("MiniGrid-"):
        label = label[len("MiniGrid-"):]
    label = re.sub(r"-v\d+$", "", label)
    label = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").lower()
    return f"minigrid_{label or 'env'}"


def build_env(env_name: str, args):
    if env_name == "minigrid":
        from environments.minigrid_env import MiniGridTextEnv
        return MiniGridTextEnv(
            args.minigrid_id,
            max_steps=args.minigrid_max_steps,
            seed_offset=args.seed_offset,
        )
    return ENVS[env_name]()


def output_env_name(env_name: str, args) -> str:
    if env_name == "minigrid":
        return minigrid_output_label(args.minigrid_id)
    return env_name


def run_experiment(method_name: str, env_name: str, args) -> None:
    outputs_dir = Path(args.outputs_dir)
    env_label = output_env_name(env_name, args)
    os.environ["LLM_TRACE_PATH"] = str(
        outputs_dir / f"llm_calls_{method_name}_{env_label}.jsonl"
    )
    print(f"LM traces will be saved to {os.environ['LLM_TRACE_PATH']}")

    method_cls, size_field, size_header = METHODS[method_name]
    env = build_env(env_name, args)
    method_kwargs = {
        "model": args.model,
        "server_url": args.server,
        "disable_thinking": args.disable_thinking,
    }
    if method_name in {"ace", "ace_once", "ace_once_minigrid"}:
        method_kwargs["updater_model"] = args.updater_model
        method_kwargs["updater_server_url"] = args.updater_server
    method = method_cls(env, **method_kwargs)
    results = method.run(args.episodes)

    print_episode_table(
        results["logs"], size_field=size_field, size_header=size_header
    )

    out_path = results_path(args.outputs_dir, method_name, env_label)
    write_results(out_path, results)
    print(f"Saved to {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an experiment.")
    parser.add_argument(
        "--method", required=True, choices=sorted(METHODS),
        help="Method to run.",
    )
    parser.add_argument(
        "--env", choices=["frozen_lake", "sokoban", "both", "minigrid"],
        default="both",
        help="Which environment to run (default: both).",
    )
    parser.add_argument(
        "--minigrid-id",
        default="MiniGrid-Empty-5x5-v0",
        help=(
            "Gymnasium MiniGrid id to use when --env minigrid "
            "(default: MiniGrid-Empty-5x5-v0)."
        ),
    )
    parser.add_argument(
        "--minigrid-max-steps",
        type=int,
        default=None,
        help=(
            "Override MiniGrid max_steps when --env minigrid. "
            "Default keeps the environment's built-in limit."
        ),
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help=(
            "Offset added to per-episode environment seeds. "
            "Useful for independent repeated MiniGrid runs."
        ),
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
        "--updater-model", type=str, default=None,
        help=(
            "Optional LM model for ACE/ACE_ONCE updater calls. "
            "Defaults to --model when unset."
        ),
    )
    parser.add_argument(
        "--updater-server", type=str, default=None,
        help=(
            "Optional OpenAI-compatible server URL for ACE/ACE_ONCE updater "
            "calls. Defaults to --server when unset."
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
