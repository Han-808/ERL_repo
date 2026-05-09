"""
Single-turn pass@k-style evaluation for notebook_minimal.

This workflow freezes each notebook state from a previous notebook_minimal
run, then evaluates that fixed notebook on fresh randomized grid games.

Default experiment shape per environment:
  40 notebook states x 8 independent samples x 10 games per sample.

Outputs are written under:
  single_turn_passk_runs/<run_name>/
    config.json
    notebook_states.jsonl
    rollouts.jsonl
    sample_summary.csv
    state_summary.csv
    llm_calls_single_turn_passk_<env>.jsonl
    plots/*.png              (optional; requires matplotlib)

The LM prompt used for each rollout step is exactly
prompts.build_notebook_agent_prompt(observation, notebook), matching the
agent-side prompts recorded in llm_calls_notebook_minimal_*.jsonl.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from common import build_client, call_lm, parse_action_single
from environments.frozen_lake import FrozenLake
from environments.sokoban import Sokoban
from prompts import build_notebook_agent_prompt


AGENT_MARKER = "Your notebook below contains knowledge accumulated"
UPDATER_MARKER = "Current notebook (line-numbered):"
NOTEBOOK_BEGIN = "<<<NOTEBOOK>>>\n"
NOTEBOOK_END = "<<<END_NOTEBOOK>>>"

DEFAULT_TRACES = {
    "frozen_lake": "llm_calls_notebook_minimal_frozen_lake.jsonl",
    "sokoban": "llm_calls_notebook_minimal_sokoban.jsonl",
}

ENV_CLASSES = {
    "frozen_lake": FrozenLake,
    "sokoban": Sokoban,
}

ENV_SEED_OFFSETS = {
    "frozen_lake": 0,
    "sokoban": 500_000_000,
}


@dataclass
class NotebookState:
    env: str
    state_x: int
    source_trace: str
    source_line: int
    notebook: str
    notebook_hash: str
    notebook_size_lines: int
    reference_observation: str
    reference_agent_prompt: str
    reference_agent_prompt_hash: str
    reference_prompt_matches_template: bool
    source_model: str | None
    source_disable_thinking: bool | None


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def append_jsonl(handle, data: Any) -> None:
    handle.write(json.dumps(data, ensure_ascii=False) + "\n")
    handle.flush()


def extract_between(text: str, begin: str, end: str) -> str:
    start = text.index(begin) + len(begin)
    stop = text.index(end, start)
    return text[start:stop]


def extract_notebook_from_agent_prompt(prompt: str) -> str:
    notebook_with_template_separator = extract_between(
        prompt, NOTEBOOK_BEGIN, NOTEBOOK_END
    )
    if notebook_with_template_separator.endswith("\n"):
        return notebook_with_template_separator[:-1]
    return notebook_with_template_separator


def extract_observation_from_agent_prompt(prompt: str) -> str:
    split_at = "\n\nYour notebook below contains knowledge accumulated"
    if split_at not in prompt:
        raise ValueError("agent prompt does not contain notebook marker")
    return prompt.split(split_at, 1)[0]


def is_agent_call(prompt: str) -> bool:
    return AGENT_MARKER in prompt and NOTEBOOK_BEGIN in prompt


def is_updater_call(prompt: str) -> bool:
    return UPDATER_MARKER in prompt and NOTEBOOK_BEGIN in prompt


def load_notebook_states(
    trace_path: Path,
    env_name: str,
    max_states: int,
) -> list[NotebookState]:
    """
    Recover Notebook x for each episode x from a notebook_minimal LM trace.

    The trace sequence is:
      agent step calls with Notebook x, then one updater call, then
      agent step calls with Notebook x+1, ...

    We therefore keep only the first agent prompt after the start of the trace
    or after an updater call. This captures the exact notebook text the agent
    saw at the beginning of each episode.
    """
    states: list[NotebookState] = []
    expect_new_episode = True

    with trace_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if len(states) >= max_states:
                break
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            prompt = row.get("prompt", "")
            if is_updater_call(prompt):
                expect_new_episode = True
                continue
            if not expect_new_episode or not is_agent_call(prompt):
                continue

            notebook = extract_notebook_from_agent_prompt(prompt)
            observation = extract_observation_from_agent_prompt(prompt)
            rebuilt = build_notebook_agent_prompt(observation, notebook)
            state_x = len(states) + 1
            states.append(
                NotebookState(
                    env=env_name,
                    state_x=state_x,
                    source_trace=str(trace_path),
                    source_line=line_number,
                    notebook=notebook,
                    notebook_hash=sha256_text(notebook),
                    notebook_size_lines=len(notebook.splitlines()),
                    reference_observation=observation,
                    reference_agent_prompt=prompt,
                    reference_agent_prompt_hash=sha256_text(prompt),
                    reference_prompt_matches_template=(rebuilt == prompt),
                    source_model=row.get("model"),
                    source_disable_thinking=row.get("disable_thinking"),
                )
            )
            expect_new_episode = False

    if not states:
        raise ValueError(f"No notebook states found in {trace_path}")
    return states


def rollout_seed(
    base_seed: int,
    env_name: str,
    state_x: int,
    sample_y: int,
    game_z: int,
) -> int:
    return (
        base_seed
        + ENV_SEED_OFFSETS[env_name]
        + state_x * 1_000_000
        + sample_y * 10_000
        + game_z
    )


def run_fixed_notebook_game(
    *,
    env_name: str,
    seed: int,
    notebook: str,
    client,
    model: str,
    disable_thinking: bool,
    reward_threshold: float,
    fail_on_empty_lm_output: bool,
) -> dict:
    env_cls = ENV_CLASSES[env_name]
    env = env_cls(seed=seed)
    initial_observation = env.get_observation()
    actions: list[str] = []
    feedbacks: list[str] = []
    step_records: list[dict] = []
    reward = 0

    while not env.done:
        step_index = len(actions) + 1
        observation = env.get_observation()
        prompt = build_notebook_agent_prompt(observation, notebook)
        lm_output = call_lm(
            client,
            model,
            prompt,
            disable_thinking=disable_thinking,
        )
        if fail_on_empty_lm_output and not lm_output.strip():
            raise RuntimeError(
                "LM returned empty output. Check --server/model, or pass "
                "--allow-empty-lm-output to keep existing fallback behavior."
            )
        action = parse_action_single(lm_output)
        _, feedback, reward, done = env.step([action])

        actions.append(action)
        feedbacks.append(feedback)
        step_records.append(
            {
                "step": step_index,
                "observation": observation,
                "prompt_hash": sha256_text(prompt),
                "lm_output": lm_output,
                "action": action,
                "feedback": feedback,
                "reward_after_step": reward,
                "done": done,
                "empty_lm_output": not bool(lm_output.strip()),
            }
        )
        if done:
            break

    success = reward >= reward_threshold
    return {
        "seed": seed,
        "initial_observation": initial_observation,
        "final_observation": env.get_observation(),
        "actions": actions,
        "feedback": " ".join(feedbacks),
        "reward": reward,
        "success": success,
        "num_steps": len(actions),
        "step_records": step_records,
    }


def summarize_samples(rollouts: list[dict]) -> tuple[list[dict], list[dict]]:
    by_sample: dict[tuple[str, int, int], list[dict]] = {}
    by_state: dict[tuple[str, int], list[dict]] = {}

    for row in rollouts:
        sample_key = (row["env"], row["state_x"], row["sample_y"])
        state_key = (row["env"], row["state_x"])
        by_sample.setdefault(sample_key, []).append(row)
        by_state.setdefault(state_key, []).append(row)

    sample_rows: list[dict] = []
    for (env, state_x, sample_y), rows in sorted(by_sample.items()):
        rewards = [float(r["reward"]) for r in rows]
        successes = [1 if r["success"] else 0 for r in rows]
        sample_rows.append(
            {
                "env": env,
                "state_x": state_x,
                "sample_y": sample_y,
                "num_games": len(rows),
                "num_success": sum(successes),
                "pass_rate": sum(successes) / len(rows) if rows else 0.0,
                "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
                "std_reward": statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
                "min_reward": min(rewards) if rewards else 0.0,
                "max_reward": max(rewards) if rewards else 0.0,
            }
        )

    sample_rates_by_state: dict[tuple[str, int], list[float]] = {}
    for row in sample_rows:
        key = (row["env"], row["state_x"])
        sample_rates_by_state.setdefault(key, []).append(float(row["pass_rate"]))

    state_rows: list[dict] = []
    for (env, state_x), rows in sorted(by_state.items()):
        successes = [1 if r["success"] else 0 for r in rows]
        sample_rates = sample_rates_by_state.get((env, state_x), [])
        mean_sample_pass = (
            sum(sample_rates) / len(sample_rates) if sample_rates else 0.0
        )
        std_sample_pass = (
            statistics.stdev(sample_rates) if len(sample_rates) > 1 else 0.0
        )
        state_rows.append(
            {
                "env": env,
                "state_x": state_x,
                "num_samples": len(sample_rates),
                "total_games": len(rows),
                "total_success": sum(successes),
                "overall_pass_rate": sum(successes) / len(rows) if rows else 0.0,
                "mean_sample_pass_rate": mean_sample_pass,
                "std_sample_pass_rate": std_sample_pass,
                "sem_sample_pass_rate": (
                    std_sample_pass / math.sqrt(len(sample_rates))
                    if sample_rates
                    else 0.0
                ),
                "min_sample_pass_rate": min(sample_rates) if sample_rates else 0.0,
                "max_sample_pass_rate": max(sample_rates) if sample_rates else 0.0,
            }
        )
    return sample_rows, state_rows


def attach_notebook_metadata(
    rows: list[dict],
    states_by_env: dict[str, list[NotebookState]],
) -> list[dict]:
    state_lookup = {
        (state.env, state.state_x): state
        for states in states_by_env.values()
        for state in states
    }
    enriched = []
    for row in rows:
        out = dict(row)
        state = state_lookup[(row["env"], row["state_x"])]
        out["notebook_hash"] = state.notebook_hash
        out["notebook_size_lines"] = state.notebook_size_lines
        out["reference_prompt_matches_template"] = (
            state.reference_prompt_matches_template
        )
        enriched.append(out)
    return enriched


def write_jsonl(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            append_jsonl(handle, row)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_visualizations(
    state_rows: list[dict],
    plots_dir: Path,
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on local env
        (plots_dir / "plot_warning.txt").write_text(
            f"matplotlib unavailable; skipped plots. Error: {exc}\n",
            encoding="utf-8",
        )
        return

    envs = sorted({row["env"] for row in state_rows})

    def rows_for(env: str) -> list[dict]:
        return sorted(
            [row for row in state_rows if row["env"] == env],
            key=lambda r: int(r["state_x"]),
        )

    plt.figure(figsize=(9, 5))
    for env in envs:
        rows = rows_for(env)
        xs = [int(r["state_x"]) for r in rows]
        ys = [float(r["mean_sample_pass_rate"]) for r in rows]
        plt.plot(xs, ys, marker="o", linewidth=1.5, label=env)
    plt.xlabel("Notebook state x")
    plt.ylabel("Mean pass rate across y samples")
    plt.title("Single-turn pass@k mean pass rate")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "mean_pass_rate.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    for env in envs:
        rows = rows_for(env)
        xs = [int(r["state_x"]) for r in rows]
        ys = [float(r["std_sample_pass_rate"]) for r in rows]
        plt.plot(xs, ys, marker="o", linewidth=1.5, label=env)
    plt.xlabel("Notebook state x")
    plt.ylabel("Std. dev. across y sample pass rates")
    plt.title("Prompt stability by notebook state")
    plt.ylim(bottom=0.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "std_pass_rate.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    for env in envs:
        rows = rows_for(env)
        xs = [int(r["state_x"]) for r in rows]
        means = [float(r["mean_sample_pass_rate"]) for r in rows]
        stds = [float(r["std_sample_pass_rate"]) for r in rows]
        lower = [max(0.0, m - s) for m, s in zip(means, stds)]
        upper = [min(1.0, m + s) for m, s in zip(means, stds)]
        plt.plot(xs, means, marker="o", linewidth=1.5, label=env)
        plt.fill_between(xs, lower, upper, alpha=0.16)
    plt.xlabel("Notebook state x")
    plt.ylabel("Pass rate")
    plt.title("Mean pass rate with +/-1 std. band")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "mean_with_std_band.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    for env in envs:
        rows = rows_for(env)
        xs = [int(r["state_x"]) for r in rows]
        ys = [int(r["notebook_size_lines"]) for r in rows]
        plt.plot(xs, ys, marker="o", linewidth=1.5, label=env)
    plt.xlabel("Notebook state x")
    plt.ylabel("Notebook lines")
    plt.title("Notebook size over states")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "notebook_size.png", dpi=160)
    plt.close()


def resolve_trace_path(args, env_name: str) -> Path:
    override = {
        "frozen_lake": args.frozen_lake_trace,
        "sokoban": args.sokoban_trace,
    }[env_name]
    if override:
        return Path(override)
    return Path(args.trace_dir) / DEFAULT_TRACES[env_name]


def selected_envs(env_arg: str) -> list[str]:
    if env_arg == "both":
        return ["frozen_lake", "sokoban"]
    return [env_arg]


def filter_states(states: list[NotebookState], args) -> list[NotebookState]:
    """Apply optional state range and shard selection."""
    out = states
    if args.state_start is not None:
        out = [state for state in out if state.state_x >= args.state_start]
    if args.state_end is not None:
        out = [state for state in out if state.state_x <= args.state_end]

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= index < num_shards")
    if args.num_shards > 1:
        out = [
            state for state in out
            if (state.state_x - 1) % args.num_shards == args.shard_index
        ]
    return out


def make_run_dir(outputs_dir: Path, run_name: str | None) -> Path:
    if run_name is None:
        run_name = time.strftime("single_turn_passk_%Y%m%d_%H%M%S")
    run_dir = outputs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def run_eval(args) -> Path:
    env_names = selected_envs(args.env)
    run_dir = make_run_dir(Path(args.outputs_dir), args.run_name)
    states_by_env: dict[str, list[NotebookState]] = {}

    for env_name in env_names:
        trace_path = resolve_trace_path(args, env_name)
        states = load_notebook_states(
            trace_path=trace_path,
            env_name=env_name,
            max_states=args.num_states,
        )
        if len(states) < args.num_states:
            print(
                f"[warning] {env_name}: requested {args.num_states} states, "
                f"found {len(states)} in {trace_path}"
            )
        states = filter_states(states, args)
        if not states:
            raise ValueError(
                f"{env_name}: no notebook states selected after applying "
                "range/shard filters."
            )
        states_by_env[env_name] = states

    prompt_mismatches = [
        state
        for states in states_by_env.values()
        for state in states
        if not state.reference_prompt_matches_template
    ]
    if prompt_mismatches and not args.allow_prompt_mismatch:
        examples = ", ".join(
            f"{state.env}:x={state.state_x}:line={state.source_line}"
            for state in prompt_mismatches[:5]
        )
        raise ValueError(
            "Reconstructed notebook prompts do not match "
            f"build_notebook_agent_prompt. Examples: {examples}. "
            "Pass --allow-prompt-mismatch to continue anyway."
        )

    source_models = sorted(
        {
            state.source_model
            for states in states_by_env.values()
            for state in states
            if state.source_model
        }
    )
    source_disable_values = sorted(
        {
            state.source_disable_thinking
            for states in states_by_env.values()
            for state in states
            if state.source_disable_thinking is not None
        }
    )
    model = args.model or (source_models[0] if source_models else "qwen3-8b")
    if args.disable_thinking is None:
        disable_thinking = (
            bool(source_disable_values[0]) if source_disable_values else False
        )
    else:
        disable_thinking = bool(args.disable_thinking)

    config = {
        "env": args.env,
        "envs": env_names,
        "num_states_requested": args.num_states,
        "state_start": args.state_start,
        "state_end": args.state_end,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "selected_states": {
            env_name: [state.state_x for state in states_by_env[env_name]]
            for env_name in env_names
        },
        "samples_y": args.samples_y,
        "games_z": args.games_z,
        "total_rollouts_planned": (
            sum(len(v) for v in states_by_env.values())
            * args.samples_y
            * args.games_z
        ),
        "base_seed": args.base_seed,
        "seed_formula": (
            "base_seed + env_offset + state_x*1000000 + "
            "sample_y*10000 + game_z"
        ),
        "reward_threshold": args.reward_threshold,
        "model": model,
        "server": args.server,
        "disable_thinking": disable_thinking,
        "call_lm_settings": {
            "max_tokens": 512,
            "temperature": 0.7,
        },
        "fail_on_empty_lm_output": not args.allow_empty_lm_output,
        "allow_prompt_mismatch": args.allow_prompt_mismatch,
        "trace_paths": {
            env_name: str(resolve_trace_path(args, env_name))
            for env_name in env_names
        },
        "source_models": source_models,
        "source_disable_thinking_values": source_disable_values,
        "created_at_unix": int(time.time()),
        "run_dir": str(run_dir),
        "extract_only": args.extract_only,
    }
    write_json(run_dir / "config.json", config)

    all_state_rows = [
        asdict(state)
        for env_name in env_names
        for state in states_by_env[env_name]
    ]
    write_jsonl(run_dir / "notebook_states.jsonl", all_state_rows)

    if args.extract_only:
        print(f"Extracted notebook states only. Wrote {run_dir}")
        return run_dir

    client = build_client(args.server)
    all_rollouts: list[dict] = []
    rollouts_path = run_dir / "rollouts.jsonl"

    with rollouts_path.open("w", encoding="utf-8") as rollouts_handle:
        for env_name in env_names:
            lm_trace_path = run_dir / f"llm_calls_single_turn_passk_{env_name}.jsonl"
            os.environ["LLM_TRACE_PATH"] = str(lm_trace_path)
            print(f"[{env_name}] LM traces -> {lm_trace_path}")

            states = states_by_env[env_name]
            for state in states:
                print(
                    f"[{env_name}] state {state.state_x}/{len(states)} "
                    f"notebook_lines={state.notebook_size_lines}"
                )
                for sample_y in range(1, args.samples_y + 1):
                    for game_z in range(1, args.games_z + 1):
                        seed = rollout_seed(
                            args.base_seed,
                            env_name,
                            state.state_x,
                            sample_y,
                            game_z,
                        )
                        game = run_fixed_notebook_game(
                            env_name=env_name,
                            seed=seed,
                            notebook=state.notebook,
                            client=client,
                            model=model,
                            disable_thinking=disable_thinking,
                            reward_threshold=args.reward_threshold,
                            fail_on_empty_lm_output=(
                                not args.allow_empty_lm_output
                            ),
                        )
                        row = {
                            "env": env_name,
                            "state_x": state.state_x,
                            "sample_y": sample_y,
                            "game_z": game_z,
                            "seed": seed,
                            "notebook_hash": state.notebook_hash,
                            "notebook_size_lines": state.notebook_size_lines,
                            "model": model,
                            "disable_thinking": disable_thinking,
                            **game,
                        }
                        all_rollouts.append(row)
                        append_jsonl(rollouts_handle, row)

    sample_rows, state_rows = summarize_samples(all_rollouts)
    sample_rows = attach_notebook_metadata(sample_rows, states_by_env)
    state_rows = attach_notebook_metadata(state_rows, states_by_env)

    write_csv(run_dir / "sample_summary.csv", sample_rows)
    write_csv(run_dir / "state_summary.csv", state_rows)

    if args.plots:
        generate_visualizations(state_rows, run_dir / "plots")

    print(f"Done. Wrote {run_dir}")
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run single-turn pass@k eval for notebook_minimal states."
    )
    parser.add_argument(
        "--env",
        choices=["frozen_lake", "sokoban", "both"],
        default="both",
        help="Environment(s) to evaluate (default: both).",
    )
    parser.add_argument(
        "--trace-dir",
        default=".",
        help="Directory containing llm_calls_notebook_minimal_*.jsonl.",
    )
    parser.add_argument(
        "--frozen-lake-trace",
        default=None,
        help="Override path for llm_calls_notebook_minimal_frozen_lake.jsonl.",
    )
    parser.add_argument(
        "--sokoban-trace",
        default=None,
        help="Override path for llm_calls_notebook_minimal_sokoban.jsonl.",
    )
    parser.add_argument(
        "--num-states",
        type=int,
        default=40,
        help="Number of notebook states to recover/evaluate (default: 40).",
    )
    parser.add_argument(
        "--state-start",
        type=int,
        default=None,
        help="Optional first notebook state x to evaluate, inclusive.",
    )
    parser.add_argument(
        "--state-end",
        type=int,
        default=None,
        help="Optional last notebook state x to evaluate, inclusive.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split selected states across this many shards (default: 1).",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard index to run when --num-shards > 1.",
    )
    parser.add_argument(
        "--samples-y",
        type=int,
        default=8,
        help="Independent sample batches per notebook state (default: 8).",
    )
    parser.add_argument(
        "--games-z",
        type=int,
        default=10,
        help="Randomized games per sample batch (default: 10).",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=20260509,
        help="Base seed for deterministic eval-grid generation.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LM model name. Defaults to the model found in source traces.",
    )
    parser.add_argument(
        "--server",
        default="http://LOCAL_SERVER/v1",
        help="OpenAI-compatible server base URL.",
    )
    parser.add_argument(
        "--disable-thinking",
        dest="disable_thinking",
        action="store_true",
        help="Force Qwen thinking disabled for eval calls.",
    )
    parser.add_argument(
        "--enable-thinking",
        dest="disable_thinking",
        action="store_false",
        help="Force Qwen thinking enabled for eval calls.",
    )
    parser.set_defaults(disable_thinking=None)
    parser.add_argument(
        "--reward-threshold",
        type=float,
        default=1.0,
        help="Reward threshold counted as pass/success (default: 1.0).",
    )
    parser.add_argument(
        "--outputs-dir",
        default="single_turn_passk_runs",
        help="Parent directory for eval outputs.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run directory name under --outputs-dir.",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only recover notebook states and write config/notebook_states.",
    )
    parser.add_argument(
        "--allow-prompt-mismatch",
        action="store_true",
        help=(
            "Continue even if source prompts cannot be exactly rebuilt from "
            "build_notebook_agent_prompt."
        ),
    )
    parser.add_argument(
        "--allow-empty-lm-output",
        action="store_true",
        help=(
            "Keep existing fallback behavior when LM output is empty. By "
            "default the eval aborts to avoid bogus server-failure results."
        ),
    )
    parser.add_argument(
        "--no-plots",
        dest="plots",
        action="store_false",
        help="Skip matplotlib plot generation.",
    )
    parser.set_defaults(plots=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
