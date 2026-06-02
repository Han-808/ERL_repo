#!/usr/bin/env python3
"""
MiniGrid notebook_minimal updater stability evaluation.

This consumes saved notebook_minimal MiniGrid llm_calls, resamples the saved
notebook updater prompt several times, applies sampled notebook edit operations
to the source notebook, and evaluates the resulting fixed notebook on a shared
bank of MiniGrid seeds.

Output:
  <outputs-root>/<run-name>/<game>/
    config.json
    updater_samples.jsonl
    rollouts.jsonl
    sample_summary.csv
    status.tsv
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    action_example_for_env,
    build_client,
    call_lm,
    default_action_for_env,
    parse_action_single_with_status,
    valid_actions_for_env,
)
from environments.minigrid_env import MiniGridTextEnv  # noqa: E402
from methods.notebook_minimal import (  # noqa: E402
    apply_notebook_operations,
    extract_json_payload,
    validate_operations,
)
from prompts import build_notebook_agent_prompt  # noqa: E402


NOTEBOOK_UPDATER_MARKER = "You are a notebook updater for an agent playing a grid puzzle."

GAME_CONFIGS: dict[str, dict[str, Any]] = {
    "minigrid_empty_random_5x5": {
        "env_id": "MiniGrid-Empty-Random-5x5-v0",
        "max_steps": 100,
        "long_game": False,
    },
    "minigrid_memorys11": {
        "env_id": "MiniGrid-MemoryS11-v0",
        "max_steps": 605,
        "long_game": True,
    },
    "minigrid_memorys13": {
        "env_id": "MiniGrid-MemoryS13-v0",
        "max_steps": 845,
        "long_game": True,
    },
    "minigrid_fourrooms": {
        "env_id": "MiniGrid-FourRooms-v0",
        "max_steps": 100,
        "long_game": False,
    },
}

SAMPLE_SUMMARY_FIELDS = [
    "game",
    "env_id",
    "source_seed",
    "source_episode",
    "sample_index",
    "eval_episodes",
    "eval_seed_offset",
    "mean_original_reward",
    "mean_binary_reward",
    "success_rate",
    "num_operations",
    "num_applied_operations",
    "updater_parse_status",
    "source_notebook_lines",
    "updated_notebook_lines",
    "updater_content_chars",
    "elapsed_seconds",
]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def append_jsonl(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=False) + "\n")
        handle.flush()


def append_tsv(path: Path, row: list[Any], header: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        if not exists:
            handle.write("\t".join(header) + "\n")
        handle.write("\t".join(str(x) for x in row) + "\n")
        handle.flush()


def ensure_csv_header(path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()


def append_csv(path: Path, row: dict[str, Any], fields: list[str]) -> None:
    ensure_csv_header(path, fields)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writerow({field: row.get(field) for field in fields})
        handle.flush()


def find_llm_calls_file(source_root: Path, game: str) -> Path:
    pattern = f"llm_calls_notebook_minimal_minigrid_{game}.jsonl"
    matches = sorted(source_root.rglob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {pattern} under {source_root}, found {len(matches)}"
        )
    return matches[0]


def response_content(payload: dict[str, Any]) -> str:
    try:
        return str(payload["response"]["choices"][0]["message"]["content"])
    except Exception:
        return ""


def finish_reason(payload: dict[str, Any]) -> str | None:
    try:
        return payload["response"]["choices"][0].get("finish_reason")
    except Exception:
        return None


def extract_line_numbered_notebook(prompt: str) -> str:
    match = re.search(
        r"Current notebook \(line-numbered\):\s*<<<NOTEBOOK>>>\s*(.*?)\s*<<<END_NOTEBOOK>>>",
        prompt,
        flags=re.DOTALL,
    )
    if not match:
        raise ValueError("Could not find line-numbered notebook in updater prompt")

    lines = []
    for line in match.group(1).splitlines():
        lines.append(re.sub(r"^\d{4}:\s?", "", line))
    return "\n".join(lines)


def extract_initial_observation(prompt: str) -> str:
    match = re.search(
        r"Initial observation:\s*<<<INITIAL>>>\s*(.*?)\s*<<<END_INITIAL>>>",
        prompt,
        flags=re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def extract_source_reward(prompt: str) -> float | None:
    match = re.search(r"\*\*Episode outcome:\s*[A-Z]+\*\*\s*\(reward\s*=\s*([^)]+)\)", prompt)
    if not match:
        return None
    try:
        return float(match.group(1).strip())
    except ValueError:
        return None


def extract_source_actions(prompt: str) -> list[str]:
    match = re.search(
        r"Actions taken:\s*(\[.*?\])\s*\n\nEnvironment feedback:",
        prompt,
        flags=re.DOTALL,
    )
    if not match:
        return []
    try:
        value = ast.literal_eval(match.group(1))
    except Exception:
        return []
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def raw_reward_from_feedback(feedback: str, counted_reward: float) -> float:
    matches = re.findall(r"Raw MiniGrid reward:\s*([-+0-9.eE]+)", feedback)
    if matches:
        try:
            return float(matches[-1])
        except ValueError:
            pass
    return float(counted_reward)


def iter_source_updater_chunks(llm_path: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    with llm_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            payload = json.loads(raw_line)
            prompt = str(payload.get("prompt", ""))
            if NOTEBOOK_UPDATER_MARKER not in prompt:
                continue

            source_episode = len(chunks) + 1
            chunks.append(
                {
                    "source_episode": source_episode,
                    "source_line": line_number,
                    "updater_prompt": prompt,
                    "source_updater_finish_reason": finish_reason(payload),
                    "source_updater_output_chars": len(response_content(payload)),
                    "source_notebook": extract_line_numbered_notebook(prompt),
                    "initial_observation": extract_initial_observation(prompt),
                    "source_actions": extract_source_actions(prompt),
                    "source_reward": extract_source_reward(prompt),
                }
            )
    return chunks


def parse_notebook_update(raw: str, source_notebook: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, str]:
    if not raw.strip():
        return [], [], source_notebook, "empty"
    try:
        payload = extract_json_payload(raw)
        operations = validate_operations(payload.get("operations", []))
    except Exception:
        return [], [], source_notebook, "unparseable"

    updated_notebook, applied = apply_notebook_operations(source_notebook, operations)
    if operations and applied:
        status = "parsed_applied"
    elif operations:
        status = "parsed_no_applied_ops"
    else:
        status = "parsed_empty_ops"
    return operations, applied, updated_notebook, status


def run_fixed_notebook_episode(
    *,
    env_id: str,
    max_steps: int,
    seed_offset: int,
    reset_seed: int,
    notebook: str,
    client,
    model: str,
    disable_thinking: bool,
) -> dict[str, Any]:
    env = MiniGridTextEnv(env_id, max_steps=max_steps, seed_offset=seed_offset)
    env.reset(seed=reset_seed)
    valid_actions = valid_actions_for_env(env)
    default_action = default_action_for_env(env)
    action_example = action_example_for_env(env)

    actions = []
    parsed_failures = 0
    feedbacks = []
    reward_binary = 0.0
    reward_original = 0.0
    started = time.time()

    try:
        while not env.done:
            obs = env.get_observation()
            prompt = build_notebook_agent_prompt(
                obs,
                notebook,
                valid_actions=valid_actions,
                action_example=action_example,
            )
            lm_output = call_lm(
                client,
                model,
                prompt,
                disable_thinking=disable_thinking,
                max_tokens=512,
            )
            action, parsed_ok = parse_action_single_with_status(
                lm_output,
                valid_actions=valid_actions,
                default_action=default_action,
            )
            if not parsed_ok:
                parsed_failures += 1
            _, feedback, counted_reward, done = env.step([action])
            actions.append(action)
            feedbacks.append(feedback)
            reward_binary = float(counted_reward if parsed_ok else 0.0)
            reward_original = raw_reward_from_feedback(feedback, reward_binary)
            if done:
                break
    finally:
        env.close()

    return {
        "reset_seed": reset_seed,
        "seed_offset": seed_offset,
        "num_steps": len(actions),
        "actions": actions,
        "parsed_failures": parsed_failures,
        "reward_binary": reward_binary,
        "reward_original": reward_original,
        "success": reward_original > 0,
        "elapsed_seconds": round(time.time() - started, 3),
        "final_feedback": feedbacks[-1] if feedbacks else "",
    }


def evaluate_game(args: argparse.Namespace) -> None:
    if args.game not in GAME_CONFIGS:
        raise ValueError(f"Unknown game {args.game}; choices={sorted(GAME_CONFIGS)}")
    game_config = GAME_CONFIGS[args.game]
    eval_episodes = (
        args.long_eval_episodes if game_config["long_game"] else args.eval_episodes
    )

    llm_path = find_llm_calls_file(Path(args.source_root), args.game)
    episode_chunks = iter_source_updater_chunks(llm_path)
    if not episode_chunks:
        raise ValueError(f"No notebook updater calls found in {llm_path}")

    run_dir = Path(args.outputs_root) / args.run_name / args.game
    run_dir.mkdir(parents=True, exist_ok=True)
    samples_path = run_dir / "updater_samples.jsonl"
    rollouts_path = run_dir / "rollouts.jsonl"
    summary_path = run_dir / "sample_summary.csv"
    status_path = run_dir / "status.tsv"

    config = {
        "game": args.game,
        "env_id": game_config["env_id"],
        "max_steps": game_config["max_steps"],
        "model": args.model,
        "server": args.server,
        "disable_thinking": args.disable_thinking,
        "source_method": "notebook_minimal_minigrid",
        "source_root": str(args.source_root),
        "source_llm_calls": str(llm_path),
        "source_episodes": len(episode_chunks),
        "samples_per_episode": args.samples_per_episode,
        "eval_episodes_per_sample": eval_episodes,
        "eval_episodes_default": args.eval_episodes,
        "long_eval_episodes": args.long_eval_episodes,
        "base_seed": args.base_seed,
        "source_seed": args.source_seed,
        "seed_stride": args.seed_stride,
        "updater_max_tokens": args.updater_max_tokens,
        "generator_max_tokens": 512,
    }
    write_json(run_dir / "config.json", config)
    ensure_csv_header(summary_path, SAMPLE_SUMMARY_FIELDS)

    total_samples = len(episode_chunks) * args.samples_per_episode
    processed_samples = 0

    print(f"Game: {args.game}", flush=True)
    print(f"Source llm_calls: {llm_path}", flush=True)
    print(f"Output: {run_dir}", flush=True)
    print(f"Source updater episodes: {len(episode_chunks)}", flush=True)
    print(f"Expected updater samples: {total_samples}", flush=True)
    print(f"Eval episodes per sample: {eval_episodes}", flush=True)

    client = build_client(args.server)
    started_game = time.time()

    for chunk in episode_chunks:
        source_episode = int(chunk["source_episode"])
        source_notebook = str(chunk["source_notebook"])
        source_notebook_lines = len(source_notebook.splitlines())

        for sample_index in range(args.samples_per_episode):
            sample_started = time.time()
            updater_output = call_lm(
                client,
                args.model,
                chunk["updater_prompt"],
                disable_thinking=args.disable_thinking,
                max_tokens=args.updater_max_tokens,
            )
            operations, applied, updated_notebook, parse_status = parse_notebook_update(
                updater_output,
                source_notebook,
            )
            updated_notebook_lines = len(updated_notebook.splitlines())

            sample_record = {
                "game": args.game,
                "env_id": game_config["env_id"],
                "source_seed": args.source_seed,
                "source_episode": source_episode,
                "sample_index": sample_index,
                "source_llm_calls": str(llm_path),
                "source_updater_line": chunk["source_line"],
                "source_updater_finish_reason": chunk["source_updater_finish_reason"],
                "source_updater_output_chars": chunk["source_updater_output_chars"],
                "source_reward": chunk["source_reward"],
                "source_actions": chunk["source_actions"],
                "initial_observation": chunk["initial_observation"],
                "source_notebook_lines": source_notebook_lines,
                "num_operations": len(operations),
                "num_applied_operations": len(applied),
                "updated_notebook_lines": updated_notebook_lines,
                "updater_parse_status": parse_status,
                "updater_content_chars": len(updater_output),
                "updater_output": updater_output,
                "operations": operations,
                "applied_operations": applied,
                "updated_notebook": updated_notebook,
            }
            append_jsonl(samples_path, sample_record)

            rollout_rows = []
            seed_offset = int(args.source_seed) * int(args.seed_stride)
            for eval_index in range(eval_episodes):
                reset_seed = int(args.base_seed) + eval_index
                rollout = run_fixed_notebook_episode(
                    env_id=game_config["env_id"],
                    max_steps=int(game_config["max_steps"]),
                    seed_offset=seed_offset,
                    reset_seed=reset_seed,
                    notebook=updated_notebook,
                    client=client,
                    model=args.model,
                    disable_thinking=args.disable_thinking,
                )
                rollout_record = {
                    "game": args.game,
                    "env_id": game_config["env_id"],
                    "source_seed": args.source_seed,
                    "source_episode": source_episode,
                    "sample_index": sample_index,
                    "eval_index": eval_index,
                    **rollout,
                }
                rollout_rows.append(rollout_record)
                append_jsonl(rollouts_path, rollout_record)

            original_rewards = [float(row["reward_original"]) for row in rollout_rows]
            binary_rewards = [float(row["reward_binary"]) for row in rollout_rows]
            successes = [1.0 if row["success"] else 0.0 for row in rollout_rows]
            summary = {
                "game": args.game,
                "env_id": game_config["env_id"],
                "source_seed": args.source_seed,
                "source_episode": source_episode,
                "sample_index": sample_index,
                "eval_episodes": eval_episodes,
                "eval_seed_offset": seed_offset,
                "mean_original_reward": (
                    statistics.fmean(original_rewards) if original_rewards else 0.0
                ),
                "mean_binary_reward": (
                    statistics.fmean(binary_rewards) if binary_rewards else 0.0
                ),
                "success_rate": statistics.fmean(successes) if successes else 0.0,
                "num_operations": len(operations),
                "num_applied_operations": len(applied),
                "updater_parse_status": parse_status,
                "source_notebook_lines": source_notebook_lines,
                "updated_notebook_lines": updated_notebook_lines,
                "updater_content_chars": len(updater_output),
                "elapsed_seconds": round(time.time() - sample_started, 3),
            }
            append_csv(summary_path, summary, SAMPLE_SUMMARY_FIELDS)
            append_tsv(
                status_path,
                [
                    args.game,
                    args.source_seed,
                    source_episode,
                    sample_index,
                    "done",
                    f"{summary['mean_original_reward']:.6g}",
                    f"{summary['success_rate']:.6g}",
                    len(operations),
                    len(applied),
                    round(time.time() - sample_started, 3),
                ],
                [
                    "game",
                    "source_seed",
                    "source_episode",
                    "sample_index",
                    "status",
                    "mean_original_reward",
                    "success_rate",
                    "num_operations",
                    "num_applied_operations",
                    "elapsed_seconds",
                ],
            )

            processed_samples += 1
            print(
                "PROGRESS "
                f"game={args.game} "
                f"processed_samples={processed_samples}/{total_samples} "
                f"source_episode={source_episode} "
                f"sample={sample_index} "
                f"mean_reward={summary['mean_original_reward']:.4f} "
                f"success_rate={summary['success_rate']:.3f}",
                flush=True,
            )

    print(
        f"DONE game={args.game} samples={processed_samples}/{total_samples} "
        f"elapsed_seconds={time.time() - started_game:.1f} output={run_dir}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3-14B notebook_minimal updater stability on MiniGrid."
    )
    parser.add_argument("--game", required=True, choices=sorted(GAME_CONFIGS))
    parser.add_argument(
        "--source-root",
        default=os.environ.get(
            "SOURCE_ROOT",
            "/gscratch/h2lab/mohanc3/projects/ERL_repo/minigrid_notebook_minimal_llm_calls",
        ),
        help="Directory containing notebook_minimal MiniGrid llm_calls.",
    )
    parser.add_argument(
        "--outputs-root",
        default=os.environ.get(
            "OUTPUTS_ROOT",
            "/gscratch/h2lab/mohanc3/projects/ERL_repo/minigrid_notebook_minimal_stability",
        ),
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--server", required=True)
    parser.add_argument("--samples-per-episode", type=int, default=4)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--long-eval-episodes", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=20260601)
    parser.add_argument("--source-seed", type=int, default=0)
    parser.add_argument("--seed-stride", type=int, default=10000)
    parser.add_argument("--updater-max-tokens", type=int, default=8192)
    parser.add_argument("--disable-thinking", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    evaluate_game(args)


if __name__ == "__main__":
    main()
