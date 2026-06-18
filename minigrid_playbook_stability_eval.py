#!/usr/bin/env python3
"""
MiniGrid ACE_ONCE playbook-update stability evaluation.

This consumes completed ACE_ONCE MiniGrid results/llm_calls, resamples the
saved merged updater prompt several times, applies each sampled ADD-only
playbook update to the episode's original playbook, and evaluates the resulting
fixed playbook on a shared bank of MiniGrid seeds.

The output is shaped for later violin plots:
  <outputs-root>/<run-name>/<game>/
    config.json
    updater_samples.jsonl
    rollouts.jsonl
    sample_summary.csv
    status.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
from dataclasses import asdict
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
from methods.ace import (  # noqa: E402
    Playbook,
    PlaybookItem,
    _parse_delta_items,
    build_generator_prompt_with_playbook,
)


GENERATOR_MARKER = "You are the Generator in an ACE"
UPDATER_MARKER = "Your job has two parts, done in a single pass"

GAME_CONFIGS: dict[str, dict[str, Any]] = {
    "minigrid_empty_random_5x5": {
        "env_id": "MiniGrid-Empty-Random-5x5-v0",
        "max_steps": 100,
        "source_episodes_per_seed": 40,
        "long_game": False,
    },
    "minigrid_memorys11": {
        "env_id": "MiniGrid-MemoryS11-v0",
        "max_steps": 605,
        "source_episodes_per_seed": 40,
        "long_game": True,
    },
    "minigrid_memorys13": {
        "env_id": "MiniGrid-MemoryS13-v0",
        "max_steps": 845,
        "source_episodes_per_seed": 30,
        "long_game": True,
    },
    "minigrid_fourrooms": {
        "env_id": "MiniGrid-FourRooms-v0",
        "max_steps": 100,
        "source_episodes_per_seed": 20,
        "long_game": False,
    },
    "minigrid_simplecrossings9n3": {
        "env_id": "MiniGrid-SimpleCrossingS9N3-v0",
        "max_steps": 324,
        "source_episodes_per_seed": 20,
        "long_game": False,
    },
    "minigrid_distshift1": {
        "env_id": "MiniGrid-DistShift1-v0",
        "max_steps": 252,
        "source_episodes_per_seed": 20,
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
    "num_delta_items",
    "updater_parse_status",
    "source_playbook_size",
    "updated_playbook_size",
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


def classify_call(payload: dict[str, Any]) -> str:
    prompt = str(payload.get("prompt", ""))
    if GENERATOR_MARKER in prompt:
        return "generator"
    if UPDATER_MARKER in prompt:
        return "updater"
    return "unknown"


def find_one_file(run_dir: Path, pattern: str) -> Path:
    matches = sorted(run_dir.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {pattern} under {run_dir}, found {len(matches)}"
        )
    return matches[0]


def parse_game_seed_from_dir(path: Path) -> tuple[str, int] | None:
    match = re.search(r"(minigrid_[A-Za-z0-9_]+)-s(\d+)$", path.name)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def find_run_dir(
    source_root: Path,
    source_run_tag: str,
    game: str,
    seed: int,
) -> Path:
    exact = source_root / f"{game}-s{seed}"
    if exact.is_dir():
        return exact

    suffix = f"{game}-s{seed}"
    candidates = []
    for path in source_root.iterdir():
        if not path.is_dir() or not path.name.endswith(suffix):
            continue
        if source_run_tag and source_run_tag not in path.name:
            continue
        candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"No source run dir found for {game} seed={seed} under {source_root}"
        )
    candidates.sort(key=lambda p: (len(p.name), p.name))
    return candidates[0]


def load_playbook(items: list[dict[str, Any]]) -> Playbook:
    playbook = Playbook()
    playbook.items = []
    next_id = 1
    for raw in items:
        item_id = int(raw.get("id", next_id))
        playbook.items.append(
            PlaybookItem(
                id=item_id,
                content=str(raw.get("content", "")).strip(),
                helpful_count=int(raw.get("helpful", raw.get("helpful_count", 0))),
                harmful_count=int(raw.get("harmful", raw.get("harmful_count", 0))),
            )
        )
        next_id = max(next_id, item_id + 1)
    playbook._next_id = next_id
    return playbook


def playbook_to_plain(playbook: Playbook) -> list[dict[str, Any]]:
    return [asdict(item) for item in playbook.items]


def extract_finish_reason(payload: dict[str, Any]) -> str | None:
    try:
        return payload["response"]["choices"][0].get("finish_reason")
    except Exception:
        return None


def align_episode_updater_prompts(run_dir: Path) -> list[dict[str, Any]]:
    results_path = find_one_file(run_dir, "results_*.json")
    llm_path = find_one_file(run_dir, "llm_calls_*.jsonl")

    with results_path.open("r", encoding="utf-8") as handle:
        results = json.load(handle)
    episodes = results.get("logs") or []
    if not episodes:
        raise ValueError(f"No logs in {results_path}")

    calls: list[dict[str, Any]] = []
    with llm_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            payload = json.loads(raw_line)
            payload["_line_number"] = line_number
            payload["_kind"] = classify_call(payload)
            calls.append(payload)

    out = []
    pos = 0
    for episode_index, episode in enumerate(episodes, start=1):
        expected_generators = len(episode.get("generator_trace1") or [])
        generators = []
        for _ in range(expected_generators):
            if pos >= len(calls):
                raise ValueError(f"Missing generator call in {run_dir} episode={episode_index}")
            call = calls[pos]
            if call["_kind"] != "generator":
                raise ValueError(
                    f"Expected generator at line {call.get('_line_number')} "
                    f"in {run_dir}, got {call['_kind']}"
                )
            generators.append(call)
            pos += 1

        if pos >= len(calls):
            raise ValueError(f"Missing updater call in {run_dir} episode={episode_index}")
        updater = calls[pos]
        if updater["_kind"] != "updater":
            raise ValueError(
                f"Expected updater at line {updater.get('_line_number')} "
                f"in {run_dir}, got {updater['_kind']}"
            )
        pos += 1

        out.append(
            {
                "episode_index": episode_index,
                "episode": episode,
                "updater_prompt": str(updater.get("prompt", "")),
                "source_updater_line": updater.get("_line_number"),
                "source_updater_finish_reason": extract_finish_reason(updater),
                "num_source_generator_calls": len(generators),
            }
        )

    if pos != len(calls):
        print(
            f"[alignment warning] {run_dir}: consumed {pos} calls, "
            f"leftover {len(calls) - pos}",
            flush=True,
        )
    return out


def raw_reward_from_feedback(feedback: str, counted_reward: float) -> float:
    matches = re.findall(r"Raw MiniGrid reward:\s*([-+0-9.eE]+)", feedback)
    if matches:
        try:
            return float(matches[-1])
        except ValueError:
            pass
    return float(counted_reward)


def run_fixed_playbook_episode(
    *,
    env_id: str,
    max_steps: int,
    seed_offset: int,
    reset_seed: int,
    playbook: Playbook,
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
            prompt = build_generator_prompt_with_playbook(
                obs,
                playbook,
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
        "source_root": str(args.source_root),
        "source_run_tag": args.source_run_tag,
        "seeds": args.seeds,
        "samples_per_episode": args.samples_per_episode,
        "eval_episodes_per_sample": eval_episodes,
        "eval_episodes_default": args.eval_episodes,
        "long_eval_episodes": args.long_eval_episodes,
        "base_seed": args.base_seed,
        "updater_max_tokens": args.updater_max_tokens,
        "generator_max_tokens": 512,
    }
    write_json(run_dir / "config.json", config)
    ensure_csv_header(summary_path, SAMPLE_SUMMARY_FIELDS)

    expected_source_episodes = (
        len(args.seeds) * int(game_config["source_episodes_per_seed"])
    )
    total_samples = expected_source_episodes * args.samples_per_episode
    processed_samples = 0

    print(f"Game: {args.game}", flush=True)
    print(f"Output: {run_dir}", flush=True)
    print(f"Expected source episodes: {expected_source_episodes}", flush=True)
    print(f"Expected updater samples: {total_samples}", flush=True)
    print(f"Eval episodes per sample: {eval_episodes}", flush=True)

    client = build_client(args.server)
    started_game = time.time()

    for source_seed in args.seeds:
        source_dir = find_run_dir(
            Path(args.source_root),
            args.source_run_tag,
            args.game,
            source_seed,
        )
        episode_chunks = align_episode_updater_prompts(source_dir)
        if len(episode_chunks) != game_config["source_episodes_per_seed"]:
            raise ValueError(
                f"{source_dir}: expected {game_config['source_episodes_per_seed']} "
                f"episodes, found {len(episode_chunks)}"
            )

        for chunk in episode_chunks:
            episode = chunk["episode"]
            source_episode = int(episode.get("episode", chunk["episode_index"]))
            source_playbook_items = (
                episode.get("context_before_episode", {}).get("playbook") or []
            )
            source_playbook_size = len(source_playbook_items)

            for sample_index in range(args.samples_per_episode):
                sample_started = time.time()
                updater_output = call_lm(
                    client,
                    args.model,
                    chunk["updater_prompt"],
                    disable_thinking=args.disable_thinking,
                    max_tokens=args.updater_max_tokens,
                )
                deltas = _parse_delta_items(updater_output)
                parse_status = "parsed_json_or_fallback" if deltas else "no_delta_or_unparseable"

                updated_playbook = load_playbook(source_playbook_items)
                updated_playbook.apply_delta(deltas)
                updated_playbook_size = len(updated_playbook.items)

                sample_record = {
                    "game": args.game,
                    "env_id": game_config["env_id"],
                    "source_seed": source_seed,
                    "source_episode": source_episode,
                    "sample_index": sample_index,
                    "source_dir": str(source_dir),
                    "source_updater_line": chunk["source_updater_line"],
                    "source_updater_finish_reason": chunk["source_updater_finish_reason"],
                    "num_source_generator_calls": chunk["num_source_generator_calls"],
                    "source_playbook_size": source_playbook_size,
                    "num_delta_items": len(deltas),
                    "updated_playbook_size": updated_playbook_size,
                    "updater_parse_status": parse_status,
                    "updater_content_chars": len(updater_output),
                    "updater_output": updater_output,
                    "delta_items": [asdict(delta) for delta in deltas],
                    "updated_playbook": playbook_to_plain(updated_playbook),
                }
                append_jsonl(samples_path, sample_record)

                rollout_rows = []
                seed_offset = source_seed * int(args.seed_stride)
                for eval_index in range(eval_episodes):
                    reset_seed = int(args.base_seed) + eval_index
                    rollout = run_fixed_playbook_episode(
                        env_id=game_config["env_id"],
                        max_steps=int(game_config["max_steps"]),
                        seed_offset=seed_offset,
                        reset_seed=reset_seed,
                        playbook=updated_playbook,
                        client=client,
                        model=args.model,
                        disable_thinking=args.disable_thinking,
                    )
                    rollout_record = {
                        "game": args.game,
                        "env_id": game_config["env_id"],
                        "source_seed": source_seed,
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
                    "source_seed": source_seed,
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
                    "num_delta_items": len(deltas),
                    "updater_parse_status": parse_status,
                    "source_playbook_size": source_playbook_size,
                    "updated_playbook_size": updated_playbook_size,
                    "updater_content_chars": len(updater_output),
                    "elapsed_seconds": round(time.time() - sample_started, 3),
                }
                append_csv(summary_path, summary, SAMPLE_SUMMARY_FIELDS)
                append_tsv(
                    status_path,
                    [
                        args.game,
                        source_seed,
                        source_episode,
                        sample_index,
                        "done",
                        f"{summary['mean_original_reward']:.6g}",
                        f"{summary['success_rate']:.6g}",
                        len(deltas),
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
                        "num_delta_items",
                        "elapsed_seconds",
                    ],
                )

                processed_samples += 1
                print(
                    "PROGRESS "
                    f"game={args.game} "
                    f"processed_samples={processed_samples}/{total_samples} "
                    f"source_seed={source_seed} "
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
        description="Evaluate Qwen3-14B playbook-update stability on MiniGrid."
    )
    parser.add_argument("--game", required=True, choices=sorted(GAME_CONFIGS))
    parser.add_argument(
        "--source-root",
        default=os.environ.get(
            "SOURCE_ROOT",
            "/gscratch/h2lab/mohanc3/projects/ERL_repo/minigrid_sft",
        ),
        help="Directory containing completed ACE_ONCE run directories.",
    )
    parser.add_argument(
        "--source-run-tag",
        default=os.environ.get("SOURCE_RUN_TAG", "20260529_131951"),
        help="Substring required for tagged source directories; exact clean dirs ignore this.",
    )
    parser.add_argument(
        "--outputs-root",
        default=os.environ.get(
            "OUTPUTS_ROOT",
            "/gscratch/h2lab/mohanc3/projects/ERL_repo/minigrid_playbook_stability",
        ),
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--server", required=True)
    parser.add_argument("--samples-per-episode", type=int, default=4)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--long-eval-episodes", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=20260601)
    parser.add_argument("--seed-stride", type=int, default=10000)
    parser.add_argument("--updater-max-tokens", type=int, default=8192)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5],
        help="Source seed indices to process.",
    )
    parser.add_argument("--disable-thinking", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    evaluate_game(args)


if __name__ == "__main__":
    main()
