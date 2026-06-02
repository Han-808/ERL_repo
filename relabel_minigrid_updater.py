#!/usr/bin/env python3
"""Offline relabel ACE_ONCE MiniGrid updater calls.

This helper reads completed MiniGrid ACE_ONCE runs, extracts the saved merged
Reflector+Curator prompt for one episode, and sends it to an OpenAI-compatible
SGLang server with a larger completion budget.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import fcntl  # type: ignore
except ImportError:  # pragma: no cover - Windows local smoke only.
    fcntl = None


DEFAULT_SOURCE_ROOT = Path("/gscratch/h2lab/mohanc3/projects/ERL_repo/minigrid_sft")
DEFAULT_OUTPUT_ROOT = Path(
    "/gscratch/h2lab/mohanc3/projects/ERL_repo/minigrid_updater_relabel"
)
DEFAULT_SOURCE_RUN_TAG = "20260529_131951"
DEFAULT_MODEL = "Qwen/Qwen3.5-27B"
DEFAULT_MAX_TOKENS = 4096

GENERATOR_MARKER = "You are the Generator in an ACE"
UPDATER_MARKER = "Your job has two parts, done in a single pass"

GAME_SPECS = (
    ("minigrid_empty_random_5x5", "MiniGrid-Empty-Random-5x5-v0", 40),
    ("minigrid_memorys11", "MiniGrid-MemoryS11-v0", 40),
    ("minigrid_memorys13", "MiniGrid-MemoryS13-v0", 30),
    ("minigrid_fourrooms", "MiniGrid-FourRooms-v0", 20),
    ("minigrid_simplecrossings9n3", "MiniGrid-SimpleCrossingS9N3-v0", 20),
    ("minigrid_distshift1", "MiniGrid-DistShift1-v0", 20),
)

TOTAL_TASKS = sum(6 * episodes for _, _, episodes in GAME_SPECS)
STATUS_HEADER = (
    "task_id\tgame\tseed\tepisode\tstatus\tfinish_reason\tcompletion_tokens\t"
    "output_file\terror\n"
)


@dataclass(frozen=True)
class Assignment:
    task_id: int
    game: str
    env_id: str
    seed: int
    episode: int
    episodes_per_seed: int


@dataclass
class ClassifiedCall:
    index: int
    kind: str
    payload: dict[str, Any]


@dataclass
class EpisodePrompt:
    assignment: Assignment
    run_dir: Path
    results_path: Path
    llm_calls_path: Path
    updater_call: ClassifiedCall
    generator_call_count: int
    source_call_count: int
    skipped_leading_calls: int


def task_assignments() -> list[Assignment]:
    assignments: list[Assignment] = []
    task_id = 0
    for game, env_id, episodes in GAME_SPECS:
        for seed in range(6):
            for episode in range(1, episodes + 1):
                assignments.append(
                    Assignment(
                        task_id=task_id,
                        game=game,
                        env_id=env_id,
                        seed=seed,
                        episode=episode,
                        episodes_per_seed=episodes,
                    )
                )
                task_id += 1
    return assignments


ASSIGNMENTS = task_assignments()


def get_assignment(task_id: int) -> Assignment:
    if task_id < 0 or task_id >= len(ASSIGNMENTS):
        raise ValueError(f"task_id must be 0..{len(ASSIGNMENTS) - 1}, got {task_id}")
    return ASSIGNMENTS[task_id]


def single_file(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {pattern!r} under {root}, found {len(matches)}"
        )
    return matches[0]


def find_run_dir(source_root: Path, source_run_tag: str, game: str, seed: int) -> Path:
    exact = source_root / f"{game}-s{seed}"
    if exact.is_dir():
        return exact

    candidates = [
        path
        for path in source_root.iterdir()
        if path.is_dir()
        and source_run_tag in path.name
        and path.name.endswith(f"{game}-s{seed}")
    ]
    if not candidates:
        candidates = [
            path
            for path in source_root.iterdir()
            if path.is_dir() and path.name.endswith(f"{game}-s{seed}")
        ]

    valid = []
    for candidate in candidates:
        if list(candidate.glob("results_*.json")) and list(candidate.glob("llm_calls_*.jsonl")):
            valid.append(candidate)

    if len(valid) != 1:
        names = ", ".join(path.name for path in valid[:10])
        raise FileNotFoundError(
            f"Expected one run dir for {game}-s{seed}, found {len(valid)}: {names}"
        )
    return valid[0]


def classify_call(index: int, payload: dict[str, Any]) -> ClassifiedCall:
    prompt = str(payload.get("prompt") or "")
    if UPDATER_MARKER in prompt:
        return ClassifiedCall(index=index, kind="updater", payload=payload)
    if GENERATOR_MARKER in prompt:
        return ClassifiedCall(index=index, kind="generator", payload=payload)
    return ClassifiedCall(index=index, kind="unknown", payload=payload)


def stream_classified_calls(llm_calls_path: Path) -> list[ClassifiedCall]:
    calls: list[ClassifiedCall] = []
    with llm_calls_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            payload = json.loads(line)
            call = classify_call(index, payload)
            if call.kind == "unknown":
                raise ValueError(f"Unknown call type at {llm_calls_path}:{index + 1}")
            calls.append(call)
    return calls


def expected_call_count(episodes: Iterable[dict[str, Any]]) -> int:
    return sum(len(episode.get("generator_trace1") or []) + 1 for episode in episodes)


def aligned_calls(
    calls: list[ClassifiedCall], episodes: list[dict[str, Any]], run_dir: Path
) -> tuple[list[ClassifiedCall], int]:
    expected = expected_call_count(episodes)
    if len(calls) < expected:
        raise ValueError(f"{run_dir.name}: only {len(calls)} calls, expected {expected}")
    skipped = len(calls) - expected
    if skipped:
        calls = calls[skipped:]
    return calls, skipped


def extract_episode_prompt(
    source_root: Path, source_run_tag: str, assignment: Assignment
) -> EpisodePrompt:
    run_dir = find_run_dir(source_root, source_run_tag, assignment.game, assignment.seed)
    results_path = single_file(run_dir, "results_*.json")
    llm_calls_path = single_file(run_dir, "llm_calls_*.jsonl")

    result_data = json.loads(results_path.read_text(encoding="utf-8"))
    episodes = result_data.get("logs") or []
    if len(episodes) != assignment.episodes_per_seed:
        raise ValueError(
            f"{run_dir.name}: found {len(episodes)} episodes, "
            f"expected {assignment.episodes_per_seed}"
        )

    calls, skipped = aligned_calls(stream_classified_calls(llm_calls_path), episodes, run_dir)
    call_pos = 0
    for episode_number, episode in enumerate(episodes, start=1):
        gen_count = len(episode.get("generator_trace1") or [])
        for _ in range(gen_count):
            call = calls[call_pos]
            call_pos += 1
            if call.kind != "generator":
                raise ValueError(
                    f"{run_dir.name} episode {episode_number}: expected generator, "
                    f"got {call.kind} at source index {call.index}"
                )
        if call_pos >= len(calls):
            raise ValueError(f"{run_dir.name} episode {episode_number}: missing updater")
        updater = calls[call_pos]
        call_pos += 1
        if updater.kind != "updater":
            raise ValueError(
                f"{run_dir.name} episode {episode_number}: expected updater, "
                f"got {updater.kind} at source index {updater.index}"
            )
        if episode_number == assignment.episode:
            return EpisodePrompt(
                assignment=assignment,
                run_dir=run_dir,
                results_path=results_path,
                llm_calls_path=llm_calls_path,
                updater_call=updater,
                generator_call_count=gen_count,
                source_call_count=len(calls) + skipped,
                skipped_leading_calls=skipped,
            )

    raise ValueError(f"Episode not found: {assignment}")


def request_messages_from_old_call(call: dict[str, Any]) -> list[dict[str, str]]:
    request = call.get("request") or {}
    messages = request.get("messages") or call.get("messages")
    if messages:
        return messages
    prompt = str(call.get("prompt") or "")
    if not prompt:
        raise ValueError("Updater call has no messages or prompt")
    return [{"role": "user", "content": prompt}]


def prompt_hash(messages: list[dict[str, str]]) -> str:
    encoded = json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def chat_completion(
    server_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout_seconds: int,
) -> tuple[dict[str, Any], float]:
    endpoint = server_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {endpoint}: {detail}") from exc
    return json.loads(raw), time.time() - started


def response_content(response: dict[str, Any]) -> str:
    try:
        return response["choices"][0]["message"].get("content") or ""
    except Exception:
        return ""


def response_finish_reason(response: dict[str, Any]) -> str:
    try:
        return response["choices"][0].get("finish_reason") or ""
    except Exception:
        return ""


def response_completion_tokens(response: dict[str, Any]) -> int | str:
    try:
        value = response.get("usage", {}).get("completion_tokens")
        return "" if value is None else int(value)
    except Exception:
        return ""


def output_path_for(output_root: Path, assignment: Assignment) -> Path:
    return (
        output_root
        / "raw_updater_calls"
        / f"{assignment.game}-s{assignment.seed}-ep{assignment.episode:03d}.json"
    )


def append_status(
    status_file: Path,
    assignment: Assignment,
    status: str,
    finish_reason: str,
    completion_tokens: int | str,
    output_file: Path | str,
    error: str = "",
) -> None:
    status_file.parent.mkdir(parents=True, exist_ok=True)
    with status_file.open("a+", encoding="utf-8") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        if not handle.read(1):
            handle.write(STATUS_HEADER)
        handle.seek(0, 2)
        row = [
            str(assignment.task_id),
            assignment.game,
            str(assignment.seed),
            str(assignment.episode),
            status,
            finish_reason,
            str(completion_tokens),
            str(output_file),
            error.replace("\t", " ").replace("\n", " ")[:500],
        ]
        handle.write("\t".join(row) + "\n")
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{time.time_ns()}")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def run_task(args: argparse.Namespace) -> None:
    assignment = get_assignment(args.task_id)
    output_root = Path(args.output_root) / args.run_tag
    status_file = output_root / "status.tsv"
    output_file = output_path_for(output_root, assignment)

    try:
        episode_prompt = extract_episode_prompt(
            Path(args.source_root), args.source_run_tag, assignment
        )
        messages = request_messages_from_old_call(episode_prompt.updater_call.payload)
        new_response, duration = chat_completion(
            args.server_url,
            args.model,
            messages,
            args.max_tokens,
            args.timeout_seconds,
        )

        old_response = episode_prompt.updater_call.payload.get("response") or {}
        old_usage = old_response.get("usage") or {}
        payload = {
            "task_id": assignment.task_id,
            "game": assignment.game,
            "env_id": assignment.env_id,
            "seed": assignment.seed,
            "episode": assignment.episode,
            "source_run_dir": str(episode_prompt.run_dir),
            "source_results_path": str(episode_prompt.results_path),
            "source_llm_calls_path": str(episode_prompt.llm_calls_path),
            "source_updater_call_index": episode_prompt.updater_call.index,
            "source_prompt_hash": prompt_hash(messages),
            "source_skipped_leading_calls": episode_prompt.skipped_leading_calls,
            "source_generator_call_count": episode_prompt.generator_call_count,
            "source_call_count": episode_prompt.source_call_count,
            "source_finish_reason": response_finish_reason(old_response),
            "source_prompt_tokens": old_usage.get("prompt_tokens"),
            "source_completion_tokens": old_usage.get("completion_tokens"),
            "request": {
                "model": args.model,
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "server_url": args.server_url,
            },
            "duration_seconds": round(duration, 3),
            "response": new_response,
            "content": response_content(new_response),
            "finish_reason": response_finish_reason(new_response),
            "usage": new_response.get("usage") or {},
        }
        write_json_atomic(output_file, payload)
        append_status(
            status_file,
            assignment,
            "done",
            payload["finish_reason"],
            response_completion_tokens(new_response),
            output_file,
        )
        print(
            f"done task_id={assignment.task_id} game={assignment.game} "
            f"seed={assignment.seed} episode={assignment.episode} "
            f"finish_reason={payload['finish_reason']} output={output_file}"
        )
    except Exception as exc:
        append_status(status_file, assignment, "failed", "", "", output_file, str(exc))
        raise


def usage_prompt_tokens(call: dict[str, Any]) -> int | None:
    try:
        value = call.get("response", {}).get("usage", {}).get("prompt_tokens")
        return None if value is None else int(value)
    except Exception:
        return None


def finish_reason(call: dict[str, Any]) -> str:
    try:
        return call.get("response", {}).get("choices", [{}])[0].get("finish_reason") or ""
    except Exception:
        return ""


def audit_token_range(args: argparse.Namespace) -> None:
    source_root = Path(args.source_root)
    prompt_tokens: list[int] = []
    prompt_chars: list[int] = []
    finish_counts: Counter[str] = Counter()
    updater_count = 0
    skipped_total = 0

    for assignment in ASSIGNMENTS:
        if assignment.episode != 1:
            continue
        run_dir = find_run_dir(source_root, args.source_run_tag, assignment.game, assignment.seed)
        results_path = single_file(run_dir, "results_*.json")
        llm_calls_path = single_file(run_dir, "llm_calls_*.jsonl")
        result_data = json.loads(results_path.read_text(encoding="utf-8"))
        episodes = result_data.get("logs") or []
        calls, skipped = aligned_calls(stream_classified_calls(llm_calls_path), episodes, run_dir)
        skipped_total += skipped
        for call in calls:
            if call.kind != "updater":
                continue
            updater_count += 1
            finish_counts[finish_reason(call.payload)] += 1
            tokens = usage_prompt_tokens(call.payload)
            if tokens is not None:
                prompt_tokens.append(tokens)
            chars = call.payload.get("prompt_chars")
            if chars is not None:
                prompt_chars.append(int(chars))

    def pct(values: list[int], q: float) -> int:
        if not values:
            return 0
        values = sorted(values)
        idx = min(len(values) - 1, max(0, round((len(values) - 1) * q)))
        return values[idx]

    print(f"updater_calls={updater_count} / {TOTAL_TASKS}")
    print(f"skipped_leading_calls_total={skipped_total}")
    if prompt_tokens:
        print(
            "prompt_tokens "
            f"min={min(prompt_tokens)} "
            f"p50={int(statistics.median(prompt_tokens))} "
            f"p95={pct(prompt_tokens, 0.95)} "
            f"max={max(prompt_tokens)}"
        )
    if prompt_chars:
        print(
            "prompt_chars "
            f"min={min(prompt_chars)} "
            f"p50={int(statistics.median(prompt_chars))} "
            f"p95={pct(prompt_chars, 0.95)} "
            f"max={max(prompt_chars)}"
        )
    print("finish_reason_counts:")
    for key, value in sorted(finish_counts.items()):
        print(f"  {key or '<missing>'}: {value}")


def list_assignments() -> None:
    for assignment in ASSIGNMENTS:
        print(
            f"{assignment.task_id}\t{assignment.game}\t{assignment.env_id}\t"
            f"seed={assignment.seed}\tepisode={assignment.episode}/"
            f"{assignment.episodes_per_seed}"
        )
    print(f"total_tasks={len(ASSIGNMENTS)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--source-run-tag", default=DEFAULT_SOURCE_RUN_TAG)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-tag", default="minigrid-updater-relabel")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--server-url")
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--audit-token-range", action="store_true")
    parser.add_argument("--list-assignments", action="store_true")
    args = parser.parse_args()

    modes = sum(
        [
            bool(args.audit_token_range),
            bool(args.list_assignments),
            args.task_id is not None,
        ]
    )
    if modes != 1:
        parser.error("choose exactly one of --audit-token-range, --list-assignments, --task-id")
    if args.task_id is not None and not args.server_url:
        parser.error("--task-id requires --server-url")
    return args


def main() -> None:
    args = parse_args()
    if args.list_assignments:
        list_assignments()
    elif args.audit_token_range:
        audit_token_range(args)
    else:
        run_task(args)


if __name__ == "__main__":
    main()
