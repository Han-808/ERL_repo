"""
Shared utilities for every method in this repo.

Defines the BaseMethod contract, the shared LM call, the action parser,
the lightweight template renderer, result IO helpers, and experiment
summary/table formatting.
"""

import json
import os
import re
import time
from pathlib import Path

from openai import OpenAI


_VALID_ACTIONS = {"Up", "Down", "Left", "Right"}


# ----------------------------------------------------------------------
# BaseMethod contract
# ----------------------------------------------------------------------

class BaseMethod:
    """
    Abstract base class for an ACE-style method.

    Each concrete method exposes a stable `name`, builds its initial context
    (memory / playbook / notebook / summary), and runs a single episode
    returning a JSON-serializable log dict.
    """

    name: str = "base"

    def initialize_context(self):
        """Return the initial context object (memory, playbook, etc.)."""
        raise NotImplementedError

    def run_episode(self, episode_num: int) -> dict:
        """Run one full task cycle and return a log dict."""
        raise NotImplementedError


# ----------------------------------------------------------------------
# LM client + shared call helper
# ----------------------------------------------------------------------

def build_client(server_url: str) -> OpenAI:
    return OpenAI(base_url=server_url, api_key="EMPTY")


def _safe_model_dump(obj):
    """Best-effort conversion of OpenAI SDK objects into JSON data."""
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if hasattr(obj, "dict"):
        return obj.dict()
    if isinstance(obj, (dict, list, str, int, float, bool)):
        return obj
    return str(obj)


def _extract_reasoning_contents(response) -> list:
    """Collect reasoning fields exposed by SGLang/OpenAI-compatible responses."""
    reasoning = []
    for choice in getattr(response, "choices", []) or []:
        message = getattr(choice, "message", None)
        if message is None:
            continue
        for key in ("reasoning_content", "reasoning_contents"):
            value = getattr(message, key, None)
            if value:
                if isinstance(value, list):
                    reasoning.extend(value)
                else:
                    reasoning.append(value)
        extra = getattr(message, "model_extra", None) or {}
        if isinstance(extra, dict):
            for key in ("reasoning_content", "reasoning_contents"):
                value = extra.get(key)
                if value:
                    if isinstance(value, list):
                        reasoning.extend(value)
                    else:
                        reasoning.append(value)
    return reasoning


def _append_lm_trace(payload: dict) -> None:
    """Append one chat-completion trace to LLM_TRACE_PATH, if configured."""
    trace_path = os.environ.get("LLM_TRACE_PATH")
    if not trace_path:
        return
    try:
        path = Path(trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"[LM trace warning] could not write trace: {exc}")


def call_lm(client, model: str, prompt: str,
            disable_thinking: bool = False) -> str:
    """
    Send a prompt to the LM and return the response text.

    Identical parameters across ERL and ACE (max_tokens=512,
    temperature=0.7). For Qwen3-style chat templates, disable_thinking sends
    enable_thinking=False through SGLang's OpenAI-compatible API.
    Returns "" on failure.
    """
    messages = [{"role": "user", "content": prompt}]
    request_kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.7,
    }
    if disable_thinking:
        request_kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": False},
        }

    try:
        response = client.chat.completions.create(**request_kwargs)
        message = response.choices[0].message
        content = message.content or ""
        reasoning_contents = _extract_reasoning_contents(response)
        _append_lm_trace({
            "timestamp": int(time.time()),
            "model": model,
            "disable_thinking": disable_thinking,
            "prompt": prompt,
            "messages": messages,
            "request": request_kwargs,
            "response": _safe_model_dump(response),
            "content": content,
            "reasoning_content": (
                reasoning_contents[0] if reasoning_contents else None
            ),
            "reasoning_contents": reasoning_contents,
            "error": None,
        })
        return content
    except Exception as exc:
        print(f"[LM error] {exc}")
        _append_lm_trace({
            "timestamp": int(time.time()),
            "model": model,
            "disable_thinking": disable_thinking,
            "prompt": prompt,
            "messages": messages,
            "request": request_kwargs,
            "response": None,
            "content": "",
            "reasoning_content": None,
            "reasoning_contents": [],
            "error": str(exc),
        })
        return ""


# ----------------------------------------------------------------------
# Action parser
# ----------------------------------------------------------------------

def parse_action_single(lm_output: str) -> str:
    """
    Extract one action from the LM's output.

    Primary format: triple backticks, e.g. ```Down```.
    Fallback 1: any backtick-quoted token, e.g. `Down`.
    Fallback 2: first valid action word found scanning lines bottom-up.
    Fallback 3: "Down" if nothing matches.
    """
    m = re.search(r"```(\w+)```", lm_output)
    if m:
        action = m.group(1).strip().title()
        if action in _VALID_ACTIONS:
            return action

    m = re.search(r"`(\w+)`", lm_output)
    if m:
        action = m.group(1).strip().title()
        if action in _VALID_ACTIONS:
            return action

    for line in reversed(lm_output.strip().split("\n")):
        for action in ("Up", "Down", "Left", "Right"):
            if action in line:
                return action

    print("[Warning] Could not parse action; using fallback 'Down'.")
    return "Down"


# ----------------------------------------------------------------------
# Jinja-style template renderer (no control flow)
# ----------------------------------------------------------------------

def render_template(template: str, **kwargs) -> str:
    """
    Substitute {{ var }} placeholders with provided values.

    We avoid str.format because prompt text and runtime payloads may contain
    literal braces that would otherwise be misinterpreted.
    """
    out = template
    for key, value in kwargs.items():
        out = out.replace("{{ " + key + " }}", str(value))
        out = out.replace("{{" + key + "}}", str(value))
    return out


def format_delta_items(deltas) -> str:
    """
    Format a list of DeltaItem objects as a human-readable block.

    Accepts any object with .operation/.id/.content/.reason fields.
    """
    if not deltas:
        return "(none)"

    lines = []
    for d in deltas:
        if d.operation == "ADD":
            lines.append(f"[ADD] {d.content}")
        elif d.operation == "MODIFY":
            lines.append(f"[MODIFY] id={d.id} {d.content}")
        elif d.operation == "DELETE":
            lines.append(f"[DELETE] id={d.id}")
        if d.reason:
            lines.append(f"reason: {d.reason}")
        lines.append("")
    return "\n".join(lines).rstrip()


# ----------------------------------------------------------------------
# Results IO
# ----------------------------------------------------------------------

def results_path(outputs_dir: str, method_name: str, env_name: str) -> Path:
    return Path(outputs_dir) / f"results_{method_name}_{env_name}.json"


def write_results(path: Path, results: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def load_results(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------
# Shared log summarizer
# ----------------------------------------------------------------------

def summarize_logs(all_logs: list, reward_threshold: float,
                   env_name: str, n_episodes: int) -> dict:
    """
    Build the results dict returned by every method's run().

    Adds a running-average curve for the single online attempt.
    """
    def cum_rate() -> list:
        hits = 0
        out = []
        for k, lg in enumerate(all_logs, start=1):
            if lg["reward1"] >= reward_threshold:
                hits += 1
            out.append(hits / k)
        return out

    running_rate = cum_rate()
    rate = running_rate[-1] if running_rate else 0.0
    n_success = sum(1 for lg in all_logs if lg["reward1"] >= reward_threshold)

    print(f"\n{'=' * 40}")
    print(f"SUMMARY ({env_name}, {n_episodes} episodes)")
    print(f"{'=' * 40}")
    print(f"Success rate: {n_success}/{n_episodes} ({rate * 100:.1f}%)")

    if n_episodes >= 4:
        for frac in (0.25, 0.5, 0.75, 1.0):
            k = max(1, int(round(frac * n_episodes)))
            print(f"  running @ K={k:3d}: {running_rate[k - 1] * 100:.1f}%")

    return {
        "logs": all_logs,
        "pass_rate": rate,
        "attempt1_rate": rate,
        "running_pass_rate": running_rate,
        "running_attempt1_rate": running_rate,
    }


# ----------------------------------------------------------------------
# Episode-log pretty-printer
# ----------------------------------------------------------------------

def print_episode_table(logs: list, size_field: str = "memory_size",
                        size_header: str = "Memory Size") -> None:
    """Print a fixed-width per-episode statistics table from episode logs."""
    headers = ["Episode", "Reward", size_header]
    rows = [
        [lg["episode"], lg["reward1"], lg.get(size_field, "")]
        for lg in logs
    ]
    widths = [
        max(len(str(row[i])) for row in [headers] + rows)
        for i in range(len(headers))
    ]

    def row(cells):
        return " | ".join(str(c).ljust(w) for c, w in zip(cells, widths))

    print(row(headers))
    print("-+-".join("-" * w for w in widths))
    for cells in rows:
        print(row(cells))
