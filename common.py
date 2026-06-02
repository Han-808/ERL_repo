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


DEFAULT_VALID_ACTIONS = ("Up", "Down", "Left", "Right")
DEFAULT_ACTION = "Down"


def valid_actions_for_env(env=None) -> tuple[str, ...]:
    """Return the action tokens an environment expects from the LM."""
    actions = getattr(env, "ACTIONS", DEFAULT_VALID_ACTIONS)
    return tuple(str(action) for action in actions)


def default_action_for_env(env=None) -> str:
    """Return the parser fallback action for an environment."""
    return str(getattr(env, "DEFAULT_ACTION", DEFAULT_ACTION))


def action_example_for_env(env=None) -> str:
    """Return the example action token to show in prompts."""
    actions = valid_actions_for_env(env)
    return str(getattr(env, "ACTION_EXAMPLE", actions[0]))


def format_action_set(valid_actions=None) -> str:
    """Format action tokens for prompt text."""
    actions = valid_actions or DEFAULT_VALID_ACTIONS
    return ", ".join(str(action) for action in actions)


def env_metadata(env) -> dict:
    """Return stable metadata useful for downstream RL data extraction."""
    return {
        "env_class": type(env).__name__,
        "env_id": getattr(env, "env_id", type(env).__name__),
        "valid_actions": list(valid_actions_for_env(env)),
    }


def success_from_reward(reward, reward_threshold: float) -> bool:
    return reward >= reward_threshold


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


def call_lm(
    client,
    model: str,
    prompt: str,
    disable_thinking: bool = False,
    max_tokens: int = 512,
) -> str:
    """
    Send a prompt to the LM and return the response text.

    Defaults match historical generator behavior (max_tokens=512,
    temperature=0.7). Updater-style calls can pass a larger max_tokens value.
    For Qwen3-style chat templates, disable_thinking sends enable_thinking=False
    through SGLang's OpenAI-compatible API.
    Returns "" on failure.
    """
    messages = [{"role": "user", "content": prompt}]
    request_kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    if disable_thinking:
        request_kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": False},
        }

    started_at = time.time()
    try:
        response = client.chat.completions.create(**request_kwargs)
        finished_at = time.time()
        message = response.choices[0].message
        content = message.content or ""
        reasoning_contents = _extract_reasoning_contents(response)
        _append_lm_trace({
            "timestamp": int(finished_at),
            "started_at": int(started_at),
            "duration_seconds": round(finished_at - started_at, 3),
            "model": model,
            "disable_thinking": disable_thinking,
            "prompt_chars": len(prompt),
            "prompt": prompt,
            "messages": messages,
            "request": request_kwargs,
            "response": _safe_model_dump(response),
            "content_chars": len(content),
            "content": content,
            "reasoning_content": (
                reasoning_contents[0] if reasoning_contents else None
            ),
            "reasoning_contents": reasoning_contents,
            "error": None,
        })
        return content
    except Exception as exc:
        finished_at = time.time()
        print(f"[LM error] {exc}")
        _append_lm_trace({
            "timestamp": int(finished_at),
            "started_at": int(started_at),
            "duration_seconds": round(finished_at - started_at, 3),
            "model": model,
            "disable_thinking": disable_thinking,
            "prompt_chars": len(prompt),
            "prompt": prompt,
            "messages": messages,
            "request": request_kwargs,
            "response": None,
            "content_chars": 0,
            "content": "",
            "reasoning_content": None,
            "reasoning_contents": [],
            "error": str(exc),
        })
        return ""


# ----------------------------------------------------------------------
# Action parser
# ----------------------------------------------------------------------

def _normalize_action_token(token: str) -> str:
    return str(token).strip().strip("`'\".,:;()[]{}").casefold()


def _match_action_token(token: str, valid_actions: tuple[str, ...]) -> str | None:
    lookup = {
        _normalize_action_token(action): action
        for action in valid_actions
    }
    return lookup.get(_normalize_action_token(token))


def parse_action_single_with_status(
    lm_output: str,
    valid_actions=None,
    default_action: str | None = None,
) -> tuple[str, bool]:
    """
    Extract one action from the LM's output and report parse success.

    Primary format: triple backticks, e.g. ```Down``` or ```forward```.
    Fallback 1: any backtick-quoted token, e.g. `Down`.
    Fallback 2: first valid action word found scanning lines bottom-up.
    Fallback 3: the supplied default action, or "Down" for legacy grids.

    Returns (action, parsed_ok). parsed_ok is False only when fallback action
    had to be used because no valid action token appeared in lm_output.
    """
    actions = tuple(str(action) for action in (valid_actions or DEFAULT_VALID_ACTIONS))
    fallback = default_action or DEFAULT_ACTION
    if _match_action_token(fallback, actions) is None:
        fallback = actions[0]

    m = re.search(
        r"```\s*(?:[A-Za-z_][\w-]*\s*\n)?\s*([\w-]+)\s*```",
        lm_output,
    )
    if m:
        action = _match_action_token(m.group(1), actions)
        if action is not None:
            return action, True

    m = re.search(r"`\s*([\w-]+)\s*`", lm_output)
    if m:
        action = _match_action_token(m.group(1), actions)
        if action is not None:
            return action, True

    for line in reversed(lm_output.strip().split("\n")):
        for action in sorted(actions, key=len, reverse=True):
            pattern = r"\b" + re.escape(action) + r"\b"
            if re.search(pattern, line, re.IGNORECASE):
                return action, True

    print(f"[Warning] Could not parse action; using fallback '{fallback}'.")
    return fallback, False


def parse_action_single(
    lm_output: str,
    valid_actions=None,
    default_action: str | None = None,
) -> str:
    """Extract one action from the LM's output, falling back on parse failure."""
    action, _ = parse_action_single_with_status(
        lm_output,
        valid_actions=valid_actions,
        default_action=default_action,
    )
    return action


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
        "rl_training_schema_version": 1,
        "trajectory_event_count": sum(
            len(lg.get("trajectory_events", [])) for lg in all_logs
        ),
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
