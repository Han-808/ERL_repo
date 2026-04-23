"""
notebook_minimal method for FrozenLake / Sokoban.

Port of appworld-context-updater/methods/notebook_minimal.py adapted
to the grid-game setting:

  * Agent makes ONE LM call per step.  Input = observation + notebook.
    Output = <reason>...</reason> block + one action in triple backticks.
  * After the episode, the updater makes ONE LM call to revise the
    notebook via line-numbered replace / insert_after / delete ops.
    Reflector and curator are merged into this single call — the
    "minimal" in the name.
  * The notebook is free-form markdown (not the structured Playbook of
    methods/ace.py) and accumulates across episodes.  It replaces the
    Playbook as the online context.

Ablation switches:
  initial_notebook = "default"  -> scaffold with section headers + hints
  initial_notebook = "empty"    -> section headers only

Two attempts are retained for schema compatibility with summarize_logs /
evaluate.py / print_episode_table, but attempt 2 is a no-op clone of
attempt 1 (this method makes exactly one attempt per episode; the
improvement column will always be 0).
"""

import json
import re

from common import (
    BaseMethod,
    build_client,
    call_lm,
    parse_action_single,
    summarize_logs,
)
from prompts import (
    build_notebook_agent_prompt,
    build_notebook_updater_prompt,
)


# ----------------------------------------------------------------------
# Initial notebooks
#
# Both variants are intentionally abstract — they do NOT reveal the
# meaning of grid symbols.  The whole point of the benchmark is that the
# agent (and the updater) must discover symbol semantics from feedback.
# ----------------------------------------------------------------------

DEFAULT_INITIAL_NOTEBOOK = """\
# Agent Notebook
This notebook contains knowledge accumulated from past episodes. Follow it when choosing actions. Entries are short single-line bullets grouped by section.

## Symbols
- Symbols (A, B, C, D, E, a, b) are abstract; their meaning must be inferred from environment feedback across episodes.

## Movement Rules
- Valid actions: Up, Down, Left, Right.
- Moving into the grid boundary keeps the agent in place.

## Strategies
- Always analyze the grid first, then pick the action whose predicted outcome is best.

## Pitfalls
- Some cells end the episode with reward 0; record which symbol was on such a cell so future episodes avoid it.

## Notes
"""

EMPTY_INITIAL_NOTEBOOK = """\
# Agent Notebook

## Symbols

## Movement Rules

## Strategies

## Pitfalls

## Notes
"""

INITIAL_NOTEBOOKS = {
    "default": DEFAULT_INITIAL_NOTEBOOK,
    "empty":   EMPTY_INITIAL_NOTEBOOK,
}


# ----------------------------------------------------------------------
# Line-numbered editing helpers (port of notebook_minimal.py in template)
# ----------------------------------------------------------------------

def number_lines(text: str) -> str:
    """Return notebook text with 1-indexed 'NNNN: ' line-number prefixes."""
    lines = text.split("\n")
    return "\n".join(f"{i + 1:04d}: {line}" for i, line in enumerate(lines))


def _op_line(op):
    return op.get("start_line", op.get("line", 0))


def _op_claimed_lines(op):
    t = op.get("type", "")
    if t in ("replace", "delete"):
        s, e = op.get("start_line", 0), op.get("end_line", 0)
        if s > 0 and e >= s:
            return set(range(s, e + 1))
    if t == "insert_after":
        ln = op.get("line", 0)
        if ln > 0:
            return {ln}
    return set()


def _reject_overlapping_ops(operations):
    """Earliest line wins on conflict.  Same behavior as AppWorld version."""
    claimed, kept = set(), []
    for op in sorted(operations, key=_op_line):
        claimed_here = _op_claimed_lines(op)
        if claimed_here & claimed:
            print(f"  Warning: skipping overlapping op {op}")
            continue
        claimed |= claimed_here
        kept.append(op)
    return kept


def apply_notebook_operations(notebook, operations):
    """
    Apply replace / insert_after / delete ops to a notebook.

    Line numbers refer to the ORIGINAL (pre-edit) notebook.  Overlapping
    operations are dropped, then remaining ops are applied bottom-up so
    earlier line numbers stay stable.
    """
    lines = notebook.split("\n")
    applied = []
    operations = _reject_overlapping_ops(operations)
    for op in sorted(operations, key=_op_line, reverse=True):
        t = op.get("type", "")
        try:
            if t == "replace":
                s, e = op["start_line"] - 1, op["end_line"] - 1
                if 0 <= s < len(lines) and 0 <= e < len(lines) and s <= e:
                    repl = op["content"].split("\n") if op["content"] else [""]
                    lines[s:e + 1] = repl
                    applied.append(op)
                    print(f"  Replaced lines {op['start_line']}-{op['end_line']}")
                else:
                    print(f"  Warning: replace range out of bounds: {op}")
            elif t == "insert_after":
                pos = op["line"]
                if 0 <= pos <= len(lines):
                    lines.insert(pos, op["content"])
                    applied.append(op)
                    print(f"  Inserted after line {op['line']}")
                else:
                    print(f"  Warning: insert_after line out of bounds: {op}")
            elif t == "delete":
                s, e = op["start_line"] - 1, op["end_line"] - 1
                if 0 <= s < len(lines) and 0 <= e < len(lines) and s <= e:
                    del lines[s:e + 1]
                    applied.append(op)
                    print(f"  Deleted lines {op['start_line']}-{op['end_line']}")
                else:
                    print(f"  Warning: delete range out of bounds: {op}")
            else:
                print(f"  Warning: unknown op type '{t}'")
        except (KeyError, TypeError) as exc:
            print(f"  Warning: malformed operation {op}: {exc}")
    return "\n".join(lines), applied


def extract_json_payload(raw: str) -> dict:
    """
    Pull a JSON object out of possibly-messy LM output.

    Tries ```json ... ``` fence first, then any ``` ... ``` fence, then
    falls back to the first brace-balanced {...} substring.
    """
    for pattern in (
        r"```json\s*(\{.*?\})\s*```",
        r"```\s*(\{.*?\})\s*```",
    ):
        m = re.search(pattern, raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

    depth, start = 0, -1
    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(raw[start:i + 1])
                except json.JSONDecodeError:
                    start = -1
    raise ValueError("no valid JSON object found in updater response")


def validate_operations(operations):
    """Filter malformed ops (same checks as the AppWorld version)."""
    if not isinstance(operations, list):
        print("[notebook_minimal] 'operations' is not a list, dropping all")
        return []
    filtered = []
    for i, op in enumerate(operations):
        if not isinstance(op, dict):
            print(f"  Skipping op {i}: not a dict")
            continue
        t = op.get("type", "")
        if t == "replace":
            if ("start_line" not in op or "end_line" not in op
                    or "content" not in op):
                print(f"  Skipping replace {i}: missing fields")
                continue
            if (not isinstance(op["end_line"], int)
                    or op["end_line"] < op["start_line"]):
                print(f"  Skipping replace {i}: bad end_line")
                continue
        elif t == "insert_after":
            if "line" not in op or "content" not in op:
                print(f"  Skipping insert_after {i}: missing fields")
                continue
            if "\n" in str(op["content"]):
                print(f"  Skipping insert_after {i}: content has newline")
                continue
        elif t == "delete":
            if "start_line" not in op or "end_line" not in op:
                print(f"  Skipping delete {i}: missing fields")
                continue
            if (not isinstance(op["end_line"], int)
                    or op["end_line"] < op["start_line"]):
                print(f"  Skipping delete {i}: bad end_line")
                continue
        else:
            print(f"  Skipping op {i}: unknown type '{t}'")
            continue
        filtered.append(op)
    return filtered


# ----------------------------------------------------------------------
# NotebookMinimalMethod
# ----------------------------------------------------------------------

class NotebookMinimalMethod(BaseMethod):
    """One LM call per step, one LM call per episode to update the notebook."""

    name = "notebook_minimal"

    def __init__(
        self,
        env,
        model: str = "qwen3-8b",
        server_url: str = "http://LOCAL_SERVER/v1",
        reward_threshold: float = 1.0,
        initial_notebook: str = "default",
        disable_thinking: bool = False,
    ):
        self.env = env
        self.model = model
        self.reward_threshold = reward_threshold
        self.initial_notebook = initial_notebook
        self.disable_thinking = disable_thinking
        self.notebook = self.initialize_context()
        self.client = build_client(server_url)

        if initial_notebook != "default":
            self.name = f"notebook_minimal_{initial_notebook}"

        print(f"Connected to LM server at {server_url}")
        print(f"Model: {self.model}")
        print(f"NotebookMinimal Method ready "
              f"(initial_notebook={initial_notebook}).")

    # -- BaseMethod contract -------------------------------------------

    def initialize_context(self) -> str:
        if self.initial_notebook not in INITIAL_NOTEBOOKS:
            print(f"[notebook_minimal] Unknown initial_notebook "
                  f"'{self.initial_notebook}', falling back to 'empty'.")
            return EMPTY_INITIAL_NOTEBOOK
        return INITIAL_NOTEBOOKS[self.initial_notebook]

    # -- Agent loop -----------------------------------------------------

    def _run_attempt(self):
        actions, feedbacks, reward = [], [], 0
        while not self.env.done:
            obs = self.env.get_observation()
            prompt = build_notebook_agent_prompt(obs, self.notebook)
            raw = call_lm(
                self.client, self.model, prompt,
                disable_thinking=self.disable_thinking,
            )
            action = parse_action_single(raw)
            actions.append(action)
            _, step_fb, reward, done = self.env.step([action])
            feedbacks.append(step_fb)
            if done:
                break
        return actions, " ".join(feedbacks), reward

    # -- Updater -------------------------------------------------------

    def _update_notebook(self, initial_obs, actions, feedback, reward):
        prompt = build_notebook_updater_prompt(
            numbered_notebook=number_lines(self.notebook),
            initial_obs=initial_obs,
            actions=actions,
            feedback=feedback,
            reward=reward,
            reward_threshold=self.reward_threshold,
        )
        raw = call_lm(
            self.client, self.model, prompt,
            disable_thinking=self.disable_thinking,
        )
        if not raw.strip():
            print("[notebook_minimal] empty updater response; no edit.")
            return ""
        try:
            payload = extract_json_payload(raw)
        except Exception as exc:
            print(f"[notebook_minimal] JSON parse failed: {exc}")
            return ""
        ops = validate_operations(payload.get("operations", []))
        reasoning = payload.get("reasoning", "")
        new_notebook, applied = apply_notebook_operations(self.notebook, ops)
        self.notebook = new_notebook
        if applied:
            print(f"[Notebook] {len(applied)} ops applied; now "
                  f"{len(self.notebook.splitlines())} lines.")
        else:
            print("[Notebook] no ops applied.")
        return reasoning

    # -- Episode loop ---------------------------------------------------

    def run_episode(self, episode_num: int) -> dict:
        initial_obs = self.env.reset(seed=episode_num)
        actions1, feedback1, reward1 = self._run_attempt()

        print(f"\n{'='*40}")
        print(f"=== Episode {episode_num} ===")
        print(f"{'='*40}")
        print(f"[Attempt 1] Actions:  {actions1}")
        print(f"[Attempt 1] Feedback: {feedback1}")
        print(f"[Attempt 1] Reward:   {reward1}")

        reasoning = self._update_notebook(
            initial_obs, actions1, feedback1, reward1
        )

        # Single-attempt method: mirror attempt 1 into attempt 2 so the
        # shared summarize_logs / print_episode_table schema stays valid.
        return {
            "episode":       episode_num,
            "actions1":      actions1,
            "feedback1":     feedback1,
            "reward1":       reward1,
            "reflection":    reasoning,
            "actions2":      actions1,
            "feedback2":     feedback1,
            "reward2":       reward1,
            "notebook_size": len(self.notebook.splitlines()),
            "gated":         True,
        }

    # -- Run over N episodes --------------------------------------------

    def run(self, n_episodes: int) -> dict:
        logs = []
        for ep in range(1, n_episodes + 1):
            logs.append(self.run_episode(ep))
        return summarize_logs(
            logs,
            self.reward_threshold,
            type(self.env).__name__,
            n_episodes,
        )
