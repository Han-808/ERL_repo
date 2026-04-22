"""
Single-LM-call notebook updater for the ACE Notebook pipeline.

After each episode the updater receives the trajectory (actions, feedback,
reward) and the current notebook, and proposes line-numbered edits that
should improve the agent's future pass rate.
"""

import json
import re

from common import call_lm, render_template
from ace.notebook import Notebook, UPDATER_PROMPT


# ----------------------------------------------------------------------
# JSON extraction
# ----------------------------------------------------------------------

def _extract_json(raw: str) -> dict:
    """
    Extract a JSON object from LM output.

    Tries in order:
      1. Strip ```json ... ``` fences and parse
      2. Strip plain ``` ... ``` fences and parse
      3. Find the first brace-balanced { ... } substring and parse
      4. Return {"reasoning": "", "operations": []} on failure
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

    print("[notebook_updater] Could not extract JSON; returning empty payload.")
    return {"reasoning": "", "operations": []}


# ----------------------------------------------------------------------
# Operation validation
# ----------------------------------------------------------------------

def _validate_operations(operations) -> list:
    """Filter out malformed operations before applying them."""
    if not isinstance(operations, list):
        print("[notebook_updater] 'operations' is not a list; dropping all.")
        return []
    filtered = []
    for i, op in enumerate(operations):
        if not isinstance(op, dict):
            print(f"  Skipping op {i}: not a dict")
            continue
        t = op.get("type", "")
        if t == "replace":
            if "start_line" not in op or "end_line" not in op or "content" not in op:
                print(f"  Skipping replace {i}: missing fields")
                continue
            if not isinstance(op["end_line"], int) or op["end_line"] < op["start_line"]:
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
            if not isinstance(op["end_line"], int) or op["end_line"] < op["start_line"]:
                print(f"  Skipping delete {i}: bad end_line")
                continue
        else:
            print(f"  Skipping op {i}: unknown type '{t}'")
            continue
        filtered.append(op)
    return filtered


# ----------------------------------------------------------------------
# Public updater call
# ----------------------------------------------------------------------

def call_notebook_updater(
    client,
    model: str,
    notebook: Notebook,
    actions: list,
    feedback: str,
    reward: int,
) -> list:
    """
    Single LM call that analyzes the episode trajectory and returns
    a validated list of notebook edit operations.

    Steps:
      1. Render UPDATER_PROMPT with reward, feedback, actions, numbered_notebook.
      2. Call the LM (max_tokens=1024, temperature=0.7).
      3. Parse the response JSON with _extract_json().
      4. Validate the operations list with _validate_operations().
      5. Return the validated list (empty list on any error).
    """
    prompt = render_template(
        UPDATER_PROMPT,
        reward=reward,
        feedback=feedback,
        actions=str(actions),
        numbered_notebook=notebook.numbered(),
    )

    raw = call_lm(client, model, prompt)
    if not raw.strip():
        print("[notebook_updater] Empty LM response; no edits.")
        return []

    payload = _extract_json(raw)
    ops = _validate_operations(payload.get("operations", []))
    reasoning = payload.get("reasoning", "")
    if reasoning:
        print(f"[Updater reasoning] {reasoning[:200]}")
    return ops
