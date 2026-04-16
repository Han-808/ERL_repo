"""
Shared utilities for the ACE pipeline.

Mirrors the LM-call / action-parsing patterns in erl_pipeline.py so that
ACE and ERL can run side-by-side against the same environments.
"""

import re


_VALID_ACTIONS = {"up", "down", "left", "right"}


def call_lm(client, model: str, prompt: str) -> str:
    """
    Send a prompt to the LM and return the response text.

    Identical parameters to ERLPipeline._call_lm (max_tokens=1024,
    temperature=0.7).  Returns "" on failure.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[LM error] {e}")
        return ""


def parse_actions(lm_output: str) -> list:
    """
    Pull the last Python-list-looking line from the LM output.

    Returns only tokens in _VALID_ACTIONS.  Falls back to ["down", "right"]
    if nothing parses.
    """
    for line in reversed(lm_output.strip().split("\n")):
        if "[" in line and "]" in line:
            inside = line[line.index("[") + 1 : line.rindex("]")]
            tokens = re.findall(r"[a-zA-Z]+", inside)
            actions = [t.lower() for t in tokens if t.lower() in _VALID_ACTIONS]
            if actions:
                return actions
    return ["down", "right"]


def format_delta_items(deltas) -> str:
    """
    Format a list of DeltaItem objects as a human-readable block for the
    Curator prompt.  Accepts any object with .operation/.id/.content/.reason.
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
