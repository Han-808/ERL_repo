"""
Notebook data structure and line-editing helpers for the ACE Notebook pipeline.

The Notebook is a free-form markdown document that accumulates strategies
across episodes.  The updater (ace/notebook_updater.py) proposes edits as
line-numbered replace / insert_after / delete operations; this module
applies them deterministically.
"""


# ----------------------------------------------------------------------
# Initial notebook content
# ----------------------------------------------------------------------

INITIAL_NOTEBOOK = """\
# Agent Notebook
This notebook contains strategies accumulated from past grid navigation tasks.
It is provided at the start of every task to help you perform better.
Entries are single-line bullets grouped by section.

## Navigation Strategies
- Always map out the grid mentally before moving.
- Avoid positions adjacent to holes (C tiles in FrozenLake).

## Common Mistakes

## Successful Patterns

## Environment-Specific Rules
"""


# ----------------------------------------------------------------------
# Updater prompt template
# Uses {{ var }} placeholders filled by render_template() from common.py.
# ----------------------------------------------------------------------

UPDATER_PROMPT = """\
You are a notebook updater for a grid navigation agent playing FrozenLake or Sokoban.
Review the agent's trajectory and update the notebook with insights that will improve
the agent's future success rate on similar grid puzzles.

The notebook will be provided to the agent at the start of every future task.
You are evaluated on whether your edits improve that agent's success rate.

**Inputs:**

Task outcome (reward): {{ reward }}

Environment feedback from this episode:
<<<FEEDBACK>>>
{{ feedback }}
<<<END_FEEDBACK>>>

Actions taken by the agent:
<<<ACTIONS>>>
{{ actions }}
<<<END_ACTIONS>>>

Current notebook (line-numbered):
<<<NOTEBOOK>>>
{{ numbered_notebook }}
<<<END_NOTEBOOK>>>

**Editing guidelines:**
- Replace when an existing note is wrong or superseded.
- Insert when adding a genuinely new insight.
- Delete when a note is redundant or misleading.
- An empty operations list is valid when nothing new was learned.
- Be specific: reference grid symbols (A=player, B=goal/box, C=hole/goal, D=floor, E=wall).
- Do not add generic advice like "plan carefully" or "think step by step".

**Output format (return ONLY valid JSON in a single ```json block):**
```json
{
  "reasoning": "What happened, why, and what is worth remembering.",
  "operations": [
    {"type": "insert_after", "line": N, "content": "text"},
    {"type": "replace", "start_line": N, "end_line": M, "content": "text"},
    {"type": "delete", "start_line": N, "end_line": M}
  ]
}
```

Rules:
- insert_after: inserts a single line after line N. Use line 0 to insert at top.
- replace: replaces lines N through M inclusive with content (may contain newlines).
- delete: removes lines N through M inclusive.
- Line numbers refer to the ORIGINAL notebook shown above.
- content for insert_after must be a single line (no newlines).
"""


# ----------------------------------------------------------------------
# Line-numbered editing helpers
# ----------------------------------------------------------------------

def _number_lines(text: str) -> str:
    """Return notebook text with 1-indexed 'NNNN: ' line-number prefixes."""
    lines = text.split("\n")
    return "\n".join(f"{i + 1:04d}: {line}" for i, line in enumerate(lines))


def _op_line(op: dict) -> int:
    return op.get("start_line", op.get("line", 0))


def _op_claimed_lines(op: dict) -> set:
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


def _reject_overlapping_ops(operations: list) -> list:
    """Earliest line wins on conflict."""
    claimed, kept = set(), []
    for op in sorted(operations, key=_op_line):
        claimed_here = _op_claimed_lines(op)
        if claimed_here & claimed:
            print(f"  Warning: skipping overlapping op {op}")
            continue
        claimed |= claimed_here
        kept.append(op)
    return kept


def apply_notebook_operations(notebook: str, operations: list) -> tuple:
    """
    Apply replace / insert_after / delete ops to a notebook.

    Line numbers refer to the ORIGINAL (pre-edit) notebook.  Overlapping
    operations are dropped, then remaining ops are applied bottom-up so
    earlier line numbers stay stable.

    Returns (new_notebook_text, list_of_applied_ops).
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


# ----------------------------------------------------------------------
# Notebook class
# ----------------------------------------------------------------------

class Notebook:
    """Free-form markdown notebook that accumulates cross-episode knowledge."""

    def __init__(self):
        self.content: str = INITIAL_NOTEBOOK

    def numbered(self) -> str:
        """Return the notebook text with 1-indexed line-number prefixes."""
        return _number_lines(self.content)

    def to_string(self) -> str:
        return self.content

    def apply_updates(self, operations: list) -> list:
        """Apply a list of edit operations in-place; return applied ops."""
        self.content, applied = apply_notebook_operations(self.content, operations)
        return applied
