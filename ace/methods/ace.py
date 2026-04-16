"""
ACE core: Playbook bullet structure, delta updates, Reflector & Curator
roles, and the grow-and-refine de-duplication pass.

Paper: "Agentic Context Engineering: Evolving Contexts for
Self-Improving Language Models" — arXiv 2510.04618.

Sections referenced below:
  - Section 3.1  Incremental delta updates, bullet schema
  - Section 3.2  Grow-and-refine
  - Appendix D   Prompt templates
"""

import difflib
import re
from dataclasses import dataclass
from pathlib import Path

from ace.common import call_lm, format_delta_items


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------

@dataclass
class PlaybookItem:
    """
    A single 'bullet' in the playbook (Section 3.1).

    Fields mirror the paper's schema: unique integer id, counters for
    how often the entry has been marked helpful or harmful by the
    Generator, plus the content itself.
    """
    id: int
    content: str
    helpful_count: int = 0
    harmful_count: int = 0


@dataclass
class DeltaItem:
    """A proposed change to the playbook emitted by Reflector / Curator."""
    operation: str            # "ADD" | "MODIFY" | "DELETE"
    id: int                   # target bullet id; -1 for ADD
    content: str              # new text; "" for DELETE
    reason: str = ""          # explanation (for logging, not applied)


# ----------------------------------------------------------------------
# Playbook
# ----------------------------------------------------------------------

class Playbook:
    """
    Ordered collection of PlaybookItems with stable integer ids.

    The playbook is the evolving context described in the paper: the
    Generator reads it, the Reflector proposes deltas, the Curator
    approves, and apply_delta merges them in deterministically.
    """

    def __init__(self):
        self.items: list = []
        self._next_id: int = 1

    # -- Primitive mutations ------------------------------------------------

    def add(self, content: str) -> PlaybookItem:
        item = PlaybookItem(id=self._next_id, content=content.strip())
        self.items.append(item)
        self._next_id += 1
        return item

    def modify(self, item_id: int, new_content: str) -> bool:
        for it in self.items:
            if it.id == item_id:
                it.content = new_content.strip()
                return True
        return False

    def delete(self, item_id: int) -> bool:
        for i, it in enumerate(self.items):
            if it.id == item_id:
                self.items.pop(i)
                return True
        return False

    def mark_helpful(self, item_id: int):
        for it in self.items:
            if it.id == item_id:
                it.helpful_count += 1
                return

    def mark_harmful(self, item_id: int):
        for it in self.items:
            if it.id == item_id:
                it.harmful_count += 1
                return

    # -- Delta application --------------------------------------------------

    def apply_delta(self, deltas: list):
        """
        Apply a list of DeltaItems in order and print a short summary.

        Non-LLM, deterministic merge — this is the 'Curator merge'
        step from Section 3.1.
        """
        added = modified = deleted = skipped = 0
        for d in deltas:
            op = d.operation.upper()
            if op == "ADD":
                if d.content.strip():
                    self.add(d.content)
                    added += 1
                else:
                    skipped += 1
            elif op == "MODIFY":
                if self.modify(d.id, d.content):
                    modified += 1
                else:
                    skipped += 1
            elif op == "DELETE":
                if self.delete(d.id):
                    deleted += 1
                else:
                    skipped += 1
            else:
                skipped += 1

        print(
            f"[Playbook] apply_delta: +{added} add / "
            f"~{modified} modify / -{deleted} delete "
            f"({skipped} skipped)"
        )

    # -- Serialization ------------------------------------------------------

    def to_prompt_string(self) -> str:
        if not self.items:
            return "No strategies recorded yet."
        lines = []
        for it in self.items:
            lines.append(
                f"[{it.id}] (helpful:{it.helpful_count}, "
                f"harmful:{it.harmful_count}) {it.content}"
            )
        return "\n".join(lines)

    def to_dict(self) -> list:
        return [
            {
                "id": it.id,
                "content": it.content,
                "helpful": it.helpful_count,
                "harmful": it.harmful_count,
            }
            for it in self.items
        ]


# ----------------------------------------------------------------------
# Delta-item parser
# ----------------------------------------------------------------------

_OP_RE = re.compile(
    r"^\s*\[(ADD|MODIFY|DELETE)\]\s*(.*)$", re.IGNORECASE
)
_ID_RE = re.compile(r"id\s*=\s*(\d+)", re.IGNORECASE)
_REASON_RE = re.compile(r"^\s*reason\s*:\s*(.*)$", re.IGNORECASE)


def _parse_delta_items(text: str) -> list:
    """
    Parse the LM's delta output into a list of DeltaItem.

    Accepted shapes (case-insensitive):
      [ADD] <content>
      reason: <text>

      [MODIFY] id=3 <content>
      reason: <text>

      [DELETE] id=3
      reason: <text>

    Returns an empty list for [NO_CHANGE] or when no items are found.
    Never raises.
    """
    if not text:
        return []

    if "[NO_CHANGE]" in text.upper():
        return []

    deltas = []
    current = None
    lines = text.split("\n")

    for raw in lines:
        line = raw.rstrip()
        m = _OP_RE.match(line)
        if m:
            if current is not None:
                deltas.append(current)
            op = m.group(1).upper()
            rest = m.group(2).strip()

            target_id = -1
            content = ""

            if op in ("MODIFY", "DELETE"):
                id_match = _ID_RE.search(rest)
                if id_match:
                    target_id = int(id_match.group(1))
                    # Content is whatever remains after the id token.
                    content = _ID_RE.sub("", rest, count=1).strip()
                else:
                    # Missing id -> skip by starting no new current
                    current = None
                    continue
            else:  # ADD
                content = rest

            current = DeltaItem(
                operation=op,
                id=target_id,
                content=content,
                reason="",
            )
            continue

        rm = _REASON_RE.match(line)
        if rm and current is not None:
            current.reason = rm.group(1).strip()
            continue

        # continuation line for content (only for ADD/MODIFY with empty content)
        if (
            current is not None
            and current.operation in ("ADD", "MODIFY")
            and not current.content
            and line.strip()
        ):
            current.content = line.strip()

    if current is not None:
        deltas.append(current)

    # Filter out anything that ended up empty after parsing.
    cleaned = []
    for d in deltas:
        if d.operation == "DELETE" and d.id >= 0:
            cleaned.append(d)
        elif d.operation in ("ADD", "MODIFY") and d.content:
            if d.operation == "MODIFY" and d.id < 0:
                continue
            cleaned.append(d)
    return cleaned


# ----------------------------------------------------------------------
# Prompt loading
# ----------------------------------------------------------------------

def _load_instruction(instruction_path: str) -> str:
    return Path(instruction_path).read_text(encoding="utf-8")


# ----------------------------------------------------------------------
# Reflector  (Section 3, Appendix D)
# ----------------------------------------------------------------------

def run_reflector(
    lm_client,
    model: str,
    observation: str,
    actions: list,
    feedback: str,
    reward: int,
    playbook: Playbook,
    instruction_path: str,
) -> list:
    """
    Critique a trajectory and propose delta updates to the playbook.

    Mirrors the Reflector role in the ACE paper: read the execution
    trace, compare against the current context, and emit structured
    lessons for the Curator to approve.
    """
    template = _load_instruction(instruction_path)
    prompt = template.format(
        observation=observation,
        actions=actions,
        feedback=feedback,
        reward=reward,
        playbook=playbook.to_prompt_string(),
    )

    raw = call_lm(lm_client, model, prompt)
    print(f"\n[Reflector raw]\n{raw}")
    deltas = _parse_delta_items(raw)
    return deltas


# ----------------------------------------------------------------------
# Curator  (Section 3, Appendix D)
# ----------------------------------------------------------------------

def run_curator(
    lm_client,
    model: str,
    delta_items: list,
    playbook: Playbook,
    instruction_path: str,
) -> list:
    """
    Filter / refine the Reflector's proposals.

    Matches the Curator role: reject vague or redundant entries, merge
    overlapping ones, and return the final approved delta list.
    """
    if not delta_items:
        return []

    template = _load_instruction(instruction_path)
    prompt = template.format(
        playbook=playbook.to_prompt_string(),
        delta_items=format_delta_items(delta_items),
    )

    raw = call_lm(lm_client, model, prompt)
    print(f"\n[Curator raw]\n{raw}")
    approved = _parse_delta_items(raw)
    return approved


# ----------------------------------------------------------------------
# Grow-and-refine  (Section 3.2)
# ----------------------------------------------------------------------

def grow_and_refine(playbook: Playbook, similarity_threshold: float = 0.85):
    """
    De-duplicate semantically near-identical bullets.

    The paper uses embeddings; we use difflib's sequence ratio so the
    pipeline has no external dependencies.  For each near-duplicate
    pair, the bullet with the higher helpful_count survives; the other
    is deleted.
    """
    if len(playbook.items) < 2:
        return

    merged = 0
    # Walk a snapshot so deletion during iteration is safe.
    checked = set()
    i = 0
    while i < len(playbook.items):
        a = playbook.items[i]
        j = i + 1
        while j < len(playbook.items):
            b = playbook.items[j]
            key = (min(a.id, b.id), max(a.id, b.id))
            if key in checked:
                j += 1
                continue
            checked.add(key)

            ratio = difflib.SequenceMatcher(
                None, a.content.lower(), b.content.lower()
            ).ratio()

            if ratio > similarity_threshold:
                # Keep the one with the higher helpful_count (ties -> a).
                if b.helpful_count > a.helpful_count:
                    playbook.delete(a.id)
                    merged += 1
                    # a has been removed; restart inner scan at same i
                    a = playbook.items[i] if i < len(playbook.items) else None
                    if a is None:
                        break
                    j = i + 1
                    continue
                else:
                    playbook.delete(b.id)
                    merged += 1
                    # don't advance j: the list shrank by one
                    continue
            j += 1
        i += 1

    if merged:
        print(f"[Playbook] grow_and_refine merged {merged} near-duplicate entries.")
    else:
        print("[Playbook] grow_and_refine: no duplicates found.")
