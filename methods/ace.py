"""
ACE (Agentic Context Engineering) method — BaseMethod-compatible.

Paper: "Agentic Context Engineering: Evolving Contexts for
Self-Improving Language Models" — arXiv 2510.04618.

Consolidates the Playbook bullet structure, delta updates, Reflector
and Curator roles, the grow-and-refine de-duplication pass, and the
two-attempt episode loop into a single module — mirroring the way
methods/ace.py is organized in the appworld-context-updater template.

Sections referenced below:
  - Section 3.1  Incremental delta updates, bullet schema
  - Section 3.2  Grow-and-refine
  - Appendix D   Prompt templates
"""

import difflib
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from common import (
    BaseMethod,
    build_client,
    call_lm,
    format_delta_items,
    parse_action_single,
    render_template,
    summarize_logs,
)
from prompts import build_attempt1_prompt


_INSTRUCTIONS_DIR = Path(__file__).resolve().parents[1] / "instructions"


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

    def apply_delta(self, deltas: list):
        """Apply a list of DeltaItems in order and print a short summary."""
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

_OP_RE = re.compile(r"^\s*\[(ADD|MODIFY|DELETE)\]\s*(.*)$", re.IGNORECASE)
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

    for raw in text.split("\n"):
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
                    content = _ID_RE.sub("", rest, count=1).strip()
                else:
                    current = None
                    continue
            else:  # ADD
                content = rest

            current = DeltaItem(operation=op, id=target_id, content=content, reason="")
            continue

        rm = _REASON_RE.match(line)
        if rm and current is not None:
            current.reason = rm.group(1).strip()
            continue

        if (
            current is not None
            and current.operation in ("ADD", "MODIFY")
            and not current.content
            and line.strip()
        ):
            current.content = line.strip()

    if current is not None:
        deltas.append(current)

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


_GENERATOR_TEMPLATE = _load_instruction(_INSTRUCTIONS_DIR / "instruction_generator.md")


def build_attempt2_prompt_with_playbook(observation: str, playbook: Playbook) -> str:
    """Per-step Generator prompt for attempt 2, with the full Playbook injected."""
    return render_template(
        _GENERATOR_TEMPLATE,
        playbook=playbook.to_prompt_string(),
        observation=observation,
    )


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
    """Critique a trajectory and propose delta updates to the playbook."""
    template = _load_instruction(instruction_path)
    prompt = render_template(
        template,
        observation=observation,
        actions=actions,
        feedback=feedback,
        reward=reward,
        playbook=playbook.to_prompt_string(),
    )
    raw = call_lm(lm_client, model, prompt)
    print(f"\n[Reflector raw]\n{raw}")
    return _parse_delta_items(raw)


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
    """Filter / refine the Reflector's proposals."""
    if not delta_items:
        return []
    template = _load_instruction(instruction_path)
    prompt = render_template(
        template,
        playbook=playbook.to_prompt_string(),
        delta_items=format_delta_items(delta_items),
    )
    raw = call_lm(lm_client, model, prompt)
    print(f"\n[Curator raw]\n{raw}")
    return _parse_delta_items(raw)


# ----------------------------------------------------------------------
# Grow-and-refine  (Section 3.2)
# ----------------------------------------------------------------------

def grow_and_refine(playbook: Playbook, similarity_threshold: float = 0.85):
    """De-duplicate semantically near-identical bullets via difflib ratio."""
    if len(playbook.items) < 2:
        return

    merged = 0
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
                if b.helpful_count > a.helpful_count:
                    playbook.delete(a.id)
                    merged += 1
                    a = playbook.items[i] if i < len(playbook.items) else None
                    if a is None:
                        break
                    j = i + 1
                    continue
                else:
                    playbook.delete(b.id)
                    merged += 1
                    continue
            j += 1
        i += 1

    if merged:
        print(f"[Playbook] grow_and_refine merged {merged} near-duplicate entries.")
    else:
        print("[Playbook] grow_and_refine: no duplicates found.")


# ----------------------------------------------------------------------
# ACEMethod — BaseMethod-compatible wrapper around the ACE loop
# ----------------------------------------------------------------------

class ACEMethod(BaseMethod):
    """
    Full ACE pipeline (Generator → Reflector → Curator → Playbook) bound
    to a single environment.
    """

    name = "ace"

    def __init__(
        self,
        env,
        model: str = "qwen3-8b",
        server_url: str = "http://LOCAL_SERVER/v1",
        reward_threshold: float = 1.0,
        refine_every: int = 5,
    ):
        self.env = env
        self.model = model
        self.reward_threshold = reward_threshold
        self.refine_every = refine_every
        self.playbook = self.initialize_context()
        self.episode_logs = []

        self.client = build_client(server_url)

        self.reflector_instruction = str(_INSTRUCTIONS_DIR / "instruction_reflector.md")
        self.curator_instruction = str(_INSTRUCTIONS_DIR / "instruction_curator.md")

        print(f"Connected to LM server at {server_url}")
        print(f"Model: {self.model}")
        print("ACE Method ready.")

    # -- BaseMethod contract ------------------------------------------------

    def initialize_context(self) -> Playbook:
        return Playbook()

    # -- Per-attempt step loop ----------------------------------------------

    def _run_attempt(self, build_step_prompt) -> tuple:
        all_actions = []
        all_feedbacks = []
        reward = 0

        while not self.env.done:
            obs = self.env.get_observation()
            lm_out = call_lm(self.client, self.model, build_step_prompt(obs))
            action = parse_action_single(lm_out)
            all_actions.append(action)
            _, step_feedback, reward, done = self.env.step([action])
            all_feedbacks.append(step_feedback)
            if done:
                break

        return all_actions, " ".join(all_feedbacks), reward

    # -- Episode loop -------------------------------------------------------

    def run_episode(self, episode_num: int) -> dict:
        initial_obs = self.env.reset(seed=episode_num)
        actions1, feedback1, reward1 = self._run_attempt(build_attempt1_prompt)

        print(f"\n{'='*40}")
        print(f"=== Episode {episode_num} ===")
        print(f"{'='*40}")
        print(f"[Attempt 1] Actions:  {actions1}")
        print(f"[Attempt 1] Feedback: {feedback1}")
        print(f"[Attempt 1] Reward:   {reward1}")

        if reward1 >= self.reward_threshold:
            print("[Gated] Attempt 1 succeeded. Skipping reflection and attempt 2.")
            approved_deltas = []
            actions2, feedback2, reward2 = actions1, feedback1, reward1
            gated = True
        else:
            gated = False

            delta_items = run_reflector(
                self.client, self.model,
                initial_obs, actions1, feedback1, reward1,
                self.playbook, self.reflector_instruction,
            )
            print(f"[Reflector] Proposed {len(delta_items)} delta items")

            approved_deltas = run_curator(
                self.client, self.model,
                delta_items, self.playbook, self.curator_instruction,
            )
            print(f"[Curator] Approved {len(approved_deltas)} delta items")

            self.playbook.apply_delta(approved_deltas)
            print(f"[Playbook] Size now: {len(self.playbook.items)}")

            if self.refine_every > 0 and episode_num % self.refine_every == 0:
                grow_and_refine(self.playbook)

            self.env.reset()
            actions2, feedback2, reward2 = self._run_attempt(
                lambda obs: build_attempt2_prompt_with_playbook(obs, self.playbook)
            )

            print(f"\n[Attempt 2] Actions:  {actions2}")
            print(f"[Attempt 2] Feedback: {feedback2}")
            print(f"[Attempt 2] Reward:   {reward2}")

        return {
            "episode": episode_num,
            "actions1": actions1,
            "feedback1": feedback1,
            "reward1": reward1,
            "delta_items": [asdict(d) for d in approved_deltas],
            "playbook": self.playbook.to_dict(),
            "actions2": actions2,
            "feedback2": feedback2,
            "reward2": reward2,
            "playbook_size": len(self.playbook.items),
            "gated": gated,
        }

    # -- Full experiment ----------------------------------------------------

    def run(self, n_episodes: int) -> dict:
        all_logs = []
        for ep in range(1, n_episodes + 1):
            all_logs.append(self.run_episode(ep))
        return summarize_logs(
            all_logs,
            self.reward_threshold,
            type(self.env).__name__,
            n_episodes,
        )
