"""
ACE (Agentic Context Engineering) method — BaseMethod-compatible.

Paper: "Agentic Context Engineering: Evolving Contexts for
Self-Improving Language Models" — arXiv 2510.04618.

Consolidates the Playbook bullet structure, Reflector diagnosis,
Curator delta updates, the grow-and-refine de-duplication pass, and the
two-attempt episode loop into a single module — mirroring the way
methods/ace.py is organized in the appworld-context-updater template.

Sections referenced below:
  - Section 3.1  Incremental delta updates, bullet schema
  - Section 3.2  Grow-and-refine
  - Appendix D   Prompt templates
"""

import difflib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from common import (
    BaseMethod,
    build_client,
    call_lm,
    parse_action_single,
    render_template,
    summarize_logs,
)
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
    """A proposed change to the playbook emitted by the Curator."""
    operation: str            # "ADD"; original ACE curator is ADD-only
    id: int                   # -1 for ADD
    content: str              # new bullet text
    reason: str = ""          # explanation (for logging, not applied)


# ----------------------------------------------------------------------
# Playbook
# ----------------------------------------------------------------------

class Playbook:
    """
    Ordered collection of PlaybookItems with stable integer ids.

    The playbook is the evolving context described in the paper: the
    Generator reads it, the Reflector diagnoses trajectories, the
    Curator proposes ADD-only deltas, and apply_delta merges them deterministically.
    """

    def __init__(self):
        self.items: list = []
        self._next_id: int = 1

    def add(self, content: str) -> PlaybookItem:
        item = PlaybookItem(id=self._next_id, content=content.strip())
        self.items.append(item)
        self._next_id += 1
        return item

    def _remove_item(self, item_id: int) -> bool:
        """Remove an item during deterministic duplicate pruning."""
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
        """Apply ADD-only Curator output in order and print a short summary."""
        added = skipped = 0
        for d in deltas:
            op = d.operation.upper()
            if op == "ADD":
                if d.content.strip():
                    self.add(d.content)
                    added += 1
                else:
                    skipped += 1
            else:
                skipped += 1

        print(f"[Playbook] apply_delta: +{added} add ({skipped} skipped)")

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

_OP_RE = re.compile(r"^\s*\[ADD\]\s*(.*)$", re.IGNORECASE)
_REASON_RE = re.compile(r"^\s*reason\s*:\s*(.*)$", re.IGNORECASE)
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_PLAYBOOK_LINE_RE = re.compile(r"playbook|relevant entries|entry ids?", re.IGNORECASE)


def _parse_delta_items(text: str) -> list:
    """
    Parse the LM's delta output into a list of DeltaItem.

    The original ACE curator is ADD-only. The preferred format is JSON:

      {"reasoning": "...", "operations": [{"type": "ADD", "content": "..."}]}

    A legacy [ADD] fallback is accepted for robustness, but MODIFY/DELETE
    are intentionally ignored.

    Returns an empty list for [NO_CHANGE] or when no items are found.
    Never raises.
    """
    if not text:
        return []
    if "[NO_CHANGE]" in text.upper():
        return []

    payload = _extract_json_payload(text)
    operations = payload.get("operations", []) if isinstance(payload, dict) else []
    if isinstance(operations, list):
        json_deltas = []
        for op in operations:
            if not isinstance(op, dict):
                continue
            if str(op.get("type", "")).strip().upper() != "ADD":
                continue
            content = str(op.get("content", "")).strip()
            if not content:
                continue
            reason = str(op.get("reason", "")).strip()
            json_deltas.append(
                DeltaItem(operation="ADD", id=-1, content=content, reason=reason)
            )
        if json_deltas:
            return json_deltas

    deltas = []
    current = None

    for raw in text.split("\n"):
        line = raw.rstrip()
        m = _OP_RE.match(line)
        if m:
            if current is not None:
                deltas.append(current)
            content = m.group(1).strip()
            current = DeltaItem(operation="ADD", id=-1, content=content, reason="")
            continue

        rm = _REASON_RE.match(line)
        if rm and current is not None:
            current.reason = rm.group(1).strip()
            continue

        if (
            current is not None
            and not current.content
            and line.strip()
        ):
            current.content = line.strip()

    if current is not None:
        deltas.append(current)

    cleaned = []
    for d in deltas:
        if d.operation == "ADD" and d.content:
            cleaned.append(d)
    return cleaned


def _extract_json_payload(raw: str) -> dict:
    """
    Extract a JSON object from an LM response.

    The Reflector and Curator prompts ask for JSON, but local models may add
    small wrappers. This keeps parsing robust while preserving the intended
    division of labor between Reflector diagnosis and Curator ADDs.
    """
    if not raw:
        return {}

    m = _JSON_FENCE_RE.search(raw)
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
    return {}


def _empty_reflection(raw: str = "") -> dict:
    return {
        "reasoning": raw.strip(),
        "error_identification": "",
        "root_cause_analysis": "",
        "correct_approach": "",
        "key_insight": "no new playbook insight",
        "playbook_feedback": [],
    }


def _normalize_reflection(raw: str) -> dict:
    payload = _extract_json_payload(raw)
    if not payload:
        return _empty_reflection(
            "Reflector output was not valid JSON; no structured diagnosis available."
        )

    reflection = _empty_reflection()
    for key in (
        "reasoning",
        "error_identification",
        "root_cause_analysis",
        "correct_approach",
        "key_insight",
    ):
        value = payload.get(key, "")
        reflection[key] = str(value).strip()

    feedback = payload.get("playbook_feedback", [])
    reflection["playbook_feedback"] = feedback if isinstance(feedback, list) else []
    return reflection


def _extract_generator_playbook_ids(lm_output: str, valid_ids: set) -> list:
    """Find playbook ids the Generator explicitly mentioned in its reasoning."""
    if not lm_output or not valid_ids:
        return []

    found = set()
    for line in lm_output.splitlines():
        if not _PLAYBOOK_LINE_RE.search(line):
            continue
        if "none" in line.lower() and not re.search(r"\d", line):
            continue
        for m in re.finditer(r"\b\d+\b", line):
            item_id = int(m.group(0))
            if item_id in valid_ids:
                found.add(item_id)
    return sorted(found)


def _format_generator_trace(step_traces: list) -> str:
    if not step_traces:
        return "(no generator trace captured)"

    chunks = []
    for tr in step_traces:
        ids = tr.get("playbook_ids", [])
        ids_text = ids if ids else "none"
        chunks.append(
            f"Step {tr.get('step')} action: {tr.get('action')}\n"
            f"Referenced playbook ids: {ids_text}\n"
            f"Generator raw output:\n{tr.get('raw_output', '')}"
        )
    return "\n\n---\n\n".join(chunks)


def _apply_playbook_feedback(
    playbook: Playbook,
    reflection: dict,
    generator_traces: list,
    reward: int,
    reward_threshold: float,
) -> dict:
    """
    Update helpful/harmful counters.

    Prefer Reflector labels because they are the ACE-style credit signal. If
    the model omits labels, fall back to episode-level credit for Generator-
    referenced ids so the metadata is still useful.
    """
    valid_ids = {it.id for it in playbook.items}
    stats = {"helpful": 0, "harmful": 0, "neutral": 0, "fallback": False}
    if not valid_ids:
        return stats

    seen = set()
    feedback = reflection.get("playbook_feedback", [])
    if isinstance(feedback, list):
        for item in feedback:
            if not isinstance(item, dict):
                continue
            try:
                item_id = int(item.get("id"))
            except (TypeError, ValueError):
                continue
            if item_id not in valid_ids or item_id in seen:
                continue
            label = str(item.get("label", "")).strip().lower()
            seen.add(item_id)
            if label == "helpful":
                playbook.mark_helpful(item_id)
                stats["helpful"] += 1
            elif label == "harmful":
                playbook.mark_harmful(item_id)
                stats["harmful"] += 1
            else:
                stats["neutral"] += 1

    if seen:
        return stats

    referenced = set()
    for tr in generator_traces:
        referenced.update(i for i in tr.get("playbook_ids", []) if i in valid_ids)
    if not referenced:
        return stats

    stats["fallback"] = True
    if reward >= reward_threshold:
        for item_id in referenced:
            playbook.mark_helpful(item_id)
        stats["helpful"] = len(referenced)
    else:
        for item_id in referenced:
            playbook.mark_harmful(item_id)
        stats["harmful"] = len(referenced)
    return stats


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
    generator_trace: str,
    playbook: Playbook,
    instruction_path: str,
    disable_thinking: bool = False,
) -> dict:
    """Critique a trajectory and return a structured reflection."""
    template = _load_instruction(instruction_path)
    prompt = render_template(
        template,
        observation=observation,
        actions=actions,
        feedback=feedback,
        reward=reward,
        generator_trace=generator_trace,
        playbook=playbook.to_prompt_string(),
    )
    raw = call_lm(
        lm_client, model, prompt, disable_thinking=disable_thinking
    )
    print(f"\n[Reflector raw]\n{raw}")
    return _normalize_reflection(raw)


# ----------------------------------------------------------------------
# Curator  (Section 3, Appendix D)
# ----------------------------------------------------------------------

def run_curator(
    lm_client,
    model: str,
    reflection: dict,
    playbook: Playbook,
    instruction_path: str,
    disable_thinking: bool = False,
) -> list:
    """Turn the Reflector's diagnosis into approved playbook deltas."""
    if not reflection:
        return []
    template = _load_instruction(instruction_path)
    prompt = render_template(
        template,
        playbook=playbook.to_prompt_string(),
        reflection=json.dumps(reflection, indent=2),
    )
    raw = call_lm(
        lm_client, model, prompt, disable_thinking=disable_thinking
    )
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
                    playbook._remove_item(a.id)
                    merged += 1
                    a = playbook.items[i] if i < len(playbook.items) else None
                    if a is None:
                        break
                    j = i + 1
                    continue
                else:
                    playbook._remove_item(b.id)
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
        disable_thinking: bool = False,
    ):
        self.env = env
        self.model = model
        self.reward_threshold = reward_threshold
        self.refine_every = refine_every
        self.disable_thinking = disable_thinking
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
        generator_traces = []
        reward = 0

        while not self.env.done:
            obs = self.env.get_observation()
            lm_out = call_lm(
                self.client, self.model, build_step_prompt(obs),
                disable_thinking=self.disable_thinking,
            )
            action = parse_action_single(lm_out)
            all_actions.append(action)
            generator_traces.append({
                "step": len(all_actions),
                "action": action,
                "playbook_ids": _extract_generator_playbook_ids(
                    lm_out, {it.id for it in self.playbook.items}
                ),
                "raw_output": lm_out,
            })
            _, step_feedback, reward, done = self.env.step([action])
            all_feedbacks.append(step_feedback)
            if done:
                break

        return all_actions, " ".join(all_feedbacks), reward, generator_traces

    # -- Episode loop -------------------------------------------------------

    def run_episode(self, episode_num: int) -> dict:
        initial_obs = self.env.reset(seed=episode_num)
        # Online ACE evaluation: solve each new stage using the playbook
        # accumulated before seeing this stage. The running attempt-1 curve is
        # therefore the average pass rate over the first K online stages.
        actions1, feedback1, reward1, generator_trace1 = self._run_attempt(
            lambda obs: build_attempt2_prompt_with_playbook(obs, self.playbook)
        )

        print(f"\n{'='*40}")
        print(f"=== Episode {episode_num} ===")
        print(f"{'='*40}")
        print(f"[Attempt 1] Actions:  {actions1}")
        print(f"[Attempt 1] Feedback: {feedback1}")
        print(f"[Attempt 1] Reward:   {reward1}")

        reflection = run_reflector(
            self.client, self.model,
            initial_obs, actions1, feedback1, reward1,
            _format_generator_trace(generator_trace1),
            self.playbook, self.reflector_instruction,
            disable_thinking=self.disable_thinking,
        )
        print("[Reflector] Structured reflection captured")

        feedback_stats = _apply_playbook_feedback(
            self.playbook, reflection, generator_trace1,
            reward1, self.reward_threshold,
        )
        if any(feedback_stats[k] for k in ("helpful", "harmful", "neutral")):
            source = "fallback" if feedback_stats["fallback"] else "reflector"
            print(
                f"[Playbook] feedback ({source}): "
                f"+{feedback_stats['helpful']} helpful / "
                f"+{feedback_stats['harmful']} harmful / "
                f"{feedback_stats['neutral']} neutral"
            )

        approved_deltas = run_curator(
            self.client, self.model,
            reflection, self.playbook, self.curator_instruction,
            disable_thinking=self.disable_thinking,
        )
        print(f"[Curator] Approved {len(approved_deltas)} delta items")

        self.playbook.apply_delta(approved_deltas)
        print(f"[Playbook] Size now: {len(self.playbook.items)}")

        if self.refine_every > 0 and episode_num % self.refine_every == 0:
            grow_and_refine(self.playbook)

        if reward1 >= self.reward_threshold:
            print("[Gated] Attempt 1 succeeded. Skipping retry.")
            actions2, feedback2, reward2 = actions1, feedback1, reward1
            generator_trace2 = generator_trace1
            gated = True
        else:
            gated = False
            self.env.reset()
            actions2, feedback2, reward2, generator_trace2 = self._run_attempt(
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
            "generator_trace1": generator_trace1,
            "reflection": reflection,
            "playbook_feedback": feedback_stats,
            "delta_items": [asdict(d) for d in approved_deltas],
            "playbook": self.playbook.to_dict(),
            "actions2": actions2,
            "feedback2": feedback2,
            "reward2": reward2,
            "generator_trace2": generator_trace2,
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
