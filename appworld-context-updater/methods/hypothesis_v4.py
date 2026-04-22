"""
hypothesis_v4: ACE + Improved Open Question Curator.

Builds on hypothesis_v3 with three key fixes:
  1. Uses a custom agent prompt that instructs the agent to explore OQs.
  2. Revised OQ curator prompt that prevents semantic drift (no topic-changing
     EDITs), avoids duplicates via reasoning, and skips already-answered questions.
  3. OQ entries are shorter and focused: question + investigation steps only.
"""

import json
import re
from pathlib import Path
from typing import Any

from jinja2 import Template

from common import LLMClient, extract_json_payload, render_conversation_history
from methods.ace import ACEModel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
HYPOTHESIS_PROMPT_FILE = (
    REPO_ROOT / "libs" / "ace-appworld" / "experiments" / "prompts"
    / "appworld_react_code_agent_hypothesis_generator_prompt.txt"
)

# ---------------------------------------------------------------------------
# OQ section helpers (shared with v3)
# ---------------------------------------------------------------------------

OQ_SECTION_HEADER = "## OPEN QUESTIONS"
OQ_PREFIX = "oq"


def _parse_oq_items(playbook: str) -> list[dict[str, str]]:
    """Return items from the ## OPEN QUESTIONS section."""
    items: list[dict[str, str]] = []
    in_oq = False
    for line in playbook.splitlines():
        stripped = line.strip()
        if stripped == OQ_SECTION_HEADER:
            in_oq = True
            continue
        if in_oq and stripped.startswith("## "):
            break
        if in_oq and stripped:
            m = re.match(r"\[(?P<tag>[^\]]+)\]\s*(?P<content>.*)", stripped)
            if m:
                items.append({"tag": m.group("tag"), "content": m.group("content")})
    return items


def _split_main_and_oq(playbook: str) -> tuple[str, list[dict[str, str]]]:
    """Split playbook into (main_body, oq_items). main_body has no OQ section."""
    lines = playbook.splitlines()
    oq_start = next(
        (i for i, l in enumerate(lines) if l.strip() == OQ_SECTION_HEADER), None
    )
    if oq_start is None:
        return playbook, []
    main_lines = lines[:oq_start]
    while main_lines and not main_lines[-1].strip():
        main_lines.pop()
    return "\n".join(main_lines), _parse_oq_items(playbook)


def _next_oq_id(oq_items: list[dict[str, str]]) -> int:
    max_id = 0
    for item in oq_items:
        m = re.search(r"(\d+)$", item["tag"])
        if m:
            max_id = max(max_id, int(m.group(1)))
    return max_id + 1


def apply_oq_operations(playbook: str, operations: list[dict]) -> str:
    """Apply ADD/EDIT/DELETE operations to the ## OPEN QUESTIONS section."""
    main_body, oq_items = _split_main_and_oq(playbook)
    index_by_tag = {item["tag"]: i for i, item in enumerate(oq_items)}
    next_id = _next_oq_id(oq_items)

    for op in operations:
        op_type = op.get("type", "").upper()
        if op_type == "ADD":
            tag = f"{OQ_PREFIX}-{next_id:05d}"
            next_id += 1
            oq_items.append({"tag": tag, "content": op.get("content", "").strip()})
            print(f"  OQ: added {tag}")
        elif op_type == "EDIT":
            tag = op.get("tag", "")
            if tag in index_by_tag:
                oq_items[index_by_tag[tag]]["content"] = op.get("content", "").strip()
                print(f"  OQ: edited {tag}")
            else:
                print(f"  OQ: Warning: EDIT tag '{tag}' not found, skipping")
        elif op_type == "DELETE":
            tag = op.get("tag", "")
            if tag in index_by_tag:
                oq_items.pop(index_by_tag[tag])
                index_by_tag = {item["tag"]: i for i, item in enumerate(oq_items)}
                print(f"  OQ: deleted {tag}")
            else:
                print(f"  OQ: Warning: DELETE tag '{tag}' not found, skipping")

    oq_lines = [OQ_SECTION_HEADER]
    oq_lines.extend(f"[{item['tag']}] {item['content']}" for item in oq_items)
    return main_body + "\n\n" + "\n".join(oq_lines)


# ---------------------------------------------------------------------------
# Improved Open Question Curator prompt
# ---------------------------------------------------------------------------

OQ_CURATOR_PROMPT = """\
You are a curator of open questions for an AppWorld coding agent. You maintain a list of unresolved uncertainties about the AppWorld environment worth investigating in future tasks.

**Context:**
- These open questions are included in the agent's playbook. The agent is instructed to explore relevant ones during task execution and to reference them by ID (e.g., "Investigating oq-00003").
- Your goal is a compact, high-signal list of questions. Each question should name a specific API, parameter, or behavior and suggest a concrete 1-2 step investigation.

**Rules:**

1. **Do NOT add a question that is already answered in the main playbook.** Before proposing an ADD, check whether the main playbook sections already contain the answer. If they do, skip it.

2. **Do NOT add a near-duplicate.** Before proposing an ADD, check the existing open questions. If an existing OQ covers the same API, parameter, or uncertainty, EDIT that existing OQ to incorporate the new evidence instead of adding a new one.

3. **EDIT must preserve the original topic.** You may EDIT an existing OQ only to:
   - Add new evidence or observations
   - Refine the investigation suggestion
   - Mark it as partially answered
   You must NOT replace the question with a different topic. If you have a new, unrelated question, use ADD instead.

4. **DELETE when resolved.** Delete an OQ only when the main playbook update definitively answers it — not just because a related entry was added. The answer must fully resolve the uncertainty.

5. **Keep entries concise.** Each OQ should be a single sentence question followed by "Suggestion:" with 1-2 concrete steps. Target 100-200 characters total. Do not include lengthy background.

**Inputs:**

- **Task Context:**
  `{{ question_context }}`

- **Current Playbook (main sections + open questions):**
  `{{ current_playbook }}`

- **Playbook Update Based on the Current Task (bullets added to the main sections by the curator):**
  `{{ playbook_delta }}`

- **Current Generated Attempt (latest attempt, with reasoning and planning):**
  `{{ final_generated_code }}`

- **Current Reflections:**
  `{{ guidebook }}`

**Examples:**

**Example 1 — ADD (no existing OQ covers this):**
{
  "reasoning": "The agent got a KeyError on result['amount']. No existing OQ asks about Venmo transaction fields, and the main playbook doesn't document this. Adding a new question.",
  "operations": [
    {
      "type": "ADD",
      "content": "What field names does get_transaction_details return? The agent assumed 'amount' but got KeyError. Suggestion: print(result) on the first call to see all field names."
    }
  ]
}

**Example 2 — EDIT (same topic, new evidence):**
Existing: [oq-00005] What field names does get_transaction_details return? Suggestion: print(result).
New evidence: The agent found the field is 'total_amount', but it's unclear if there are other unexpected fields.

{
  "reasoning": "oq-00005 already asks about transaction fields. The agent found 'total_amount' works. Updating with this partial answer and narrowing the remaining uncertainty.",
  "operations": [
    {
      "type": "EDIT",
      "tag": "oq-00005",
      "content": "What other fields does get_transaction_details return besides 'total_amount'? The 'amount' KeyError is resolved ('total_amount' is correct), but the full schema is still unknown. Suggestion: print(list(result.keys())) to discover all fields."
    }
  ]
}

**Example 3 — DELETE (fully resolved):**
Existing: [oq-00002] Does pagination start at page_index=0? Suggestion: print first page result.
Playbook update added: "All AppWorld APIs use page_index=0 as the first page."

{
  "reasoning": "The playbook update definitively answers this: pagination starts at 0 for all APIs. Deleting.",
  "operations": [
    {"type": "DELETE", "tag": "oq-00002"}
  ]
}

**Example 4 — Skip (already in playbook):**
Reflection mentions the agent should use apis.phone.search_contacts() for relationships.
Main playbook already has: "[shr-00010] Always use Phone app contacts for identifying relationships."

{
  "reasoning": "The playbook already documents this (shr-00010). No new OQ needed.",
  "operations": []
}

**Example 5 — Merge into existing (near-duplicate):**
Existing: [oq-00007] What parameters does show_playlist_library accept? Suggestion: check API docs first.
New uncertainty: The agent also found that the response format of show_playlist_library is unclear.

{
  "reasoning": "oq-00007 already covers show_playlist_library. Merging the response format question into it instead of adding a duplicate.",
  "operations": [
    {
      "type": "EDIT",
      "tag": "oq-00007",
      "content": "What parameters does show_playlist_library accept and what does it return? Suggestion: call show_api_doc('spotify', 'show_playlist_library') and print one page of results to confirm field names."
    }
  ]
}

**Your Task:**
Output ONLY a valid JSON object:
{
  "reasoning": "[Step-by-step: (1) list new uncertainties from this task, (2) for each, check if it is already answered in the main playbook — if yes skip, (3) check if an existing OQ covers the same topic — if yes EDIT that OQ, (4) otherwise ADD, (5) check which existing OQs are now resolved — DELETE those]",
  "operations": [...]
}

**Available Operations:**
1. ADD — new open question (system assigns ID)
    - type: "ADD"
    - content: "Question? Suggestion: steps."
2. EDIT — update an existing question (same topic only)
    - type: "EDIT"
    - tag: "oq-NNNNN"
    - content: "Revised question? Suggestion: steps."
3. DELETE — remove a resolved question
    - type: "DELETE"
    - tag: "oq-NNNNN"

**RESPONSE FORMAT — Output ONLY this JSON (no markdown, no code blocks):**
{
  "reasoning": "[Your reasoning]",
  "operations": [...]
}"""


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class HypothesisV4Model(ACEModel):
    """
    ACE pipeline + Improved Open Question Curator.

    Key improvements over V3:
    1. Custom agent prompt that instructs the agent to explore OQs.
    2. OQ curator prevents semantic drift, avoids duplicates, skips
       already-answered questions.
    3. Shorter, more focused OQ entries.
    """

    def __init__(self, use_ground_truth: bool = True):
        super().__init__(
            use_ground_truth=use_ground_truth,
            use_test_report=True,
            initial_playbook="default",
        )
        self.name = "hypothesis_v4" if use_ground_truth else "hypothesis_v4_nogt"
        self.prompt_file = HYPOTHESIS_PROMPT_FILE

    def initialize_context(self) -> str:
        base = super().initialize_context().rstrip()
        return base + f"\n\n{OQ_SECTION_HEADER}\n"

    # ---- OQ curator --------------------------------------------------------

    def _call_oq_curator(
        self,
        llm_client: LLMClient,
        playbook: str,
        task_instruction: str,
        reflection: dict,
        full_trace: list[dict[str, Any]],
        ace_operations: list[dict],
    ) -> dict:
        prompt = Template(OQ_CURATOR_PROMPT).render(
            question_context=task_instruction,
            current_playbook=playbook,
            guidebook=json.dumps(reflection, indent=2),
            final_generated_code="See full conversation history below",
            playbook_delta=json.dumps(ace_operations, indent=2) if ace_operations else "(no changes)",
        )
        prompt += "\n\n" + render_conversation_history(full_trace)

        raw = llm_client.generate([{"role": "user", "content": prompt}])["content"]
        if not raw.strip():
            print(f"[{self.name}] Warning: empty OQ curator response.")
            return {"reasoning": "", "operations": []}
        try:
            return extract_json_payload(raw)
        except Exception as exc:
            print(f"[{self.name}] Warning: failed to parse OQ curator JSON: {exc}")
            return {"reasoning": "", "operations": []}

    def _validate_oq_operations(self, operations: Any) -> list[dict]:
        if not isinstance(operations, list):
            return []
        filtered: list[dict] = []
        for i, op in enumerate(operations):
            if not isinstance(op, dict):
                continue
            op_type = op.get("type", "").upper()
            if op_type == "ADD":
                if "content" not in op:
                    print(f"  Skipping OQ ADD {i}: missing content")
                    continue
            elif op_type == "EDIT":
                if "tag" not in op or "content" not in op:
                    print(f"  Skipping OQ EDIT {i}: missing tag or content")
                    continue
            elif op_type == "DELETE":
                if "tag" not in op:
                    print(f"  Skipping OQ DELETE {i}: missing tag")
                    continue
            else:
                print(f"  Skipping OQ operation {i}: unknown type '{op.get('type')}'")
                continue
            filtered.append(op)
        return filtered

    # ---- main entry point --------------------------------------------------

    def update_context(
        self,
        llm_client: LLMClient,
        current_context: str,
        task_instruction: str,
        full_trace: list[dict[str, Any]],
        test_report: str,
        success: bool,
        ground_truth_code: str = "",
    ) -> tuple[str, Any]:
        # Step 1: reflector (identical to ACE)
        reflection = self._call_reflector(
            llm_client, current_context, full_trace, test_report, ground_truth_code,
        )

        # Step 2: ACE curator — main sections only
        curator_result = self._call_curator(
            llm_client, current_context, task_instruction, reflection, full_trace,
        )
        ace_operations = self._validate_operations(curator_result.get("operations", []))
        playbook_after_ace = self._apply_curator_ops(current_context, ace_operations)

        # Step 3: OQ curator — ## OPEN QUESTIONS section only
        oq_result = self._call_oq_curator(
            llm_client, playbook_after_ace, task_instruction, reflection, full_trace, ace_operations,
        )
        oq_operations = self._validate_oq_operations(oq_result.get("operations", []))
        final_context = apply_oq_operations(playbook_after_ace, oq_operations)

        context_delta = {"ace_curator": ace_operations, "oq_curator": oq_operations}
        return final_context, context_delta
