"""
hypothesis_v3: ACE + Open Question Curator.

Uses the exact ACE pipeline (Reflector + ADD-only Curator) for the 8 standard
playbook sections, then adds a third LLM call — the Open Question Curator —
that manages an ## OPEN QUESTIONS section with ADD/EDIT/DELETE operations.

The final context = ACE-curated playbook  +  ## OPEN QUESTIONS section.
"""

import json
import re
from typing import Any

from jinja2 import Template

from common import LLMClient, extract_json_payload, render_conversation_history
from methods.ace import ACEModel

# ---------------------------------------------------------------------------
# OQ section helpers
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
# Open Question Curator prompt
# ---------------------------------------------------------------------------

OQ_CURATOR_PROMPT = """\
You are a curator of open questions for an AppWorld coding agent. Your job is to maintain a list of open questions — unresolved uncertainties about the AppWorld environment worth investigating in future tasks.

**Context:**
- These open questions will be included in the agent's playbook for future tasks.
- The reflection is generated using ground truth answers that will NOT be available when the playbook is used.

**Instructions:**
- Review the current open questions and the reflection from the latest task
- ADD new questions for uncertainties this trace raised or left unresolved
- EDIT existing questions when new evidence sharpens, partially answers, or redirects them
- DELETE questions that have been fully resolved by the playbook update
- Avoid duplicates — prefer editing an existing question over adding a near-duplicate
- Do NOT add questions for facts already definitively answered in the main playbook or the playbook update below

**Item format — every open question must have two parts:**

Part 1 — The question: describe what is uncertain and why it matters for future tasks.
  - Be specific: name the API, parameter, or behavior in question
  - Explain the consequence of getting it wrong (e.g., wrong results, wasted turns, errors)

Part 2 — Suggestions: concrete investigation steps the agent should take next time it encounters this.
  - Prefer direct inspection: print the full object returned by an API call before processing it
  - Print intermediate results (counts, keys, a sample item) before committing to a filtering or deletion strategy
  - Try minimal calls first (e.g., call with one parameter before adding more) and print the result
  - When uncertain about message structure or content, print the raw message before acting on it
  - Use small test calls on one item before looping over all items

Write each item as a single flowing paragraph: question first, then "Suggestion: ..." inline.

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

**Example 1 — ADD a new question:**
Reflection: "It's unclear whether search_contacts needs both `query` and `relationship` or if `query` alone suffices. The agent passed both and got correct results, but we don't know if both are required."

Response:
{
  "reasoning": "The trace succeeded using both params together but neither was tested alone. If `query` alone works, using both is redundant and could mask filtering bugs. Worth verifying to simplify future contact-lookup code.",
  "operations": [
    {
      "type": "ADD",
      "content": "Does search_contacts need `relationship` or does `query` alone suffice? The agent passed both query='roommate' and relationship='roommate' and got correct results, but it is unclear which parameter is doing the filtering — using only one might be sufficient. Suggestion: next time, first call apis.phone.search_contacts(query='roommate') without relationship, print the full result list and count. Then call again with relationship='roommate' added and compare. This one-step probe takes a single turn and resolves the ambiguity."
    }
  ]
}

**Example 2 — ADD a question about unknown object structure:**
Reflection: "The agent called apis.venmo.get_transaction_details(transaction_id) but immediately accessed result['amount'] without printing the object. It got a KeyError because the field is named 'total_amount'."

Response:
{
  "reasoning": "The agent assumed a field name without inspecting the object first. This is a recurring risk for any API that returns a complex object — we don't know field names until we print one.",
  "operations": [
    {
      "type": "ADD",
      "content": "What does the Venmo transaction detail object look like? The agent assumed a field named 'amount' but the actual field is 'total_amount', causing a KeyError. The correct schema is not yet fully known. Suggestion: on the first call to get_transaction_details in any task, print the full result object before accessing any fields — e.g., print(result) or print(list(result.keys())) — to confirm field names before writing filtering or aggregation logic."
    }
  ]
}

**Example 3 — DELETE a resolved question:**
Current OQ has: [oq-00002] Does pagination always start at page_index=0? Suggestion: print the first page result...
Playbook update added: "Pagination always starts at page_index=0 for all AppWorld APIs."

Response:
{
  "reasoning": "The playbook update just documented that page_index starts at 0. This question is resolved.",
  "operations": [
    {
      "type": "DELETE",
      "tag": "oq-00002"
    }
  ]
}

**Example 4 — EDIT an existing question with new evidence:**
Current OQ has: [oq-00005] What does like_transaction return on success? Suggestion: print the return value on the first call.
Reflection: "like_transaction raises HTTP 422 if the transaction was already liked — it is not a silent no-op."

Response:
{
  "reasoning": "The return value on success is still unknown, but a more important risk emerged: duplicate likes raise an error. Update the question to capture the new finding and redirect the investigation.",
  "operations": [
    {
      "type": "EDIT",
      "tag": "oq-00005",
      "content": "Does like_transaction silently skip or raise an error on already-liked transactions? It raises HTTP 422 'You have already liked this transaction' — not a silent no-op — which will crash a loop that likes all items without checking. The correct handling strategy (try/except vs. pre-check) is still unclear. Suggestion: before bulk-liking, call like_transaction on one item and print the full response. Then try calling it again on the same item and print the error to confirm the 422 pattern. Decide whether to add a try/except around the loop or to check liked status first."
    }
  ]
}

**Your Task:**
Output ONLY a valid JSON object with these exact fields:
- reasoning: chain of thought about what to add, edit, or delete
- operations: list of operations on the open questions section

**Available Operations:**
1. ADD — add a new open question (no tag needed; system assigns oq-NNNNN)
    - type: "ADD"
    - content: question + suggestion as a single paragraph
2. EDIT — update an existing question
    - type: "EDIT"
    - tag: existing tag (e.g., "oq-00003")
    - content: revised question + suggestion as a single paragraph
3. DELETE — remove a resolved or superseded question
    - type: "DELETE"
    - tag: existing tag (e.g., "oq-00003")

**RESPONSE FORMAT — Output ONLY this JSON (no markdown, no code blocks):**
{
  "reasoning": "[Your reasoning here]",
  "operations": [
    {"type": "ADD", "content": "[Question description and why it matters. Suggestion: concrete investigation steps...]"},
    {"type": "EDIT", "tag": "oq-00003", "content": "[Revised question. Suggestion: ...]"},
    {"type": "DELETE", "tag": "oq-00005"},
    ...
  ]
}"""


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class HypothesisV3Model(ACEModel):
    """
    ACE pipeline + Open Question Curator.

    Calls 1 & 2 are identical to ACEModel (reflector + ADD-only curator for the
    8 standard sections). Call 3 is a new OQ Curator that manages an
    ## OPEN QUESTIONS section with ADD/EDIT/DELETE using oq-NNNNN bullet IDs.

    use_ground_truth=True passes GT code to the reflector (hypothesis_v3_gt).
    """

    def __init__(self, use_ground_truth: bool = True):
        super().__init__(
            use_ground_truth=use_ground_truth,
            use_test_report=True,
            initial_playbook="default",
        )
        self.name = "hypothesis_v3" if use_ground_truth else "hypothesis_v3_nogt"

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
