from jinja2 import Template

from common import (
    BaseModel,
    apply_tagged_operations,
    extract_json_payload,
    render_conversation_history,
)

SUMMARIZE_DELTA_PROMPT = """You are an expert playbook editor for an AI coding agent operating in the AppWorld environment.

AppWorld is a simulated phone-like world with apps such as messaging, contacts, Spotify, Venmo, files, and other app APIs. The agent interacts through a ReAct loop:
1. it writes Python,
2. the environment executes it and returns output,
3. it writes the next Python step.

The notebook is a flat list of tagged items. Each item has the format:
[<tag>] <title>: <content>

Your job is to improve this notebook using the latest task trace and test report.

**Core objective:**
The notebook should help the agent solve future AppWorld tasks more quickly and reliably.

**What belongs in the notebook:**
- API documentation: function names, parameter meanings, output shapes, side effects, preconditions, constraints, error patterns, and when to use an API
- App logic: what an app can or cannot do, and cross-app workflows
- Problem-solving strategy: when to explore, when to print data, when to batch actions, and how to verify assumptions efficiently
- Reusable hypotheses only if clearly marked as uncertain

**What to avoid:**
- Trace-by-trace retellings
- One-off facts with low reuse value
- Duplicate or overlapping items
- Vague generic advice
- Bloated or multi-topic items

**Editing policy:**
Use these operations only:
1. `edit`
2. `add`
3. `delete`

Prefer:
1. editing an existing item if new information fits that topic
2. deleting or replacing misleading / redundant items
3. adding a new item only when the topic is genuinely new

**Structural rules:**
- Every item must begin with a short title
- Keep items concise and retrieval-friendly
- New items should be inserted near the most related existing item using `after_tag` when possible
- For `add`, do NOT provide a `tag`; the system will assign one
- New tags in this method always use the `item-*` format
- If omitted, `section` defaults to `item` for this flat notebook format

**Output format:**
Return JSON only: a list of operations.

Allowed examples:
[
  {
    "action": "edit",
    "tag": "item-00005",
    "content": "New title: Revised item content"
  },
  {
    "action": "add",
    "section": "item",
    "after_tag": "item-00005",
    "content": "New title: New item content"
  },
  {
    "action": "delete",
    "tag": "item-00003"
  }
]

**Inputs:**

Current context:
<<<CURRENT_CONTEXT>>>
{{ current_context }}
<<<END_CURRENT_CONTEXT>>>

Task instruction:
<<<TASK_INSTRUCTION>>>
{{ task_instruction }}
<<<END_TASK_INSTRUCTION>>>

Full trace:
<<<FULL_TRACE>>>
{{ full_trace }}
<<<END_FULL_TRACE>>>

Test report:
<<<TEST_REPORT>>>
{{ test_report }}
<<<END_TEST_REPORT>>>

Success: {{ success }}

**Output requirement:**
Return ONLY a valid JSON list of operations. No markdown. No explanation. No code fences.
"""


class SummaryDeltaV1Model(BaseModel):
    name = "summary_delta_v1"

    def update_context(
        self,
        llm_client,
        current_context,
        task_instruction,
        full_trace,
        test_report,
        success,
        ground_truth_code="",
    ):
        prompt = Template(SUMMARIZE_DELTA_PROMPT).render(
            current_context=current_context,
            task_instruction=task_instruction,
            full_trace=render_conversation_history(full_trace),
            test_report=test_report,
            success=success,
        )
        raw = llm_client.generate([{"role": "user", "content": prompt}])["content"]
        if not raw.strip():
            print("[summary_delta_v1] Warning: empty model response; applying no-op delta.")
            operations = []
        else:
            try:
                operations = extract_json_payload(raw)
            except Exception as exc:
                print(
                    f"[summary_delta_v1] Warning: failed to parse delta response; "
                    f"applying no-op delta. Error: {exc}"
                )
                operations = []
        new_context, normalized_ops = apply_tagged_operations(current_context, operations)
        return new_context, normalized_ops
