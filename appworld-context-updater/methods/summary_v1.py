from jinja2 import Template

from common import BaseModel, render_conversation_history

SUMMARIZE_REWRITE_PROMPT = """You are an expert AppWorld learning-context editor. Your job is to revise the entire context after one task so that it becomes a stronger reusable notebook for future AppWorld tasks.

**Goal:**
Produce a full replacement context text. The output will be inserted into the generator prompt for future tasks, so it should preserve useful prior knowledge while integrating new learnings from the latest trajectory.

**What you receive:**
- The current full context
- The task instruction
- The full trace for this task, containing only code and execution results per turn
- The final test report
- Whether the task succeeded

**What good output looks like:**
- Compact and high-signal
- Focused on reusable AppWorld knowledge, not this single task narrative
- Actionable: APIs, argument semantics, workflows, failure patterns, verification habits, and reliable strategies
- Specific enough to change future behavior
- Free of redundancy, stale content, and weak generic advice

**What to preserve:**
- Existing context items that remain useful and accurate
- Hard-won environment knowledge that generalizes
- Strategy guidance that improves future efficiency or reliability

**What to add or revise:**
- API behavior discovered in the trace
- Preconditions, side effects, pagination behavior, output schemas, and common failure modes
- Better decomposition strategies if the trace or test report shows an important planning failure
- Corrections to existing context when the latest trace contradicts or sharpens it

**What to avoid:**
- Do not retell the whole trace
- Do not mention task IDs
- Do not include chain-of-thought or analysis notes
- Do not include markdown fences
- Do not add generic advice like "be careful" unless it is concretely operationalized

**Important:**
- The output replaces the entire prior context verbatim
- Use only information supported by the provided context, trace, and test report
- No ground truth code is available

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
Return ONLY the revised full context text. No JSON. No explanation. No code fences.
"""


class SummaryV1Model(BaseModel):
    name = "summary_v1"

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
        prompt = Template(SUMMARIZE_REWRITE_PROMPT).render(
            current_context=current_context,
            task_instruction=task_instruction,
            full_trace=render_conversation_history(full_trace),
            test_report=test_report,
            success=success,
        )
        updated = llm_client.generate([{"role": "user", "content": prompt}])["content"].strip()
        if not updated:
            print("[summary_v1] Warning: empty model response; keeping current context.")
            return current_context, current_context
        return updated, updated
