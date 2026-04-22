import json

from jinja2 import Template

from common import (
    BaseModel,
    apply_tagged_operations,
    extract_json_payload,
    render_conversation_history,
)

OBSERVATION_PROMPT = """You are an AppWorld notebook analyst. Your job is to read the latest task trace and extract what the agent actually learned about the environment.

This notebook is different from a normal declarative playbook. It should capture observations grounded in evidence, including uncertainty when the trace does not fully resolve something.

**Your task:**
- Analyze what happened in the trace
- Identify API behaviors, workflow patterns, assumptions that held or broke, and concrete environment facts learned from the interaction
- When an existing notebook item is relevant, note whether the new trace supports, sharpens, or contradicts it

**Important:**
- Express certainty through language, not labels
- Focus on reusable environment knowledge
- Do not summarize the whole task
- Do not output strategy advice as an observation unless it is directly tied to observed environment behavior

**Output JSON format:**
{
  "reasoning": "Chain of thought analyzing what happened in the trace, what assumptions held or broke, what patterns emerged...",
  "observations": [
    "Short title: Observation grounded in evidence from the trace and test report."
  ]
}

**Inputs:**

Current notebook:
<<<CURRENT_NOTEBOOK>>>
{{ current_context }}
<<<END_CURRENT_NOTEBOOK>>>

Full trace:
<<<FULL_TRACE>>>
{{ full_trace }}
<<<END_FULL_TRACE>>>

Test report:
<<<TEST_REPORT>>>
{{ test_report }}
<<<END_TEST_REPORT>>>

**Output requirement:**
Return ONLY a valid JSON object. No markdown. No code fences.
"""


QUESTION_PROMPT = """You are an AppWorld notebook analyst. Your job is to identify what the agent still does not know about the environment and what would be most valuable to investigate in future tasks.

These are open questions, not factual notes. They should help the agent actively discover more through interaction.

**Your task:**
- Identify unresolved uncertainties about apps, APIs, argument semantics, data shapes, side effects, or workflow choices
- Prefer questions that are worth spending turns on in future tasks
- Consider the cost of investigation: recommend probing when the risk of being wrong is high or the API is unfamiliar, but avoid wasting turns on already-settled patterns

**Important:**
- Do not simply restate observations
- Do not produce generic advice
- Questions should be operational and future-facing

**Output JSON format:**
{
  "reasoning": "Chain of thought about what the agent still doesn't know and what would be most valuable to investigate...",
  "questions": [
    "Short title: Concrete future-facing question or investigation plan."
  ]
}

**Inputs:**

Current notebook:
<<<CURRENT_NOTEBOOK>>>
{{ current_context }}
<<<END_CURRENT_NOTEBOOK>>>

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

**Output requirement:**
Return ONLY a valid JSON object. No markdown. No code fences.
"""


NOTEBOOK_INTEGRATOR_PROMPT = """You maintain an AppWorld notebook as tagged items organized into two sections:
- `observation`
- `question`

The rendered notebook headers are `## OBSERVATIONS` and `## OPEN QUESTIONS`, but the operation payload must always use the singular internal names `observation` and `question`.

**Your task:**
- Decide how the notebook should change after reading the new observations and open questions
- Add genuinely new items
- Edit existing items when new evidence sharpens or corrects them
- Delete items that have been resolved, superseded, or are no longer good notebook entries

**Content rules:**
- Each item's content must start with a short title, followed by concise detail
- Each item must stay within 150 words
- Keep the notebook compact, specific, and reusable
- Prefer editing over adding when the topic already exists
- Use `after_tag` when inserting near a related existing item
- For `add`, do NOT provide a `tag`; the system will assign one using only `observation-*` or `question-*`

**Output JSON format:**
{
  "reasoning": "Chain of thought about what to add/edit/delete in the notebook...",
  "operations": [
    {"action": "add", "section": "observation", "after_tag": "observation-00003", "content": "Short title: Item content..."},
    {"action": "edit", "tag": "observation-00003", "section": "observation", "content": "Short title: Revised content..."},
    {"action": "delete", "tag": "question-00007"}
  ]
}

**Inputs:**

Current notebook:
<<<CURRENT_NOTEBOOK>>>
{{ current_context }}
<<<END_CURRENT_NOTEBOOK>>>

New observations:
<<<OBSERVATIONS>>>
{{ observations_json }}
<<<END_OBSERVATIONS>>>

New questions:
<<<OPEN_QUESTIONS>>>
{{ questions_json }}
<<<END_OPEN_QUESTIONS>>>

**Output requirement:**
Return ONLY a valid JSON object. No markdown. No code fences.
"""


class HypothesisV1Model(BaseModel):
    name = "hypothesis_v1"

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
        full_trace_text = render_conversation_history(full_trace)
        observation_raw = llm_client.generate(
            [{"role": "user", "content": Template(OBSERVATION_PROMPT).render(
                current_context=current_context,
                full_trace=full_trace_text,
                test_report=test_report,
            )}]
        )["content"]
        if not observation_raw.strip():
            print("[hypothesis_v1] Warning: empty observation response; using empty observations.")
            observations = {"reasoning": "", "observations": []}
        else:
            try:
                observations = extract_json_payload(observation_raw)
            except Exception as exc:
                print(
                    f"[hypothesis_v1] Warning: failed to parse observation response; "
                    f"using empty observations. Error: {exc}"
                )
                observations = {"reasoning": "", "observations": []}

        question_raw = llm_client.generate(
            [{"role": "user", "content": Template(QUESTION_PROMPT).render(
                current_context=current_context,
                task_instruction=task_instruction,
                full_trace=full_trace_text,
                test_report=test_report,
            )}]
        )["content"]
        if not question_raw.strip():
            print("[hypothesis_v1] Warning: empty question response; using empty questions.")
            questions = {"reasoning": "", "questions": []}
        else:
            try:
                questions = extract_json_payload(question_raw)
            except Exception as exc:
                print(
                    f"[hypothesis_v1] Warning: failed to parse question response; "
                    f"using empty questions. Error: {exc}"
                )
                questions = {"reasoning": "", "questions": []}

        integrator_raw = llm_client.generate(
            [{"role": "user", "content": Template(NOTEBOOK_INTEGRATOR_PROMPT).render(
                current_context=current_context,
                observations_json=json.dumps(observations, indent=2),
                questions_json=json.dumps(questions, indent=2),
            )}]
        )["content"]
        if not integrator_raw.strip():
            print("[hypothesis_v1] Warning: empty integrator response; applying no-op delta.")
            operations_payload = {"reasoning": "", "operations": []}
        else:
            try:
                operations_payload = extract_json_payload(integrator_raw)
            except Exception as exc:
                print(
                    f"[hypothesis_v1] Warning: failed to parse integrator response; "
                    f"applying no-op delta. Error: {exc}"
                )
                operations_payload = {"reasoning": "", "operations": []}
        operations = operations_payload.get("operations", [])
        new_context, normalized_ops = apply_tagged_operations(
            current_context,
            operations,
            include_sections=True,
        )
        return new_context, normalized_ops
