import json
import re

from jinja2 import Template

from common import BaseModel, extract_json_payload, render_conversation_history

OBSERVATION_PROMPT = """You are an AppWorld notebook analyst. Your job is to extract reusable environment observations from the latest task trace.

These observations are not final notebook items yet. They are natural-language analytical notes that will later be rewritten into notebook items.

**Goal:**
- Record what the trace showed
- Explain what that implies for future action
- Mention anything surprising or mismatched with the prior assumption when relevant

**Style requirements:**
- Write plain natural-language observations, not notebook items
- Each observation should describe:
  1. what was observed
  2. what that implies
  3. what was surprising, mismatched, or still uncertain if relevant
- It is okay to refer to this run, this trace, or the test report here
- Keep observations grounded in the trace and test report
- Do not output vague advice with no evidence

**Examples:**

Example 1:
Trace evidence:
- `song['like_count']` raised `KeyError`
- A printed playlist song looked like `{'id': 136, 'title': 'Finding Solace in the Abyss', 'artist_ids': [9]}`

Good output observation:
"The playlist song objects in this trace exposed only sparse identifiers such as `id`, `title`, and `artist_ids`, while `like_count` access failed with `KeyError`. This suggests playlist detail is not a reliable source for popularity metadata. The mismatch is that the task wording implied a like-related signal, but the observed object shape did not expose one."

Example 2:
Trace evidence:
- The agent only called `spotify.login(...)`
- The test report passed `assert no model changes`

Good output observation:
"The trace showed successful Spotify authentication without any model changes, and the evaluator explicitly passed the no-model-changes check. This implies login itself is read-only in this environment. Nothing in this run contradicted that assumption."

**Output JSON format:**
{
  "reasoning": "Short reasoning about what the trace revealed and why these observations matter for future action.",
  "observations": [
    "Observation 1",
    "Observation 2"
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


QUESTION_PROMPT = """You are an AppWorld notebook analyst. Your job is to identify the highest-value questions the agent should investigate in future tasks.

These questions are not final notebook items yet. They are natural-language unresolved questions that will later be rewritten into notebook items.

**Goal:**
- Point to what the agent should test, verify, or distinguish next time
- Use the trace to motivate the uncertainty
- Prefer questions whose answers would change future action

**Style requirements:**
- Write plain natural-language unresolved questions, not notebook items
- Each question should describe:
  1. what remains unclear
  2. why that uncertainty matters
  3. what evidence raised the question
- It is okay to mention what kind of future probe could resolve the uncertainty
- Do not output broad research wishlists detached from the trace

**Examples:**

Example 1:
Trace evidence:
- Playlist song objects had no user-liked field
- The task wording referred to songs "I have liked"

Good output question:
"It is still unclear whether liked-song membership comes from a separate saved-songs endpoint or from song detail, because playlist song objects in this trace did not expose a user-liked field even though the task referred to songs the user had liked. Resolving this would change how future playlist filtering tasks should be approached."

Example 2:
Trace evidence:
- `create_song_review` raised `No API named ... found`
- The evaluator expected added and updated `spotify.SongReview` records

Good output question:
"It remains unclear how SongReview creation versus update is supposed to work, because this trace ruled out `create_song_review` while the evaluator still expected SongReview mutations. That uncertainty matters because future review tasks may require a different write endpoint or a separate lookup of existing review ids."

**Output JSON format:**
{
  "reasoning": "Short reasoning about what remains uncertain and which probes would most improve future performance.",
  "questions": [
    "Question 1",
    "Question 2"
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


NOTEBOOK_INTEGRATOR_PROMPT = """You maintain an AppWorld notebook with two sections:
- `observation`
- `question`

The rendered notebook headers are `## OBSERVATIONS` and `## OPEN QUESTIONS`.

**Goal:**
- Keep a compact notebook that improves future action selection
- Preserve strong items, sharpen them when new evidence arrives, and add only high-value new ones
- Remove resolved or low-value items

**Item style requirements:**
- Every notebook item must be understandable without extra context
- Start with a short action-oriented title that tells the agent what to do
- Then combine that action with the reason, evidence, implication, or uncertainty
- Keep each item concise and self-contained
- Rewrite away references such as "this trace", "this run", "this task", or "the agent above"
- The final notebook item must still make sense when read alone in a future task
- Prefer editing an existing related item over adding a near-duplicate
- For `add`, do NOT provide a `tag`; the system will assign `obs-*` or `que-*`

**Examples:**

Example 1:
Current notebook has:
- `obs-00003`: "Inspect playlist song fields before ranking: ..."

New observation says:
- playlist song objects again lacked `rating`
- the newer trace reinforces that playlist detail is sparse

Good operation:
{
  "action": "edit",
  "tag": "obs-00003",
  "section": "observation",
  "content": "Inspect playlist song fields before ranking: Multiple observations showed playlist song objects exposing only sparse identifiers, while accesses for fields like `like_count` and `rating` failed. This implies playlist songs are not reliable popularity objects and should be probed before using them for ranking."
}

Example 2:
New question says:
- it is unclear how SongReview creation versus update works
- the failed guessed API and evaluator expectations point to a hidden write workflow

Good operation:
{
  "action": "add",
  "section": "question",
  "after_tag": "que-00004",
  "content": "Check the review write path before planning bulk updates: Guessed create-review APIs may be missing even when evaluation expects SongReview mutations. Verify which create or update endpoint performs review writes and whether existing review ids are needed before attempting bulk rating changes."
}

**Output JSON format:**
{
  "reasoning": "Short reasoning about what to add, edit, or delete.",
  "operations": [
    {"action": "add", "section": "observation", "after_tag": "obs-00003", "content": "Short title: New item."},
    {"action": "edit", "tag": "obs-00003", "section": "observation", "content": "Short title: Revised item."},
    {"action": "delete", "tag": "que-00007"}
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
<<<QUESTIONS>>>
{{ questions_json }}
<<<END_QUESTIONS>>>

**Output requirement:**
Return ONLY a valid JSON object. No markdown. No code fences.
"""


def _normalize_section_name(section: str) -> str:
    normalized = section.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized.endswith("s") and len(normalized) > 1:
        normalized = normalized[:-1]
    return normalized


def _prefix_for_section(section: str) -> str:
    normalized = _normalize_section_name(section)
    if normalized == "observation":
        return "obs"
    if normalized == "question":
        return "que"
    return "item"


def _section_from_tag(tag: str) -> str:
    if tag.startswith("obs-"):
        return "observation"
    if tag.startswith("que-"):
        return "question"
    return "item"


def _parse_tagged_context(context: str) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    current_section = ""
    for raw_line in context.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("## "):
            header = line[3:].strip()
            if header == "OPEN QUESTIONS":
                current_section = "question"
            else:
                current_section = _normalize_section_name(header)
            continue
        match = re.match(r"\[(?P<tag>[^\]]+)\]\s*(?P<content>.*)", line)
        if match:
            tag = match.group("tag")
            section = current_section or _section_from_tag(tag)
            items.append({"tag": tag, "content": match.group("content"), "section": section})
    return items


def _render_notebook(items: list[dict[str, str]]) -> str:
    observations = [item for item in items if item.get("section") == "observation"]
    questions = [item for item in items if item.get("section") == "question"]
    blocks = ["## OBSERVATIONS"]
    blocks.extend(f"[{item['tag']}] {item['content']}" for item in observations)
    blocks.extend(["", "## OPEN QUESTIONS"])
    blocks.extend(f"[{item['tag']}] {item['content']}" for item in questions)
    return "\n".join(blocks).strip()


def _apply_hypothesis_operations(
    context: str,
    operations: list[dict],
) -> tuple[str, list[dict]]:
    items = _parse_tagged_context(context)
    index_by_tag = {item["tag"]: idx for idx, item in enumerate(items)}
    next_indices = {"obs": 1, "que": 1}
    for item in items:
        prefix = _prefix_for_section(item.get("section", ""))
        match = re.search(r"(\d+)$", item["tag"])
        if prefix in next_indices and match:
            next_indices[prefix] = max(next_indices[prefix], int(match.group(1)) + 1)

    normalized_ops: list[dict] = []
    for op in operations:
        action = op["action"].lower()
        if action == "edit":
            tag = op["tag"]
            if tag in index_by_tag:
                section = _normalize_section_name(op.get("section", items[index_by_tag[tag]]["section"]))
                items[index_by_tag[tag]]["content"] = op["content"].strip()
                items[index_by_tag[tag]]["section"] = section
                normalized_ops.append(op)
        elif action == "delete":
            tag = op["tag"]
            if tag in index_by_tag:
                items.pop(index_by_tag[tag])
                index_by_tag = {item["tag"]: idx for idx, item in enumerate(items)}
                normalized_ops.append(op)
        elif action == "add":
            section = _normalize_section_name(op.get("section", ""))
            prefix = _prefix_for_section(section)
            next_index = next_indices.get(prefix, 1)
            tag = f"{prefix}-{next_index:05d}"
            next_indices[prefix] = next_index + 1
            item = {"tag": tag, "content": op["content"].strip(), "section": section}
            after_tag = op.get("after_tag")
            if after_tag and after_tag in index_by_tag:
                insert_at = index_by_tag[after_tag] + 1
                items.insert(insert_at, item)
            else:
                items.append(item)
            index_by_tag = {entry["tag"]: idx for idx, entry in enumerate(items)}
            normalized = dict(op)
            normalized["tag"] = tag
            normalized_ops.append(normalized)

    return _render_notebook(items), normalized_ops


class HypothesisV2Model(BaseModel):
    name = "hypothesis_v2"

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
            print("[hypothesis_v2] Warning: empty observation response; using empty observations.")
            observations = {"reasoning": "", "observations": []}
        else:
            try:
                observations = extract_json_payload(observation_raw)
            except Exception as exc:
                print(
                    f"[hypothesis_v2] Warning: failed to parse observation response; "
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
            print("[hypothesis_v2] Warning: empty question response; using empty questions.")
            questions = {"reasoning": "", "questions": []}
        else:
            try:
                questions = extract_json_payload(question_raw)
            except Exception as exc:
                print(
                    f"[hypothesis_v2] Warning: failed to parse question response; "
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
            print("[hypothesis_v2] Warning: empty integrator response; applying no-op delta.")
            operations_payload = {"reasoning": "", "operations": []}
        else:
            try:
                operations_payload = extract_json_payload(integrator_raw)
            except Exception as exc:
                print(
                    f"[hypothesis_v2] Warning: failed to parse integrator response; "
                    f"applying no-op delta. Error: {exc}"
                )
                operations_payload = {"reasoning": "", "operations": []}

        operations = operations_payload.get("operations", [])
        new_context, normalized_ops = _apply_hypothesis_operations(current_context, operations)
        return new_context, normalized_ops
