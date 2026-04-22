"""
ACE (Accumulated Coding Experience) context updater.

Faithful reimplementation of the official ACE repo's reflector-curator pipeline.
Reference: libs/ace-appworld/experiments/code/ace/adaptation_react.py

Pipeline per task:
  1. Solve task with PlaybookReActAgent (done by harness in common.py)
  2. Reflector: diagnose trajectory using GT code, test report, execution errors,
     conversation history, and current playbook
  3. Curator: update playbook with new insights from the reflection

Ablation switches (class-level):
  - use_ground_truth: include ground-truth code in reflector input
  - use_test_report: include test report in reflector input
  - initial_playbook: "default" (initial), "empty", or "null"
"""

import json
import re
from pathlib import Path
from typing import Any

from jinja2 import Template

from common import (
    BaseModel,
    LLMClient,
    extract_json_payload,
    load_text,
    render_conversation_history,
    strip_reasoning_blocks,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ACE_PLAYBOOKS = REPO_ROOT / "libs" / "ace-appworld" / "experiments" / "playbooks"
APPWORLD_INITIAL_PLAYBOOK = ACE_PLAYBOOKS / "appworld_initial_playbook.txt"
APPWORLD_EMPTY_PLAYBOOK = ACE_PLAYBOOKS / "appworld_empty_playbook.txt"
APPWORLD_NULL_PLAYBOOK = ACE_PLAYBOOKS / "appworld_null_playbook.txt"

PLAYBOOK_MAP = {
    "default": APPWORLD_INITIAL_PLAYBOOK,
    "empty": APPWORLD_EMPTY_PLAYBOOK,
    "null": APPWORLD_NULL_PLAYBOOK,
}

ALLOWED_SECTIONS = {
    "strategies_and_hard_rules",
    "apis_to_use_for_specific_information",
    "useful_code_snippets_and_templates",
    "common_mistakes_and_correct_strategies",
    "problem_solving_heuristics_and_workflows",
    "verification_checklist",
    "troubleshooting_and_pitfalls",
    "others",
}


# ---------------------------------------------------------------------------
# Playbook helpers
# ---------------------------------------------------------------------------

def get_section_slug(section_name: str) -> str:
    slug_map = {
        "strategies_and_hard_rules": "shr",
        "hard_rules": "hr",
        "strategies_and_insights": "si",
        "apis_to_use_for_specific_information": "api",
        "code_snippets_and_templates": "code",
        "useful_code_snippets_and_templates": "code",
        "common_mistakes_to_avoid": "err",
        "common_mistakes_and_correct_strategies": "cms",
        "problem_solving_heuristics": "prob",
        "problem_solving_heuristics_and_workflows": "psw",
        "verification_checklist": "vc",
        "troubleshooting_and_pitfalls": "ts",
        "others": "misc",
        "meta_strategies": "meta",
    }
    clean_name = section_name.lower().strip().replace(" ", "_").replace("&", "and").rstrip(":")
    if clean_name in slug_map:
        return slug_map[clean_name]
    words = clean_name.split("_")
    if len(words) == 1:
        return words[0][:4]
    return "".join(word[0] for word in words[:5])


def parse_playbook_line(line: str) -> dict | None:
    text = line.strip()
    pattern_full = r"\[([^\]]+)\]\s*helpful=(\d+)\s*harmful=(\d+)\s*::\s*(.*)"
    match = re.match(pattern_full, text)
    if match:
        return {
            "id": match.group(1),
            "helpful": int(match.group(2)),
            "harmful": int(match.group(3)),
            "content": match.group(4),
            "raw_line": line,
        }
    pattern_simple = r"\[([^\]]+)\]\s*(.*)"
    match = re.match(pattern_simple, text)
    if match:
        return {
            "id": match.group(1),
            "helpful": 0,
            "harmful": 0,
            "content": match.group(2).strip(),
            "raw_line": line,
        }
    return None


def get_next_global_id(playbook_text: str) -> int:
    max_id = 0
    for line in playbook_text.strip().split("\n"):
        parsed = parse_playbook_line(line)
        if not parsed:
            continue
        id_match = re.search(r"-(\d+)$", parsed["id"])
        if id_match:
            max_id = max(max_id, int(id_match.group(1)))
    return max_id + 1


def format_playbook_line(bullet_id: str, content: str) -> str:
    return f"[{bullet_id}] {content}"


def apply_curator_operations(playbook_text: str, operations: list[dict], next_id: int) -> tuple[str, int]:
    lines = playbook_text.strip().split("\n")
    sections: dict[str, list[tuple[int, str]]] = {}
    current_section = "general"

    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("##"):
            section_header = stripped[2:].strip()
            current_section = section_header.lower().replace(" ", "_").replace("&", "and").rstrip(":")
            sections.setdefault(current_section, [])
        elif stripped:
            sections.setdefault(current_section, []).append((index, line))

    bullets_to_add: list[tuple[str, str]] = []
    for op in operations:
        if op.get("type") != "ADD":
            continue
        section_raw = op.get("section", "general")
        section = section_raw.lower().replace(" ", "_").replace("&", "and").rstrip(":")
        if section not in sections and section != "general":
            print(f"Warning: Section '{section_raw}' not found, adding to OTHERS")
            section = "others"
        slug = get_section_slug(section)
        new_id = f"{slug}-{next_id:05d}"
        next_id += 1
        bullets_to_add.append((section, format_playbook_line(new_id, op.get("content", ""))))
        print(f"  Added bullet {new_id} to section {section}")

    final_lines: list[str] = []
    current_section = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("##"):
            if current_section:
                section_adds = [bullet for section, bullet in bullets_to_add if section == current_section]
                final_lines.extend(section_adds)
                bullets_to_add = [(section, bullet) for section, bullet in bullets_to_add if section != current_section]
            section_header = stripped[2:].strip()
            current_section = section_header.lower().replace(" ", "_").replace("&", "and").rstrip(":")
        final_lines.append(line)

    if current_section:
        section_adds = [bullet for section, bullet in bullets_to_add if section == current_section]
        final_lines.extend(section_adds)
        bullets_to_add = [(section, bullet) for section, bullet in bullets_to_add if section != current_section]

    if bullets_to_add:
        print(f"Warning: {len(bullets_to_add)} bullets have no matching section, adding to OTHERS")
        others_bullets = [bullet for _, bullet in bullets_to_add]
        others_index = -1
        for index, line in enumerate(final_lines):
            if line.strip() == "## OTHERS":
                others_index = index
                break
        if others_index >= 0:
            for index, bullet in enumerate(others_bullets):
                final_lines.insert(others_index + 1 + index, bullet)
        else:
            final_lines.extend(others_bullets)

    return "\n".join(final_lines), next_id


# ---------------------------------------------------------------------------
# Prompts -- copied verbatim from the official ACE repo prompt files, adapted
# for Jinja2 Template.render() ({{ var }} instead of .replace()).
#
# Source: libs/ace-appworld/experiments/prompts/appworld_react_reflector_with_gt_prompt.txt
# Source: libs/ace-appworld/experiments/prompts/appworld_react_curator_prompt.txt
# ---------------------------------------------------------------------------

REFLECTOR_PROMPT = """\
You are an expert AppWorld coding agent and educator. Your job is to diagnose the current trajectory: identify what went wrong (or could be better), grounded in execution feedback, API usage, unit test report, and ground truth when applicable.

**Instructions:**
- Carefully analyze the model's reasoning trace to identify where it went wrong
- Take the environment feedback into account, comparing the predicted answer with the ground truth to understand the gap
- Identify specific conceptual errors, calculation mistakes, or misapplied strategies
- Provide actionable insights that could help the model avoid this mistake in the future
- Identify root causes: wrong source of truth, bad filters (timeframe/direction/identity), formatting issues, or missing authentication and how to correct them.
- Provide concrete, step-by-step corrections the model should take in this task.
- Be specific about what the model should have done differently
- You will receive bulletpoints that are part of playbook that's used by the generator to answer the question.
- You need to analyze these bulletpoints, and give the tag for each bulletpoint, tag can be ['helpful', 'harmful', 'neutral'] (for the generator to generate the correct answer)
- Explicitly curate from the environment feedback the output format/schema of APIs used when unclear or mismatched with expectations (e.g., `apis.blah.show_contents()` returns a list of content_ids (strings), not content objects)

**Inputs:**
- Ground truth code (reference, known-correct):
<<<GROUND_TRUTH_CODE_START>>>
{{ ground_truth_code }}
<<<GROUND_TRUTH_CODE_END>>>

- Generated code (candidate to critique):
<<<GENERATED_CODE_START>>>
{{ generated_code }}
<<<GENERATED_CODE_END>>>

- Execution error (if the generated code was run and failed):
<<<EXECUTION_ERROR_START>>>
{{ execution_error }}
<<<EXECUTION_ERROR_END>>>

- Test report (unit tests result for the task after the generated code was run):
<<<TEST_REPORT>>>
{{ test_report }}
<<<TEST_REPORT>>>

- (Optional) Generated plan/reflection/comments:
<<<GENERATED_RATIONALE_START>>>
{{ generated_rationale }}
<<<GENERATED_RATIONALE_END>>>

- (Optional) Task spec / API docs excerpt (if available):
<<<SPEC_OR_API_START>>>
{{ spec_or_api_docs }}
<<<SPEC_OR_API_END>>>

- (Optional) Playbook (playbook that's used by model for code generation):
<<<PLAYBOOK_GUIDE>>>
{{ playbook }}
<<<PLAYBOOK_GUIDE>>>

- (Optional) Reflections (reflection of error from a prior review pass):
<<<PRIOR_REFLECTION>>>
{{ previous_reflection }}
<<<PRIOR_REFLECTION>>>

**Examples:**

**Example 1:**
Ground Truth Code: [Code that uses apis.phone.search_contacts() to find roommates, then filters Venmo transactions]
Generated Code: [Code that tries to identify roommates by parsing Venmo transaction descriptions using keywords like "rent", "utilities"]
Execution Error: AssertionError: Expected 1068.0 but got 79.0
Test Report: FAILED - Wrong total amount calculated due to incorrect roommate identification

Response:
{
  "reasoning": "The generated code attempted to identify roommates by parsing Venmo transaction descriptions rather than using the authoritative Phone app contacts. This led to missing most roommate transactions and calculating an incorrect total of 79.0 instead of 1068.0.",
  "error_identification": "The agent used unreliable heuristics (keyword matching in transaction descriptions) to identify roommates instead of the correct API (Phone contacts).",
  "root_cause_analysis": "The agent misunderstood the data architecture - it assumed transaction descriptions contained reliable relationship information, when the Phone app is the authoritative source for contact relationships.",
  "correct_approach": "First authenticate with Phone app, use apis.phone.search_contacts() to identify contacts with 'roommate' relationship, then filter Venmo transactions by those specific contact emails/phone numbers.",
  "key_insight": "Always resolve identities from the correct source app - Phone app for relationships, never rely on transaction descriptions or other indirect heuristics which are unreliable."
}

**Example 2:**
Ground Truth Code: [Code that uses proper while True pagination loop to get all Spotify playlists]
Generated Code: [Code that uses for i in range(10) to paginate through playlists]
Execution Error: None (code ran successfully)
Test Report: FAILED - Expected 23 playlists but got 10 due to incomplete pagination

Response:
{
  "reasoning": "The generated code used a fixed range loop (range(10)) for pagination instead of properly iterating until no more results are returned. This caused the agent to only collect the first 10 pages of playlists, missing 13 additional playlists that existed on later pages.",
  "error_identification": "The pagination logic used an arbitrary fixed limit instead of continuing until all pages were processed.",
  "root_cause_analysis": "The agent used a cautious approach with a fixed upper bound to avoid infinite loops, but this prevented complete data collection when the actual data exceeded the arbitrary limit.",
  "correct_approach": "Use while True loop with proper break condition: continue calling the API with incrementing page_index until the API returns empty results or null, then break.",
  "key_insight": "For pagination, always use while True loop instead of fixed range iterations to ensure complete data collection across all available pages."
}

**Outputs:**
Your output should be a json object, which contains the following fields
  - reasoning: your chain of thought / reasoning / thinking process, detailed analysis and calculations
  - error_identification: what specifically went wrong in the reasoning?
  - root_cause_analysis: why did this error occur? What concept was misunderstood?
  - correct_approach: what should the model have done instead?
  - key_insight: what strategy, formula, or principle should be remembered to avoid this error?

**Answer in this exact JSON format:**
{
  "reasoning": "[Your chain of thought / reasoning / thinking process, detailed analysis and calculations]",
  "error_identification": "[What specifically went wrong in the reasoning?]",
  "root_cause_analysis": "[Why did this error occur? What concept was misunderstood?]",
  "correct_approach": "[What should the model have done instead?]",
  "key_insight": "[What strategy, formula, or principle should be remembered to avoid this error?]",
}"""

CURATOR_PROMPT = """\
You are a master curator of knowledge. Your job is to identify what new insights should be added to an existing playbook based on a reflection from a previous attempt.

**Context:**
- The playbook you created will be used to help answering similar questions.
- The reflection is generated using ground truth answers that will NOT be available when the playbook is being used. So you need to come up with content that can aid the playbook user to create predictions that likely align with ground truth.

**Instructions:**
- Review the existing playbook and the reflection from the previous attempt
- Identify ONLY the NEW insights, strategies, or mistakes that are MISSING from the current playbook
- Avoid redundancy - if similar advice already exists, only add new content that is a perfect complement to the existing playbook
- Do NOT regenerate the entire playbook - only provide the additions needed
- Focus on quality over quantity - a focused, well-organized playbook is better than an exhaustive one
- Format your response as a PURE JSON object with specific sections
- For any operation if no new content to add, return an empty list for the operations field
- Be concise and specific - each addition should be actionable
- For coding tasks, explicitly curate from the reflections the output format/schema of APIs used when unclear or mismatched with expectations (e.g., `apis.blah.show_contents()` returns a list of content_ids (strings), not content objects)

- **Task Context (the actual task instruction):**
  `{{ question_context }}`

- **Current Playbook:**
  `{{ current_playbook }}`

- **Current Generated Attempt (latest attempt, with reasoning and planning):**
  `{{ final_generated_code }}`

- **Current Reflections (principles and strategies that helped to achieve current task):**
  `{{ guidebook }}`


**Examples:**

**Example 1:**
Task Context: "Find money sent to roommates since Jan 1 this year"
Current Playbook: [Basic API usage guidelines]
Generated Attempt: [Code that failed because it used transaction descriptions to identify roommates instead of Phone contacts]
Reflections: "The agent failed because it tried to identify roommates by parsing Venmo transaction descriptions instead of using the Phone app's contact relationships. This led to incorrect identification and wrong results."

Response:
{
  "reasoning": "The reflection shows a critical error where the agent used unreliable heuristics (transaction descriptions) instead of the authoritative source (Phone app contacts) to identify relationships. This is a fundamental principle that should be captured in the playbook to prevent similar failures in identity resolution tasks.",
  "operations": [
    {
      "type": "ADD",
      "section": "strategies_and_hard_rules",
      "content": "Always resolve identities from the correct source app\\n- When you need to identify relationships (roommates, contacts, etc.), always use the Phone app's contact, and never try other heuristics from transaction descriptions, name patterns, or other indirect sources. These heuristics are unreliable and will cause incorrect results."
    }
  ]
}

**Example 2:**
Task Context: "Count all playlists in Spotify"
Current Playbook: [Basic authentication and API calling guidelines]
Generated Attempt: [Code that used for i in range(10) loop and missed playlists on later pages]
Reflections: "The agent used a fixed range loop for pagination instead of properly iterating through all pages until no more results are returned. This caused incomplete data collection."

Response:
{
  "reasoning": "The reflection identifies a pagination handling error where the agent used an arbitrary fixed range instead of proper pagination logic. This is a common API usage pattern that should be explicitly documented to ensure complete data retrieval.",
  "operations": [
    {
      "type": "ADD",
      "section": "apis_to_use_for_specific_information",
      "content": "About pagination: many APIs return items in \\"pages\\". Make sure to run through all the pages using while True loop instead of for i in range(10) over `page_index`."
    }
  ]
}

**Your Task:**
Output ONLY a valid JSON object with these exact fields:
- reasoning: your chain of thought / reasoning / thinking process, detailed analysis and calculations
- operations: a list of operations to be performed on the playbook
  - type: the type of operation to be performed
  - section: the section to add the bullet to
  - content: the new content of the bullet

**Available Operations:**
1. ADD: Create new bullet points with fresh IDs
    - section: the section to add the new bullet to
    - content: the new content of the bullet. Note: no need to include the bullet_id in the content like '[ctx-00263] helpful=1 harmful=0 ::', the bullet_id will be added by the system.

**RESPONSE FORMAT - Output ONLY this JSON structure (no markdown, no code blocks):**
{
  "reasoning": "[Your chain of thought / reasoning / thinking process, detailed analysis and calculations here]",
  "operations": [
    {
      "type": "ADD",
      "section": "verification_checklist",
      "content": "[New checklist item or API schema clarification...]"
    }
  ]
}"""


# ---------------------------------------------------------------------------
# ACEModel
# ---------------------------------------------------------------------------

class ACEModel(BaseModel):
    """
    ACE context updater faithful to the official repo.

    Ablation switches:
      use_ground_truth  -- pass GT code to the reflector (default True)
      use_test_report   -- pass test report to the reflector (default True)
      initial_playbook  -- "default" | "empty" | "null"
    """

    def __init__(
        self,
        use_ground_truth: bool = True,
        use_test_report: bool = True,
        initial_playbook: str = "default",
    ):
        self.use_ground_truth = use_ground_truth
        self.use_test_report = use_test_report
        self.initial_playbook = initial_playbook
        self.name = self._build_name()
        self._curator_prompt = CURATOR_PROMPT

    def _build_name(self) -> str:
        parts = ["ace"]
        parts.append("gt" if self.use_ground_truth else "nogt")
        parts.append("test" if self.use_test_report else "notest")
        parts.append(f"pb{self.initial_playbook}")
        return "_".join(parts)

    def initialize_context(self) -> str:
        path = PLAYBOOK_MAP[self.initial_playbook]
        if not path.exists():
            return ""
        return load_text(path)

    # ---- reflector --------------------------------------------------------

    def _build_reflector_input(
        self,
        current_context: str,
        full_trace: list[dict[str, Any]],
        test_report: str,
        ground_truth_code: str,
    ) -> str:
        prompt = Template(REFLECTOR_PROMPT).render(
            ground_truth_code=ground_truth_code if self.use_ground_truth else "",
            generated_code="See full conversation history below",
            execution_error="See full conversation history below",
            test_report=test_report if self.use_test_report else "",
            generated_rationale="See full conversation history below",
            spec_or_api_docs="See full conversation history below",
            playbook=current_context,
            previous_reflection="",
        )
        prompt += "\n\n" + render_conversation_history(full_trace)
        return prompt

    def _call_reflector(
        self,
        llm_client: LLMClient,
        current_context: str,
        full_trace: list[dict[str, Any]],
        test_report: str,
        ground_truth_code: str,
    ) -> dict:
        prompt = self._build_reflector_input(
            current_context, full_trace, test_report, ground_truth_code,
        )
        raw = llm_client.generate([{"role": "user", "content": prompt}])["content"]
        if not raw.strip():
            print(f"[{self.name}] Warning: empty reflector response.")
            return {
                "reasoning": "",
                "error_identification": "",
                "root_cause_analysis": "",
                "correct_approach": "",
                "key_insight": "",
            }
        try:
            return extract_json_payload(raw)
        except Exception as exc:
            print(f"[{self.name}] Warning: failed to parse reflector JSON: {exc}")
            return {
                "reasoning": "",
                "error_identification": "",
                "root_cause_analysis": "",
                "correct_approach": "",
                "key_insight": "",
            }

    # ---- curator ----------------------------------------------------------

    def _build_curator_input(
        self,
        current_context: str,
        task_instruction: str,
        reflection: dict,
        full_trace: list[dict[str, Any]],
    ) -> str:
        prompt = Template(self._curator_prompt).render(
            question_context=task_instruction,
            current_playbook=current_context,
            final_generated_code="See full conversation history below",
            guidebook=json.dumps(reflection, indent=2),
        )
        prompt += "\n\n" + render_conversation_history(full_trace)
        return prompt

    def _call_curator(
        self,
        llm_client: LLMClient,
        current_context: str,
        task_instruction: str,
        reflection: dict,
        full_trace: list[dict[str, Any]],
    ) -> dict:
        prompt = self._build_curator_input(
            current_context, task_instruction, reflection, full_trace,
        )
        raw = llm_client.generate([{"role": "user", "content": prompt}])["content"]
        if not raw.strip():
            print(f"[{self.name}] Warning: empty curator response.")
            return {"reasoning": "", "operations": []}
        try:
            return extract_json_payload(raw)
        except Exception as exc:
            print(f"[{self.name}] Warning: failed to parse curator JSON: {exc}")
            return {"reasoning": "", "operations": []}

    # ---- operation validation & application --------------------------------

    def _validate_operations(self, operations: Any) -> list[dict]:
        if not isinstance(operations, list):
            print(f"[{self.name}] Warning: 'operations' field is not a list, skipping.")
            return []
        filtered_ops: list[dict] = []
        for i, op in enumerate(operations):
            if not isinstance(op, dict):
                print(f"  Skipping operation {i}: not a dictionary")
                continue
            if op.get("type") != "ADD":
                print(
                    f"  Skipping operation {i}: invalid type '{op.get('type')}'. Only 'ADD' is supported"
                )
                continue
            missing_fields = {"type", "section", "content"} - set(op.keys())
            if missing_fields:
                print(f"  Skipping operation {i}: ADD missing fields {list(missing_fields)}")
                continue
            section_name = (
                str(op.get("section", "")).strip().lower().replace(" ", "_").replace("&", "and").rstrip(":")
            )
            if section_name not in ALLOWED_SECTIONS:
                print(
                    f"  Skipping operation {i}: disallowed section '{op.get('section')}' "
                    f"(normalized: '{section_name}')"
                )
                continue
            filtered_ops.append(op)
        return filtered_ops

    # ---- main entry point -------------------------------------------------

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
        # Step 1: reflector
        reflection = self._call_reflector(
            llm_client, current_context, full_trace, test_report, ground_truth_code,
        )

        # Step 2: curator
        curator_result = self._call_curator(
            llm_client, current_context, task_instruction, reflection, full_trace,
        )

        # Step 3: validate and apply operations
        operations = self._validate_operations(curator_result.get("operations", []))
        new_context = self._apply_curator_ops(current_context, operations)

        return new_context, operations

    def _apply_curator_ops(self, current_context: str, operations: list[dict]) -> str:
        new_context, _ = apply_curator_operations(
            current_context,
            operations,
            get_next_global_id(current_context),
        )
        return new_context
