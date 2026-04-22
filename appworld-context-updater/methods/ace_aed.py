"""
ace_aed: ACE with Add/Edit/Delete curator operations.

Same reflector as ACE, but the curator supports ADD, EDIT, and DELETE
operations on the sectioned playbook instead of ADD-only.
"""

from methods.ace import (
    ALLOWED_SECTIONS,
    ACEModel,
    format_playbook_line,
    get_next_global_id,
    get_section_slug,
    parse_playbook_line,
)

# ---------------------------------------------------------------------------
# Curator prompt -- ADD / EDIT / DELETE
# ---------------------------------------------------------------------------
CURATOR_AED_PROMPT = """\
You are a master curator of knowledge. Your job is to update an existing playbook based on a reflection from a previous attempt.

**Context:**
- The playbook you created will be used to help answering similar questions.
- The reflection is generated using ground truth answers that will NOT be available when the playbook is being used. So you need to come up with content that can aid the playbook user to create predictions that likely align with ground truth.

**Instructions:**
- Review the existing playbook and the reflection from the previous attempt
- Identify what changes are needed: new insights to add, existing bullets to sharpen with new evidence, and outdated or wrong bullets to remove
- Avoid redundancy — prefer editing an existing bullet over adding a near-duplicate
- Do NOT regenerate the entire playbook — only provide the operations needed
- Focus on quality over quantity — a focused, well-organized playbook is better than an exhaustive one
- Format your response as a PURE JSON object
- For any operation if no changes needed, return an empty list for the operations field
- Be concise and specific — each bullet should be actionable
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
Current Playbook has:
  [api-00003] About pagination: many APIs return items in "pages". Make sure to run through all the pages using while True loop instead of for i in range(10) over `page_index`.

Reflections: "The agent paginated correctly with a while True loop but hit HTTP 422 because page_limit=50 exceeds the max of 20. Using page_limit=20 or omitting it works."

Response:
{
  "reasoning": "The existing pagination bullet is correct but incomplete — it should mention the page_limit constraint discovered in this trace.",
  "operations": [
    {
      "type": "EDIT",
      "tag": "api-00003",
      "content": "About pagination: many APIs return items in \\"pages\\". Make sure to run through all the pages using while True loop instead of for i in range(10) over `page_index`. Note: some endpoints enforce page_limit <= 20; passing page_limit=50 returns HTTP 422."
    }
  ]
}

**Example 3:**
Current Playbook has:
  [cms-00012] Use create_song_review to add reviews.

Reflections: "create_song_review does not exist — calling it raised 'No API named create_song_review found in the spotify app'. The ground truth code used review_song to create and update reviews."

Response:
{
  "reasoning": "The existing bullet recommends a non-existent API that will waste agent turns. Delete it and add the correct endpoint.",
  "operations": [
    {
      "type": "DELETE",
      "tag": "cms-00012"
    },
    {
      "type": "ADD",
      "section": "apis_to_use_for_specific_information",
      "content": "Use review_song (not create_song_review) to add or update Spotify song reviews. create_song_review does not exist and will raise an error."
    }
  ]
}

**When NOT to operate:**
- If the task was already successful and the reflection contains no new insights, return an empty operations list.
- If the reflection only restates what the playbook already covers, do not add duplicates — return an empty operations list.
- Only propose changes when the reflection reveals genuinely new information, corrections, or outdated advice.

**Your Task:**
Output ONLY a valid JSON object with these exact fields:
- reasoning: your chain of thought / reasoning / thinking process
- operations: a list of operations to be performed on the playbook (empty list if no changes needed)

**About tags:**
Each bullet in the playbook is prefixed with a tag in square brackets, e.g., `[api-00003]` or `[cms-00012]`. For EDIT and DELETE operations, you must use the exact tag from the current playbook. Do NOT invent tags — only reference tags that appear in the current playbook above.

**Available Operations:**
1. ADD — add a new bullet to a section
    - type: "ADD"
    - section: the section to add the bullet to
    - content: the new bullet content (do NOT include the tag — it will be assigned automatically)
2. EDIT — replace the content of an existing bullet
    - type: "EDIT"
    - tag: the exact tag from the current playbook (e.g., "api-00003")
    - content: the new content for that bullet (do NOT include the tag — it is preserved automatically)
3. DELETE — remove an existing bullet
    - type: "DELETE"
    - tag: the exact tag from the current playbook (e.g., "cms-00012")

**Available Sections (for ADD):**
- strategies_and_hard_rules
- apis_to_use_for_specific_information
- useful_code_snippets_and_templates
- common_mistakes_and_correct_strategies
- problem_solving_heuristics_and_workflows
- verification_checklist
- troubleshooting_and_pitfalls
- others

**RESPONSE FORMAT — Output ONLY this JSON structure (no markdown, no code blocks):**
{
  "reasoning": "[Your reasoning here]",
  "operations": [
    {"type": "ADD", "section": "verification_checklist", "content": "[New item...]"},
    {"type": "EDIT", "tag": "api-00003", "content": "[Revised item...]"},
    {"type": "DELETE", "tag": "cms-00012"}
  ]
}"""


# ---------------------------------------------------------------------------
# Apply ADD / EDIT / DELETE operations on a sectioned playbook
# ---------------------------------------------------------------------------
def apply_curator_aed_operations(
    playbook_text: str, operations: list[dict], next_id: int
) -> tuple[str, int]:
    lines = playbook_text.strip().split("\n")

    # Build index of existing bullets by tag
    bullet_indices: dict[str, int] = {}
    sections: dict[str, list[int]] = {}
    current_section = "general"
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("##"):
            section_header = stripped[2:].strip()
            current_section = (
                section_header.lower().replace(" ", "_").replace("&", "and").rstrip(":")
            )
            sections.setdefault(current_section, [])
        else:
            parsed = parse_playbook_line(stripped)
            if parsed:
                bullet_indices[parsed["id"]] = index
                sections.setdefault(current_section, []).append(index)

    lines_to_delete: set[int] = set()
    lines_to_edit: dict[int, str] = {}
    bullets_to_add: list[tuple[str, str]] = []

    for op in operations:
        op_type = op.get("type", "").upper()

        if op_type == "DELETE":
            tag = op.get("tag", "")
            if tag in bullet_indices:
                lines_to_delete.add(bullet_indices[tag])
                print(f"  Deleted bullet {tag}")
            else:
                print(f"  Warning: DELETE tag '{tag}' not found, skipping")

        elif op_type == "EDIT":
            tag = op.get("tag", "")
            content = op.get("content", "")
            if tag in bullet_indices:
                lines_to_edit[bullet_indices[tag]] = format_playbook_line(tag, content)
                print(f"  Edited bullet {tag}")
            else:
                print(f"  Warning: EDIT tag '{tag}' not found, skipping")

        elif op_type == "ADD":
            section_raw = op.get("section", "general")
            section = section_raw.lower().replace(" ", "_").replace("&", "and").rstrip(":")
            if section not in sections and section != "general":
                print(f"  Warning: Section '{section_raw}' not found, adding to OTHERS")
                section = "others"
            slug = get_section_slug(section)
            new_id = f"{slug}-{next_id:05d}"
            next_id += 1
            bullets_to_add.append((section, format_playbook_line(new_id, op.get("content", ""))))
            print(f"  Added bullet {new_id} to section {section}")

    # Rebuild playbook
    final_lines: list[str] = []
    current_section = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("##"):
            if current_section is not None:
                section_adds = [bullet for sec, bullet in bullets_to_add if sec == current_section]
                final_lines.extend(section_adds)
                bullets_to_add = [(sec, bullet) for sec, bullet in bullets_to_add if sec != current_section]
            section_header = stripped[2:].strip()
            current_section = section_header.lower().replace(" ", "_").replace("&", "and").rstrip(":")

        if index in lines_to_delete:
            continue
        if index in lines_to_edit:
            final_lines.append(lines_to_edit[index])
        else:
            final_lines.append(line)

    if current_section is not None:
        section_adds = [bullet for sec, bullet in bullets_to_add if sec == current_section]
        final_lines.extend(section_adds)
        bullets_to_add = [(sec, bullet) for sec, bullet in bullets_to_add if sec != current_section]

    if bullets_to_add:
        print(f"  Warning: {len(bullets_to_add)} bullets have no matching section, adding to OTHERS")
        others_bullets = [bullet for _, bullet in bullets_to_add]
        others_index = -1
        for idx, line in enumerate(final_lines):
            if line.strip() == "## OTHERS":
                others_index = idx
                break
        if others_index >= 0:
            for j, bullet in enumerate(others_bullets):
                final_lines.insert(others_index + 1 + j, bullet)
        else:
            final_lines.extend(others_bullets)

    return "\n".join(final_lines), next_id


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class ACEAEDModel(ACEModel):
    """ACE with ADD/EDIT/DELETE curator. Starts from the empty playbook, no GT."""

    def __init__(
        self,
        use_ground_truth: bool = True,
        use_test_report: bool = True,
        initial_playbook: str = "default",
    ):
        super().__init__(
            use_ground_truth=use_ground_truth, 
            use_test_report=use_test_report,
            initial_playbook=initial_playbook,
        )
        self.name = "ace_aed"
        self._curator_prompt = CURATOR_AED_PROMPT

    def _validate_operations(self, operations) -> list[dict]:
        if not isinstance(operations, list):
            print(f"[{self.name}] Warning: 'operations' field is not a list, skipping.")
            return []
        filtered_ops: list[dict] = []
        for i, op in enumerate(operations):
            if not isinstance(op, dict):
                print(f"  Skipping operation {i}: not a dictionary")
                continue
            op_type = op.get("type", "").upper()
            if op_type == "ADD":
                missing = {"type", "section", "content"} - set(op.keys())
                if missing:
                    print(f"  Skipping ADD operation {i}: missing fields {list(missing)}")
                    continue
                section_name = (
                    str(op.get("section", "")).strip().lower()
                    .replace(" ", "_").replace("&", "and").rstrip(":")
                )
                if section_name not in ALLOWED_SECTIONS:
                    print(f"  Skipping ADD operation {i}: disallowed section '{op.get('section')}' (normalized: '{section_name}')")
                    continue
            elif op_type == "EDIT":
                if "tag" not in op or "content" not in op:
                    print(f"  Skipping EDIT operation {i}: missing tag or content")
                    continue
            elif op_type == "DELETE":
                if "tag" not in op:
                    print(f"  Skipping DELETE operation {i}: missing tag")
                    continue
            else:
                print(f"  Skipping operation {i}: unknown type '{op.get('type')}'")
                continue
            filtered_ops.append(op)
        return filtered_ops

    def _apply_curator_ops(self, current_context: str, operations: list[dict]) -> str:
        new_context, _ = apply_curator_aed_operations(
            current_context,
            operations,
            get_next_global_id(current_context),
        )
        return new_context
