"""
All prompt templates for the ERL pipeline.

No prompt strings should appear anywhere else in the codebase.

Action format (matches the official microsoft/experiential_rl repo):
  - The LM is called once per step.
  - Output a <reason>...</reason> block, then the next action wrapped in
    triple backticks on the last line, e.g. ```Up```.
  - Valid actions are capitalized: Up, Down, Left, Right.

notebook_minimal builders (ACE simplified, single-LM-call per update):
  - build_notebook_agent_prompt: per-step prompt with notebook as context
  - build_notebook_updater_prompt: post-episode one-shot updater
    emitting JSON {reasoning, operations[]} over a line-numbered notebook.
"""


def build_attempt1_prompt(observation: str, memory: list | None = None) -> str:
    """
    Per-step prompt for attempt 1.

    Shows the current grid state and asks for the single next action using
    the reasoning structure from the official repo.
    """
    memory_block = (
        "Past reflections and strategies from previous episodes:"
        + chr(10)
        + chr(10).join(f"  {i+1}. {m}" for i, m in enumerate(memory))
        if memory
        else "No past reflections or strategies recorded yet."
    )

    return f"""{observation}

{memory_block}

You are an agent playing a game on a grid, acting as a reasoning engine.
Your decisions are based on your current game rules (your best guess of how the game works)
and your strategic playbook (your learned strategies). These may be incomplete or incorrect.
Your only way to interact with the environment is by choosing your NEXT ACTION.

Instructions:
1. Analyze State: Summarize the current state.
2. Predict Long-term Value of Outcomes (Value Function Evaluation): Evaluate the strategic value
   and potential of the current state for the future.
3. Predict Immediate Consequences (World Model Simulation): For the top two candidate actions,
   predict their consequences using a "result-because" structure.
4. Select the Best Action: Choose the action leading to the most advantageous future state.

Your response MUST strictly follow this structure:
<reason>
**1. Analysis of the Current State:**
[Summary of the board state.]

**2. Prediction of the Value of Current States:**
[Assessment of the state's strategic value.]
- **Value:** High / Medium / Low value with justification.

**3. Prediction of Immediate Consequences:**
[Analyze ONLY the top 2 candidate actions using the "result-because" structure.]
- **Action A:** result-because structure.
- **Action B:** result-because structure.
</reason>

Then output the NEXT ACTION inside triple backticks, like this:
```Up```

Always remember:
- Valid actions: Up, Down, Left, Right.
- Think step by step, but make the final line only the next action wrapped in triple backticks.
"""


# ----------------------------------------------------------------------
# notebook_minimal prompts (simplified ACE: one LM call per update)
# ----------------------------------------------------------------------

def build_notebook_agent_prompt(observation: str, notebook: str) -> str:
    """
    Per-step prompt for the notebook_minimal agent.

    The notebook is the accumulated context; it grows across episodes.
    The agent emits a short reason block and exactly one action.
    """
    return f"""{observation}

Your notebook below contains knowledge accumulated from past episodes.
Read it carefully and use it when choosing your next action.

<<<NOTEBOOK>>>
{notebook}
<<<END_NOTEBOOK>>>

You are an agent playing a grid puzzle. Look at the grid and pick the
single best next action.

Your response MUST strictly follow this structure:
<reason>
Short analysis of the current grid, what you think the symbols mean,
and which action is safest.
</reason>

Then output the NEXT ACTION inside triple backticks, like this:
```Up```

Always remember:
- Valid actions: Up, Down, Left, Right.
- The final line must be ONLY the next action wrapped in triple backticks.
"""


def build_notebook_updater_prompt(
    numbered_notebook: str,
    initial_obs: str,
    actions: list,
    feedback: str,
    reward: int,
    reward_threshold: float,
) -> str:
    """
    Post-episode updater prompt for notebook_minimal.

    Reflector + curator merged into one LM call.  Input: line-numbered
    notebook + episode trace.  Output: JSON {reasoning, operations[]}
    with replace / insert_after / delete ops over original line numbers.
    """
    outcome = "SUCCESS" if reward >= reward_threshold else "FAILURE"
    return f"""You are a notebook updater for an agent playing a grid puzzle.
Review the episode below and edit the notebook with insights that will
improve the agent's future success rate.

The notebook is provided to the agent at the start of every future
episode.  Your edits are evaluated on whether they raise the agent's
pass rate across many episodes.

**Episode outcome: {outcome}** (reward = {reward})

Initial observation:
<<<INITIAL>>>
{initial_obs}
<<<END_INITIAL>>>

Actions taken: {actions}

Environment feedback:
<<<FEEDBACK>>>
{feedback}
<<<END_FEEDBACK>>>

Current notebook (line-numbered):
<<<NOTEBOOK>>>
{numbered_notebook}
<<<END_NOTEBOOK>>>

**Editing guidelines:**
- Replace when an existing note is wrong or superseded.
- Insert when adding a genuinely new insight (e.g., what a symbol
  seems to mean, a pattern that succeeded or failed).
- Delete when a note is redundant or misleading.
- Keep entries short, concrete, and reusable across episodes.
- An empty operations list is valid when nothing new was learned.

**Output format (return ONLY valid JSON in a single ```json block):**
```json
{{
  "reasoning": "What happened, why, and what is worth remembering.",
  "operations": [
    {{"type": "insert_after", "line": N, "content": "single-line text"}},
    {{"type": "replace", "start_line": N, "end_line": M, "content": "text"}},
    {{"type": "delete", "start_line": N, "end_line": M}}
  ]
}}
```
- insert_after: inserts after line N (use line 0 for top of file).
- replace: replaces lines N through M inclusive.
- delete: removes lines N through M inclusive.
- content for insert_after must be a single line (no newlines).
- Line numbers refer to the numbered notebook shown above.
"""


def build_reflection_prompt(
    observation: str,
    attempt1_actions: list,
    feedback1: str,
    reward1: int,
    memory: list,
) -> str:
    """
    Reflection prompt shown after attempt 1 fails.

    Includes the initial grid, the full attempt-1 trajectory, environment
    feedback, reward, and any past reflections from memory.
    Asks the LM to reason about failures and produce a concrete strategy.
    """
    memory_block = (
        "Past reflections and strategies (use these as reference):"
        + chr(10)
        + chr(10).join(f"  {i+1}. {m}" for i, m in enumerate(memory))
        if memory
        else ""
    )

    return f"""You attempted a puzzle. Here is the situation:

Original grid observation:
{observation}

Actions taken:  {attempt1_actions}
Feedback:       {feedback1}
Reward:         {reward1}

{memory_block}

Reflect on what went wrong and describe a concrete improved strategy in free text.
Explain step by step how you would navigate the grid differently to achieve a higher reward.
Reference the grid symbols (A=player, B=goal or box, C=hole or goal tile, D=floor, E=wall)
and specific positions where your previous attempt failed.
"""
