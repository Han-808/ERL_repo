"""
All prompt templates for the ERL pipeline.

No prompt strings should appear anywhere else in the codebase.

Action format follows the paper (Table 2, Appendix B):
  - LM is called once per step.
  - Output one action inside triple backticks on the last line.
  - Example: '''down'''
"""


def build_attempt1_prompt(observation: str) -> str:
    """
    Per-step prompt for attempt 1.

    Shows the current grid state and asks for the single next action.
    Contains no rules or hints about symbol meanings.

    """
    return (
        f"{observation}\n\n"
        "Choose the next action to complete the task.\n"
        "Output your action inside triple backticks on the last line.\n"
        "Example: '''down'''"
    )


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
    feedback, reward, and any past successful strategies from memory.
    Asks the LM to reason about failures and produce a concrete strategy.
    """
    parts = []

    parts.append("You attempted a puzzle. Here is the situation:\n")
    parts.append(f"Initial grid:\n{observation}\n")
    parts.append(f"Actions taken:  {attempt1_actions}")
    parts.append(f"Feedback:       {feedback1}")
    parts.append(f"Reward:         {reward1}\n")

    if memory:
        parts.append("Past successful strategies (use these as reference):")
        for i, entry in enumerate(memory, start=1):
            parts.append(f"  {i}. {entry}")
        parts.append("")

    parts.append(
        "Reflect on what went wrong and describe a concrete improved strategy "
        "in free text. Explain step by step how you would navigate the grid "
        "differently to achieve a higher reward."
    )

    return "\n".join(parts)


def build_attempt2_prompt(observation: str, reflection: str) -> str:
    """
    Per-step prompt for attempt 2.

    Shows the current (updated) grid state and the reflection as a strategy
    guide, then asks for the single next action in the same triple-backtick
    format as attempt 1.
    """
    return (
        f"Strategy from reflection:\n{reflection}\n\n"
        f"Current grid:\n{observation}\n\n"
        "Based on your strategy, choose the next action.\n"
        "Output your action inside triple backticks on the last line.\n"
        "Example: '''down'''"
    )
