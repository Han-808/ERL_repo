"""
ACE-once grid-game ablation.

This method keeps the current grid-game ACE generator, Playbook structure,
ADD-only update behavior, and episode loop, but merges the Reflector and
Curator into one LLM call. The merged updater prompt follows the AppWorld
ace_once structure while replacing code-task inputs with FrozenLake/Sokoban
trajectory inputs.
"""

from dataclasses import asdict

from common import call_lm, render_template
from methods.ace import (
    ACEMethod,
    DeltaItem,
    Playbook,
    _apply_playbook_feedback,
    _format_generator_trace,
    _format_playbook_stats,
    _normalize_reflection,
    _parse_delta_items,
    build_generator_prompt_with_playbook,
    grow_and_refine,
)


MERGED_PROMPT = """\
You are an expert grid-navigation agent, educator, and knowledge curator. \
Your job has two parts, done in a single pass:

1. **Reflect** on the current trajectory: diagnose what went wrong (or could be \
better), grounded in the initial observation, generator reasoning trace, actions, \
environment feedback, reward, and current playbook.

2. **Curate** the playbook: based on your reflection, identify what new insights \
should be added to the existing playbook to help future attempts.

---

## Part 1 - Reflection

**Instructions:**
- Carefully analyze the model's reasoning trace to identify where it went wrong
- Take the environment feedback into account, comparing the selected action \
sequence with the observed outcome and reward
- Identify specific conceptual errors, misread grid relationships, invalid move \
assumptions, terminal-risk mistakes, or misapplied strategies
- Provide actionable insights that could help the model avoid this mistake in \
future grid-game episodes
- Identify root causes: wrong source of evidence, bad inference about movement, \
bad terminal-state reasoning, poor multi-step planning, or misuse of playbook \
entries
- Provide concrete, step-by-step corrections the model should take in this task
- Be specific about what the model should have done differently
- You will receive bulletpoints that are part of the playbook used by the \
generator to choose actions
- You need to analyze these bulletpoints, and give the tag for each bulletpoint, \
tag can be ['helpful', 'harmful', 'neutral'] for the generator's behavior
- Ground every claim in the observation, reasoning trace, feedback, reward, and \
playbook. Do not invent unobserved grid-symbol meanings.
- Do not hard-code symbol meanings unless they were directly supported by the \
current trajectory or accumulated playbook evidence.

## Part 2 - Curation

**Context:**
- The playbook you update will be used to help answering similar grid-navigation \
questions.
- The reflection is generated using environment feedback that will NOT be \
available when the playbook is being used. So you need to come up with content \
that can aid the playbook user to choose actions that likely produce high reward.

**Instructions:**
- Review the existing playbook and your reflection above
- Identify ONLY the NEW insights, strategies, or mistakes that are MISSING from \
the current playbook
- Avoid redundancy - if similar advice already exists, only add new content that \
is a perfect complement to the existing playbook
- Do NOT regenerate the entire playbook - only provide the additions needed
- Focus on quality over quantity - a focused, well-organized playbook is better \
than an exhaustive one
- For any operation if no new content to add, return an empty list for the \
operations field
- Be concise and specific - each addition should be actionable
- For grid games, curate reusable rules about movement, terminal success/failure, \
obstacles, object interaction, multi-step feasibility, and when a playbook entry \
does or does not apply
- Avoid generic filler such as "be careful" or "think step by step"; each \
addition should name a concrete grid-game condition and action implication
- Do not add direct symbol labels such as "C is a hole" unless the trajectory or \
playbook evidence directly supports that generalization.

---

## Inputs

- Initial observation before the generator acted:
<<<OBSERVATION_START>>>
{{ observation }}
<<<OBSERVATION_END>>>

- Generator reasoning trace and per-step actions:
<<<GENERATOR_TRACE_START>>>
{{ generator_trace }}
<<<GENERATOR_TRACE_END>>>

- Model's selected action sequence:
<<<ACTIONS_START>>>
{{ actions }}
<<<ACTIONS_END>>>

- Environment feedback after executing the selected actions:
<<<FEEDBACK_START>>>
{{ feedback }}
<<<FEEDBACK_END>>>

- Outcome reward:
<<<REWARD_START>>>
{{ reward }}
<<<REWARD_END>>>

- Training progress:
<<<TRAINING_PROGRESS_START>>>
Sample {{ current_step }} out of {{ total_samples }}
<<<TRAINING_PROGRESS_END>>>

- Current Playbook stats:
<<<PLAYBOOK_STATS_START>>>
{{ playbook_stats }}
<<<PLAYBOOK_STATS_END>>>

- Current Playbook, used by the generator for action selection:
<<<PLAYBOOK_GUIDE>>>
{{ current_playbook }}
<<<PLAYBOOK_GUIDE>>>

- Task context:
<<<TASK_CONTEXT>>>
Deterministic grid-navigation task. The Generator sees the current observation, \
predicts one action per step, and the environment returns feedback plus reward \
for the trajectory. Valid actions are Up, Down, Left, and Right.
<<<TASK_CONTEXT>>>

---

## Examples

**Example 1:**
Initial Observation: [A grid state with a nearby terminal-risk cell, a step \
budget, and valid actions]
Generator Trace: [The model chooses a move because it assumes an unverified \
symbol is safe]
Actions: ['Right']
Feedback: The move ended the episode with reward 0.
Current Playbook: [Basic movement guidelines]

Response:
{
  "reasoning": "The generator treated an unverified symbol as safe even though \
the playbook did not support that inference. The feedback shows the selected \
move ended the episode with reward 0, so future decisions should avoid assuming \
unknown symbols are safe without evidence.",
  "error_identification": "The agent inferred a symbol meaning that was not \
grounded in playbook or trajectory evidence.",
  "root_cause_analysis": "The agent overgeneralized from grid appearance instead \
of using observed feedback and terminal-risk evidence.",
  "correct_approach": "Compare candidate moves, prefer actions that preserve \
future options, and treat unverified terminal-risk cells conservatively.",
  "key_insight": "Do not infer abstract symbol meanings from appearance alone; \
use feedback and repeated evidence before adding a symbol-specific rule.",
  "bullet_tags": [],
  "operations": [
    {
      "type": "ADD",
      "content": "When an action into an unverified symbol immediately ends an \
episode with reward 0, record the risky condition and avoid repeating that move \
unless later evidence contradicts it."
    }
  ]
}

**Example 2:**
Initial Observation: [A Sokoban-like grid where a movable object can be pushed]
Generator Trace: [The model moves next to an object but pushes it toward a wall]
Actions: ['Left', 'Left']
Feedback: The push was rejected and the step budget was wasted.
Current Playbook: [Basic action-format guidelines]

Response:
{
  "reasoning": "The generator chose a push direction without checking whether \
the destination behind the object was open. The feedback indicates the push was \
rejected, so the reusable rule is to verify push preconditions before committing \
steps.",
  "error_identification": "The agent failed to check the cell behind the pushed \
object.",
  "root_cause_analysis": "It planned the immediate contact with the object but \
not the consequence of the push.",
  "correct_approach": "Before pushing, inspect whether the object can move into \
the next cell and whether the push improves the path to success.",
  "key_insight": "For object-pushing grids, a push is useful only if the object \
has a valid destination and the resulting position preserves future moves.",
  "bullet_tags": [],
  "operations": [
    {
      "type": "ADD",
      "content": "Before pushing a movable object, check that the cell beyond it \
is not blocked and that the resulting object position keeps a feasible route to \
the goal."
    }
  ]
}

---

## Output format

Output ONLY a valid JSON object with these exact fields (no markdown, no code blocks):
{
  "reasoning": "[Chain of thought / reasoning / thinking process, detailed analysis]",
  "error_identification": "[What specifically went wrong in the reasoning?]",
  "root_cause_analysis": "[Why did this error occur? What concept was misunderstood?]",
  "correct_approach": "[What should the model have done instead?]",
  "key_insight": "[What strategy, formula, or principle should be remembered?]",
  "bullet_tags": [
    {"id": 1, "tag": "helpful"},
    {"id": 2, "tag": "harmful"}
  ],
  "operations": [
    {
      "type": "ADD",
      "content": "[New reusable grid-navigation strategy. The system will assign the bullet id.]"
    }
  ]
}"""


def build_merged_prompt(
    observation: str,
    actions: list,
    feedback: str,
    reward: int,
    generator_trace: str,
    playbook: Playbook,
    current_step: int | str = "not provided",
    total_samples: int | str = "not provided",
) -> str:
    """Build the one-call ACE updater prompt for grid-game trajectories."""
    return render_template(
        MERGED_PROMPT,
        observation=observation,
        actions=actions,
        feedback=feedback,
        reward=reward,
        generator_trace=generator_trace,
        current_playbook=playbook.to_prompt_string(),
        playbook_stats=_format_playbook_stats(playbook),
        current_step=current_step,
        total_samples=total_samples,
    )


def run_merged_reflector_curator(
    lm_client,
    model: str,
    observation: str,
    actions: list,
    feedback: str,
    reward: int,
    generator_trace: str,
    playbook: Playbook,
    current_step: int | str = "not provided",
    total_samples: int | str = "not provided",
    disable_thinking: bool = False,
) -> tuple[dict, list[DeltaItem]]:
    """Reflect on the trajectory and curate ADD-only playbook deltas in one call."""
    prompt = build_merged_prompt(
        observation=observation,
        actions=actions,
        feedback=feedback,
        reward=reward,
        generator_trace=generator_trace,
        playbook=playbook,
        current_step=current_step,
        total_samples=total_samples,
    )
    raw = call_lm(
        lm_client, model, prompt, disable_thinking=disable_thinking
    )
    print(f"\n[ACEOnce raw]\n{raw}")
    reflection = _normalize_reflection(raw)
    deltas = _parse_delta_items(raw)
    return reflection, deltas


class ACEOnceMethod(ACEMethod):
    """
    ACE ablation: Generator -> merged Reflector+Curator -> Playbook.

    All grid-game generator behavior and playbook application semantics are
    inherited from ACEMethod; only the updater call is collapsed.
    """

    name = "ace_once"

    def run_episode(self, episode_num: int) -> dict:
        initial_obs = self.env.reset(seed=episode_num)
        actions1, feedback1, reward1, generator_trace1 = self._run_attempt(
            lambda obs: build_generator_prompt_with_playbook(obs, self.playbook)
        )

        print(f"\n{'=' * 40}")
        print(f"=== Episode {episode_num} ===")
        print(f"{'=' * 40}")
        print(f"[Attempt 1] Actions:  {actions1}")
        print(f"[Attempt 1] Feedback: {feedback1}")
        print(f"[Attempt 1] Reward:   {reward1}")

        reflection, approved_deltas = run_merged_reflector_curator(
            self.client, self.model,
            initial_obs, actions1, feedback1, reward1,
            _format_generator_trace(generator_trace1),
            self.playbook,
            current_step=episode_num,
            total_samples=self.total_episodes,
            disable_thinking=self.disable_thinking,
        )
        print("[ACEOnce] Structured reflection and curated deltas captured")

        feedback_stats = _apply_playbook_feedback(
            self.playbook, reflection, generator_trace1,
            reward1, self.reward_threshold,
        )
        if any(feedback_stats[k] for k in ("helpful", "harmful", "neutral")):
            source = "fallback" if feedback_stats["fallback"] else "merged"
            print(
                f"[Playbook] feedback ({source}): "
                f"+{feedback_stats['helpful']} helpful / "
                f"+{feedback_stats['harmful']} harmful / "
                f"{feedback_stats['neutral']} neutral"
            )

        print(f"[ACEOnce] Approved {len(approved_deltas)} delta items")
        self.playbook.apply_delta(approved_deltas)
        print(f"[Playbook] Size now: {len(self.playbook.items)}")

        if self.refine_every > 0 and episode_num % self.refine_every == 0:
            grow_and_refine(self.playbook)

        return {
            "episode": episode_num,
            "actions1": actions1,
            "feedback1": feedback1,
            "reward1": reward1,
            "generator_trace1": generator_trace1,
            "reflection": reflection,
            "playbook_feedback": feedback_stats,
            "delta_items": [asdict(d) for d in approved_deltas],
            "playbook": self.playbook.to_dict(),
            "playbook_size": len(self.playbook.items),
        }
