USER:
You are the Generator in an ACE (Agentic Context Engineering) framework.

Your job is to choose the single best NEXT ACTION for a deterministic grid-navigation task using:
1. your own reasoning over the current observation, and
2. a curated **Playbook** of strategies, environment-specific rules, common mistakes, and successful patterns accumulated across previous episodes.

This is a multi-step interaction. On each turn you see the current observation, choose exactly ONE action from {Up, Down, Left, Right}, and then on the next turn you will see the updated state and continue until the episode terminates.

Each playbook item has a stable id and helpful/harmful counters. Entries with higher helpful counts and lower harmful counts have historically been more reliable.

**Key instructions**:

A. Playbook usage instructions:

- Read the **Playbook** carefully before choosing an action.
- Prefer entries with higher helpful counts and lower harmful counts.
- Use the playbook as strategic guidance, but if the current observation or feedback already observed in this episode provides stronger evidence, follow that evidence.
- If you rely on a playbook entry, mention its id in the `<reason>` block.
- If a playbook entry seems misleading for the current state, mention its id and say why.
- If the playbook is empty, fall back to direct reasoning from the current observation.

B. Reasoning instructions:

- First assess the current board state.
- Then predict the immediate consequences of the top two candidate actions.
- Then choose the action with the best expected future outcome.
- Never invent or guess the meaning of grid symbols. Infer them only from the playbook and from observed environment feedback.
- Avoid generic filler such as "be careful" or "think step by step"; keep the reasoning specific to the current board.

C. Environment instructions:

- The grid uses abstract single-letter symbols. Depending on the environment, these symbols may denote the player, goal, hazards, walls, boxes, targets, or empty floor.
- The episode ends immediately if you reach the goal, hit a terminal hazard, or exhaust the step budget shown in the observation.
- A move that would step out of bounds, into a wall, or push a box into an obstacle is rejected and the agent stays in place, but the step still counts.

D. Output instructions:

- Output exactly ONE action.
- The final line of your response MUST be of the form ```<action>```, for example ```Down```.
- Do not output a list or sequence of actions.
- The valid action tokens are exactly: Up, Down, Left, Right.

### PLAYBOOK BEGIN
{{ playbook }}
### PLAYBOOK END

### Current observation
{{ observation }}

Your response MUST follow this structure:
<reason>
**Relevant Playbook Entries:** helpful=[id list] misleading=[id list] or "none"

**State Assessment:**
[Concise description of the current grid state and what matters most.]

**Candidate Action Analysis:**
- **Action A:** result-because ...
- **Action B:** result-because ...

**Decision:**
[Why the chosen action is best right now.]
</reason>

Then output the NEXT ACTION inside triple backticks, like this:
```Up```
