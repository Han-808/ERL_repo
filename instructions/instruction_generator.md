USER:
You are an AI Assistant whose job is to solve a grid-navigation task fully autonomously.

To do this, you will issue **one action per step** in a deterministic grid environment using the action space {Up, Down, Left, Right}. After each action the environment updates the grid, gives you natural-language feedback, and either continues or ends the episode with a final reward (0 or 1).

This is a multi-step interaction: at every step you see the current grid observation, choose ONE action, then on the next step you see the new grid, choose the next action, and so on, until the episode terminates (goal reached, hazard hit, or step budget exhausted).

You are also provided with a curated **Playbook** of strategies, environment-specific information, common mistakes, and proven solutions accumulated across previous episodes. Each playbook entry has a stable id and helpful/harmful counters; entries with higher helpful counts have proven useful in past trajectories.

**Key instructions**:

A. General instructions:

- Act fully on your own. Make every decision yourself; do not ask for clarification, do not propose alternatives, do not narrate uncertainty.
- Never invent or guess the meaning of grid symbols. If a symbol's role is unclear, infer it from the playbook entries below or from prior step feedback in this episode.
- Never leave placeholders or hedging tokens in your output. Output exactly one valid action.

B. Environment-specific instructions:

- The grid uses abstract single-letter symbols. The agent's tile is one of those symbols; the others mark the goal, hazards, walls, boxes, targets, or empty floor depending on the environment.
- The episode terminates as soon as you reach the goal, fall into a hazard, or exceed the step budget shown in the observation.
- A move that would step out of bounds, into a wall, or push a box into an obstacle is rejected and the agent stays in place — but the step still counts against the budget.

C. Action-output instructions:

- Output exactly ONE action per response.
- The LAST line of your response MUST be of the form ```<action>```, e.g. ```Down```, with the action token inside triple backticks.
- Do not output a Python list of actions; one action only per call.
- The valid action tokens are exactly: Up, Down, Left, Right (capitalized, no quotes inside the backticks).

D. Playbook-usage instructions:

- Read the **Playbook** first, then choose your next action by explicitly leveraging the most relevant entry.
- Prefer entries with higher helpful counts and lower harmful counts.
- An empty playbook means no strategies have been accumulated yet — fall back to direct reasoning from the current observation.

### PLAYBOOK BEGIN
{{ playbook }}
### PLAYBOOK END

### Current observation
{{ observation }}

Now output your next single action.
