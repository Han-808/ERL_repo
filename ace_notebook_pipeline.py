"""
ACE Notebook pipeline for FrozenLake and Sokoban.

Online setting: K different randomly-generated instances are played in
sequence.  The Notebook accumulates experience across all K instances —
each episode the updater proposes line-numbered edits that the agent
uses in the next episode.

One LM call per step (agent) + one LM call per episode (updater).
"""

from environments.frozen_lake import FrozenLake
from environments.sokoban import Sokoban
from ace.notebook import Notebook
from ace.notebook_updater import call_notebook_updater
from common import build_client, parse_action_single


_VALID_ACTIONS = {"Up", "Down", "Left", "Right"}


# ----------------------------------------------------------------------
# Per-step agent prompt with notebook injected
# ----------------------------------------------------------------------

def build_attempt_with_notebook_prompt(observation: str, notebook: Notebook) -> str:
    """
    Return a prompt that shows the notebook context above the current grid,
    then asks for the single best next action using the paper's reasoning
    structure.
    """
    return f"""{notebook.to_string()}

---

{observation}

You are an agent playing a game on a grid, acting as a reasoning engine.
Your decisions are based on your current game rules (your best guess of how
the game works) and your strategic notebook above (your learned strategies).
These may be incomplete or incorrect.
Your only way to interact with the environment is by choosing your NEXT ACTION.

Instructions:
1. Analyze State: Summarize the current state.
2. Predict Long-term Value of Outcomes (Value Function Evaluation): Evaluate
   the strategic value and potential of the current state for the future.
3. Predict Immediate Consequences (World Model Simulation): For the top two
   candidate actions, predict their consequences using a "result-because" structure.
4. Select the Best Action: Choose the action leading to the most advantageous
   future state.

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
- Think step by step, but make the final line only the next action in triple backticks.
"""


# ----------------------------------------------------------------------
# ACENotebookPipeline
# ----------------------------------------------------------------------

class ACENotebookPipeline:
    """
    Full online-learning loop: notebook grows across K random instances.

    For each episode:
      1. A fresh randomly-generated environment is created (seed=episode).
      2. The agent plays step-by-step with the notebook as context.
      3. The updater edits the notebook based on the trajectory.
    """

    def __init__(
        self,
        env,
        model: str = "qwen3-8b",
        server_url: str = "http://LOCAL_SERVER/v1",
        reward_threshold: float = 1.0,
        disable_thinking: bool = False,
    ):
        self.env = env
        self.model = model
        self.reward_threshold = reward_threshold
        self.disable_thinking = disable_thinking
        self.notebook = Notebook()
        self.client = build_client(server_url)

        print(f"Connected to LM server at {server_url}")
        print(f"Model: {self.model}")
        print("ACE Notebook Pipeline ready.")

    def _call_lm(self, prompt: str) -> str:
        try:
            request_kwargs = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1024,
                "temperature": 0.7,
            }
            if self.disable_thinking:
                request_kwargs["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            response = self.client.chat.completions.create(**request_kwargs)
            return response.choices[0].message.content
        except Exception as e:
            print(f"[LM error] {e}")
            return ""

    # -- Step loop -------------------------------------------------------

    def _run_attempt(self, build_step_prompt) -> tuple:
        all_actions, all_feedbacks, reward = [], [], 0
        while not self.env.done:
            obs = self.env.get_observation()
            lm_out = self._call_lm(build_step_prompt(obs))
            action = parse_action_single(lm_out)
            all_actions.append(action)
            _, step_feedback, reward, done = self.env.step([action])
            all_feedbacks.append(step_feedback)
            if done:
                break
        return all_actions, " ".join(all_feedbacks), reward

    # -- Episode ---------------------------------------------------------

    def run_episode(self, episode_num: int) -> dict:
        # env was already set to the correct seed by run(); just reset agent.
        initial_obs = self.env.reset()

        actions, feedback, reward = self._run_attempt(
            lambda obs: build_attempt_with_notebook_prompt(obs, self.notebook)
        )

        print(f"\n{'='*40}")
        print(f"=== Episode {episode_num} ===")
        print(f"{'='*40}")
        print(f"[Attempt] Actions:  {actions}")
        print(f"[Attempt] Feedback: {feedback}")
        print(f"[Attempt] Reward:   {reward}")
        print(f"[Notebook size] {len(self.notebook.to_string().splitlines())} lines")

        # Update notebook after every episode (success and failure)
        print("[Notebook Updater] Calling LM...")
        operations = call_notebook_updater(
            self.client, self.model,
            self.notebook, actions, feedback, reward,
        )
        applied = self.notebook.apply_updates(operations)
        print(f"[Notebook Updater] Applied {len(applied)} operations")

        return {
            "episode":        episode_num,
            "actions":        actions,
            "feedback":       feedback,
            "reward":         reward,
            "reward1":        reward,
            "operations":     [str(op) for op in applied],
            "notebook_lines": len(self.notebook.to_string().splitlines()),
        }

    # -- Full experiment -------------------------------------------------

    def run(self, n_episodes: int = 50) -> dict:
        """
        Online setting: K different random instances.

        Notebook accumulates experience across all K instances.
        Each instance is a new randomly generated map/puzzle (seed=ep).
        """
        all_logs = []
        env_class = type(self.env)

        for ep in range(1, n_episodes + 1):
            self.env = env_class(seed=ep)
            log = self.run_episode(ep)
            all_logs.append(log)

        n_success = sum(
            1 for lg in all_logs if lg["reward"] >= self.reward_threshold
        )
        pass_rate = n_success / n_episodes
        env_name = env_class.__name__

        print(f"\n{'='*40}")
        print(f"SUMMARY ({env_name}, {n_episodes} instances)")
        print(f"{'='*40}")
        print(f"Pass rate: {n_success}/{n_episodes} ({pass_rate*100:.1f}%)")

        return {
            "logs":      all_logs,
            "pass_rate": pass_rate,
        }
