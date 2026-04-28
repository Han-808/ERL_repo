"""
ERL (Experiential Reinforcement Learning) method — BaseMethod-compatible.

Implements the two-attempt + reflection + memory loop described in
arXiv 2602.13949.  No training or gradient updates — inference only.

Interaction model: the LM is called once per action step, sees the
current grid after every move, and outputs a single action inside
triple backticks (e.g. ```Down```), matching the paper's Table 2 format.
"""

from common import (
    BaseMethod,
    build_client,
    call_lm,
    parse_action_single,
    summarize_logs,
)
from prompts import (
    build_attempt1_prompt,
    build_attempt2_prompt,
    build_reflection_prompt,
)


_SKIPPED = "Skipped (attempt 1 succeeded)"


class ERLMethod(BaseMethod):
    """ERL two-attempt + reflection + memory loop as a BaseMethod."""

    name = "erl"

    def __init__(
        self,
        env,
        model: str = "qwen3-8b",
        server_url: str = "http://LOCAL_SERVER/v1",
        reward_threshold: float = 1.0,   # τ = 1 as specified in Appendix A
        memory_size: int = 5,
        disable_thinking: bool = False,
    ):
        self.env = env
        self.model = model
        self.reward_threshold = reward_threshold
        self.memory_size = memory_size
        self.disable_thinking = disable_thinking
        self.memory = self.initialize_context()
        self.episode_logs = []

        self.client = build_client(server_url)

        print(f"Connected to LM server at {server_url}")
        print(f"Model: {self.model}")
        print("ERL Method ready.")

    # -- BaseMethod contract ------------------------------------------------

    def initialize_context(self) -> list:
        return []

    # -- Per-attempt step loop ----------------------------------------------

    def _run_attempt(self, build_step_prompt) -> tuple:
        """
        Execute one full attempt in step-by-step mode.

        build_step_prompt : callable (obs: str) -> str
        """
        all_actions = []
        all_feedbacks = []
        reward = 0

        while not self.env.done:
            obs = self.env.get_observation()
            lm_out = call_lm(
                self.client, self.model, build_step_prompt(obs),
                disable_thinking=self.disable_thinking,
            )
            action = parse_action_single(lm_out)
            all_actions.append(action)
            _, step_feedback, reward, done = self.env.step([action])
            all_feedbacks.append(step_feedback)
            if done:
                break

        return all_actions, " ".join(all_feedbacks), reward

    # -- Episode loop -------------------------------------------------------

    def run_episode(self, episode_num: int) -> dict:
        initial_obs = self.env.reset(seed=episode_num)
        actions1, feedback1, reward1 = self._run_attempt(build_attempt1_prompt)

        print(f"\n{'='*40}")
        print(f"=== Episode {episode_num} ===")
        print(f"{'='*40}")
        print(f"[Attempt 1] Actions:  {actions1}")
        print(f"[Attempt 1] Feedback: {feedback1}")
        print(f"[Attempt 1] Reward:   {reward1}")

        if reward1 >= self.reward_threshold:
            print("[Gated] Attempt 1 succeeded. Skipping reflection and attempt 2.")
            reflection = _SKIPPED
            actions2, feedback2, reward2 = actions1, feedback1, reward1
            gated = True
        else:
            gated = False

            prompt_r = build_reflection_prompt(
                initial_obs, actions1, feedback1, reward1, self.memory
            )
            reflection = call_lm(
                self.client, self.model, prompt_r,
                disable_thinking=self.disable_thinking,
            )
            print(f"\n[Reflection] {reflection}")

            self.env.reset()
            actions2, feedback2, reward2 = self._run_attempt(
                lambda obs: build_attempt2_prompt(obs, reflection)
            )

            print(f"\n[Attempt 2] Actions:  {actions2}")
            print(f"[Attempt 2] Feedback: {feedback2}")
            print(f"[Attempt 2] Reward:   {reward2}")

        if reward2 >= self.reward_threshold and reflection != _SKIPPED:
            self.memory.append(reflection)
            if len(self.memory) > self.memory_size:
                self.memory.pop(0)
            print(f"[Memory updated] Size: {len(self.memory)}")

        return {
            "episode": episode_num,
            "actions1": actions1,
            "feedback1": feedback1,
            "reward1": reward1,
            "reflection": reflection,
            "actions2": actions2,
            "feedback2": feedback2,
            "reward2": reward2,
            "memory_size": len(self.memory),
            "gated": gated,
        }

    # -- Full experiment ----------------------------------------------------

    def run(self, n_episodes: int) -> dict:
        all_logs = []
        for ep in range(1, n_episodes + 1):
            all_logs.append(self.run_episode(ep))
        return summarize_logs(
            all_logs,
            self.reward_threshold,
            type(self.env).__name__,
            n_episodes,
        )
