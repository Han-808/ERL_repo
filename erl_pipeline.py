"""
ERL (Experiential Reinforcement Learning) inference pipeline.

Implements the two-attempt + reflection + memory loop described in
arXiv 2602.13949.  No training or gradient updates — inference only.

Interaction model: the LM is called once per action step, sees the
current grid after every move, and outputs a single action inside
triple backticks (e.g. '''down'''), matching the paper's Table 2 format.
"""

import re

from openai import OpenAI

from prompts import (
    build_attempt1_prompt,
    build_reflection_prompt,
    build_attempt2_prompt,
)

_VALID_ACTIONS = {"up", "down", "left", "right"}
_SKIPPED = "Skipped (attempt 1 succeeded)"


class ERLPipeline:

    def __init__(
        self,
        env,
        model: str = "qwen3-8b",
        server_url: str = "http://LOCAL_SERVER/v1",
        reward_threshold: float = 1.0,   # τ = 1 as specified in Appendix A
        memory_size: int = 5,
    ):
        self.env = env
        self.model = model
        self.reward_threshold = reward_threshold
        self.memory = []  # cross-episode memory
        self.memory_size = memory_size
        self.episode_logs = []

        self.client = OpenAI(
            base_url=server_url,
            api_key="EMPTY",
        )

        print(f"Connected to LM server at {server_url}")
        print(f"Model: {self.model}")
        print("Environments ready.")

    # ------------------------------------------------------------------
    # LM interaction
    # ------------------------------------------------------------------

    def _call_lm(self, prompt: str) -> str:
        """Send a prompt to the LM and return the response text."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[LM error] {e}")
            return ""

    def _parse_action_single(self, lm_output: str) -> str:
        """
        Extract one action from the LM's output.

        Primary format (paper Table 2): triple backticks, e.g. '''down'''
        Fallback 1: any single/double-backtick token, e.g. `down`
        Fallback 2: first valid action word found in the last non-empty line.
        Fallback 3: "down" if nothing matches.
        """
        # Primary: triple backticks
        m = re.search(r"'''(\w+)'''", lm_output)
        if m:
            action = m.group(1).lower()
            if action in _VALID_ACTIONS:
                return action

        # Fallback 1: backtick-quoted token
        m = re.search(r"`(\w+)`", lm_output)
        if m:
            action = m.group(1).lower()
            if action in _VALID_ACTIONS:
                return action

        # Fallback 2: scan lines bottom-up for a valid action word
        for line in reversed(lm_output.strip().split("\n")):
            line_lower = line.lower()
            for action in ("up", "down", "left", "right"):
                if action in line_lower:
                    return action

        print("[Warning] Could not parse action; using fallback 'down'.")
        return "down"

    # ------------------------------------------------------------------
    # Step-by-step attempt runner
    # ------------------------------------------------------------------

    def _run_attempt(self, build_step_prompt) -> tuple:
        """
        Execute one full attempt in step-by-step mode.

        The LM is called once per action.  After each action the
        environment updates its state and get_observation() returns the
        new grid, which is passed to the next LM call.

        Parameters
        ----------
        build_step_prompt : callable
            A function (obs: str) -> str that builds the per-step prompt
            from the current grid observation.

        Returns
        -------
        actions  : list[str]   — every action taken in order
        feedback : str         — all step feedbacks joined
        reward   : int         — final episode reward (0 or 1)
        """
        all_actions = []
        all_feedbacks = []
        reward = 0

        while not self.env.done:
            obs = self.env.get_observation()
            lm_out = self._call_lm(build_step_prompt(obs))
            action = self._parse_action_single(lm_out)
            all_actions.append(action)
            _, step_feedback, reward, done = self.env.step([action])
            all_feedbacks.append(step_feedback)
            if done:
                break

        return all_actions, " ".join(all_feedbacks), reward

    # ------------------------------------------------------------------
    # Episode loop
    # ------------------------------------------------------------------

    def run_episode(self, episode_num: int) -> dict:
        """
        Run one full ERL episode:
          1. Attempt 1 — step-by-step, LM called once per action
          2. Gated reflection (Appendix A): skip if reward1 >= τ
          3. Attempt 2 — step-by-step, reflection used as strategy guide
          4. Memory update if reward2 >= τ and reflection is not _SKIPPED
        """
        # ---- Step 1: reset + attempt 1 --------------------------------
        initial_obs = self.env.reset()
        actions1, feedback1, reward1 = self._run_attempt(
            build_attempt1_prompt
        )

        # ---- Step 3: print attempt 1 results --------------------------
        print(f"\n{'='*40}")
        print(f"=== Episode {episode_num} ===")
        print(f"{'='*40}")
        print(f"[Attempt 1] Actions:  {actions1}")
        print(f"[Attempt 1] Feedback: {feedback1}")
        print(f"[Attempt 1] Reward:   {reward1}")

        # ---- Gated reflection (Appendix A): τ = reward_threshold ------
        if reward1 >= self.reward_threshold:
            # CASE A: attempt 1 succeeded — skip reflection + attempt 2
            print(
                "[Gated] Attempt 1 succeeded. "
                "Skipping reflection and attempt 2."
            )
            reflection = _SKIPPED
            actions2 = actions1
            feedback2 = feedback1
            reward2 = reward1
            gated = True
        else:
            # CASE B: attempt 1 failed — reflect then retry
            gated = False

            # ---- Step 4: reflection -----------------------------------
            prompt_r = build_reflection_prompt(
                initial_obs, actions1, feedback1, reward1, self.memory
            )
            reflection = self._call_lm(prompt_r)
            print(f"\n[Reflection] {reflection}")

            # ---- Step 5-6: reset + attempt 2 --------------------------
            self.env.reset()
            actions2, feedback2, reward2 = self._run_attempt(
                lambda obs: build_attempt2_prompt(obs, reflection)
            )

            # ---- Step 7: print attempt 2 results ----------------------
            print(f"\n[Attempt 2] Actions:  {actions2}")
            print(f"[Attempt 2] Feedback: {feedback2}")
            print(f"[Attempt 2] Reward:   {reward2}")

        # ---- Step 8: memory update ------------------------------------
        if reward2 >= self.reward_threshold and reflection != _SKIPPED:
            self.memory.append(reflection)
            if len(self.memory) > self.memory_size:
                self.memory.pop(0)       # evict oldest entry
            print(f"[Memory updated] Size: {len(self.memory)}")

        # ---- Step 9: return log ---------------------------------------
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

    # ------------------------------------------------------------------
    # Full experiment
    # ------------------------------------------------------------------

    def run(self, n_episodes: int) -> dict:
        """Run n_episodes episodes and return aggregated results."""
        all_logs = []
        for ep in range(1, n_episodes + 1):
            log = self.run_episode(ep)
            all_logs.append(log)

        n1 = sum(
            1 for lg in all_logs if lg["reward1"] >= self.reward_threshold
        )
        n2 = sum(
            1 for lg in all_logs if lg["reward2"] >= self.reward_threshold
        )
        rate1 = n1 / n_episodes
        rate2 = n2 / n_episodes
        improvement = rate2 - rate1

        env_name = type(self.env).__name__

        print(f"\n{'='*40}")
        print(f"SUMMARY ({env_name}, {n_episodes} episodes)")
        print(f"{'='*40}")
        print(
            f"Attempt 1 success rate: {n1}/{n_episodes} ({rate1*100:.1f}%)"
        )
        print(
            f"Attempt 2 success rate: {n2}/{n_episodes} ({rate2*100:.1f}%)"
        )
        print(f"Improvement:            {improvement*100:+.1f}%")

        return {
            "logs": all_logs,
            "attempt1_rate": rate1,
            "attempt2_rate": rate2,
            "improvement": improvement,
        }
