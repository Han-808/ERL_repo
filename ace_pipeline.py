"""
ACE (Agentic Context Engineering) inference pipeline.

"""

import re
from dataclasses import asdict
from pathlib import Path

from openai import OpenAI

from prompts import build_attempt1_prompt
from ace.common import render_template
from ace.methods.ace import (
    Playbook,
    run_reflector,
    run_curator,
    grow_and_refine,
)


_VALID_ACTIONS = {"up", "down", "left", "right"}

# Resolve instruction paths relative to this file so `python main_ace.py`
# works no matter which directory it was launched from.
_ACE_DIR = Path(__file__).resolve().parent / "ace"


# ----------------------------------------------------------------------
# Prompt builder for attempt 2 (playbook-guided)
#
# Loaded from ace/instruction_generator.md so all prompt strings live in
# .md files, not in Python.  Mirrors the {{ playbook }} / {{ observation }}
# placeholder convention used by the paper's Appendix-D Generator prompt.
# ----------------------------------------------------------------------

_GENERATOR_TEMPLATE = (_ACE_DIR / "instruction_generator.md").read_text(
    encoding="utf-8"
)


def build_attempt2_prompt_with_playbook(observation: str, playbook) -> str:
    """Per-step Generator prompt for attempt 2, with the full Playbook injected."""
    return render_template(
        _GENERATOR_TEMPLATE,
        playbook=playbook.to_prompt_string(),
        observation=observation,
    )


class ACEPipeline:

    def __init__(
        self,
        env,
        model: str = "qwen3-8b",
        server_url: str = "http://LOCAL_SERVER/v1",
        reward_threshold: float = 1.0,
        refine_every: int = 5,
    ):
        self.env = env
        self.model = model
        self.reward_threshold = reward_threshold
        self.refine_every = refine_every
        self.playbook = Playbook()
        self.episode_logs = []

        self.client = OpenAI(
            base_url=server_url,
            api_key="EMPTY",
        )

        # Paths to prompt templates (no prompt strings live in .py files)
        self.reflector_instruction = str(_ACE_DIR / "instruction_reflector.md")
        self.curator_instruction = str(_ACE_DIR / "instruction_curator.md")

        print(f"Connected to LM server at {server_url}")
        print(f"Model: {self.model}")
        print("ACE Pipeline ready.")

    # ------------------------------------------------------------------
    # LM interaction (identical to ERLPipeline._call_lm /
    # _parse_action_single so the step loop matches the ERL baseline)
    # ------------------------------------------------------------------

    def _call_lm(self, prompt: str) -> str:
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
        m = re.search(r"'''(\w+)'''", lm_output)
        if m:
            action = m.group(1).lower()
            if action in _VALID_ACTIONS:
                return action

        m = re.search(r"`(\w+)`", lm_output)
        if m:
            action = m.group(1).lower()
            if action in _VALID_ACTIONS:
                return action

        for line in reversed(lm_output.strip().split("\n")):
            line_lower = line.lower()
            for action in ("up", "down", "left", "right"):
                if action in line_lower:
                    return action

        print("[Warning] Could not parse action; using fallback 'down'.")
        return "down"

    # ------------------------------------------------------------------
    # Step-by-step attempt runner (verbatim from ERLPipeline)
    # ------------------------------------------------------------------

    def _run_attempt(self, build_step_prompt) -> tuple:
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
        initial_obs = self.env.reset()

        # ---- Attempt 1 (Generator, no playbook) --------------------------
        actions1, feedback1, reward1 = self._run_attempt(build_attempt1_prompt)

        print(f"\n{'='*40}")
        print(f"=== Episode {episode_num} ===")
        print(f"{'='*40}")
        print(f"[Attempt 1] Actions:  {actions1}")
        print(f"[Attempt 1] Feedback: {feedback1}")
        print(f"[Attempt 1] Reward:   {reward1}")

        # ---- Gated: success short-circuits reflection & attempt 2 --------
        if reward1 >= self.reward_threshold:
            print("[Gated] Attempt 1 succeeded. Skipping reflection and attempt 2.")
            delta_items = []
            approved_deltas = []
            actions2, feedback2, reward2 = actions1, feedback1, reward1
            gated = True
        else:
            gated = False

            # ---- Reflector -----------------------------------------------
            delta_items = run_reflector(
                self.client,
                self.model,
                initial_obs,
                actions1,
                feedback1,
                reward1,
                self.playbook,
                self.reflector_instruction,
            )
            print(f"[Reflector] Proposed {len(delta_items)} delta items")

            # ---- Curator -------------------------------------------------
            approved_deltas = run_curator(
                self.client,
                self.model,
                delta_items,
                self.playbook,
                self.curator_instruction,
            )
            print(f"[Curator] Approved {len(approved_deltas)} delta items")

            # ---- Apply deltas --------------------------------------------
            self.playbook.apply_delta(approved_deltas)
            print(f"[Playbook] Size now: {len(self.playbook.items)}")

            # ---- Grow-and-refine (lazy, every refine_every episodes) -----
            if self.refine_every > 0 and episode_num % self.refine_every == 0:
                grow_and_refine(self.playbook)

            # ---- Attempt 2 (Generator with playbook) ---------------------
            self.env.reset()
            actions2, feedback2, reward2 = self._run_attempt(
                lambda obs: build_attempt2_prompt_with_playbook(
                    obs, self.playbook
                )
            )

            print(f"\n[Attempt 2] Actions:  {actions2}")
            print(f"[Attempt 2] Feedback: {feedback2}")
            print(f"[Attempt 2] Reward:   {reward2}")

        return {
            "episode": episode_num,
            "actions1": actions1,
            "feedback1": feedback1,
            "reward1": reward1,
            "delta_items": [asdict(d) for d in approved_deltas],
            "playbook": self.playbook.to_dict(),
            "actions2": actions2,
            "feedback2": feedback2,
            "reward2": reward2,
            "playbook_size": len(self.playbook.items),
            "gated": gated,
        }

    # ------------------------------------------------------------------
    # Full experiment
    # ------------------------------------------------------------------

    def run(self, n_episodes: int) -> dict:
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
