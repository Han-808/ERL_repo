"""MiniGrid adapter exposing this repo's text grid-game interface."""

from __future__ import annotations

import re
from typing import Any

import gymnasium as gym
import minigrid  # noqa: F401 - imported for Gymnasium environment registration
from minigrid.core.constants import IDX_TO_COLOR, IDX_TO_OBJECT


class MiniGridTextEnv:
    """
    Wrap a MiniGrid/Gymnasium environment in the repo's grid-game API.

    The LM sees MiniGrid's native partial agent-view observation decoded into
    text. Rewards are normalized to binary success/failure for consistency with
    the existing FrozenLake/Sokoban evaluation code.
    """

    ACTIONS = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]
    DEFAULT_ACTION = "forward"
    ACTION_EXAMPLE = "forward"

    _ACTION_TO_ID = {action: i for i, action in enumerate(ACTIONS)}
    _DIR_NAMES = {
        0: "right",
        1: "down",
        2: "left",
        3: "up",
    }

    def __init__(self, env_id: str = "MiniGrid-Empty-5x5-v0", seed: int = 0):
        self.env_id = env_id
        self._master_seed = seed
        self._current_seed = seed
        self._env = gym.make(env_id, render_mode=None)
        self._last_obs: dict[str, Any] | None = None
        self._last_info: dict[str, Any] = {}
        self.done = False
        self.reset(seed=seed)

    def reset(self, seed: int | None = None):
        """
        Reset the environment and return the text observation.

        When seed is None, MiniGrid is reset with the most recent explicit seed
        so callers can replay the current task layout.
        """
        if seed is not None:
            self._current_seed = seed
        obs, info = self._env.reset(seed=self._current_seed)
        self._last_obs = obs
        self._last_info = info
        self.done = False
        return self.get_observation()

    def get_observation(self):
        if self._last_obs is None:
            return self.reset(seed=self._current_seed)

        mission = self._last_obs.get("mission", "")
        base = self._base_env()
        direction = self._DIR_NAMES.get(int(base.agent_dir), str(base.agent_dir))
        carrying = self._format_carrying(getattr(base, "carrying", None))
        grid = self._render_agent_view(self._last_obs["image"])

        return (
            f"Environment: {self.env_id}\n"
            f"Mission: {mission}\n"
            f"Direction: {direction}\n"
            f"Carrying: {carrying}\n\n"
            f"Agent-view grid (native partial observation):\n{grid}\n\n"
            f"Valid actions: {self.ACTIONS}\n"
            f"Step: {base.step_count}/{base.max_steps}"
        )

    def render(self):
        print(self._render_agent_view(self._last_obs["image"]))

    def close(self):
        self._env.close()

    def step(self, actions):
        if self.done:
            return self._render_current_grid(), "Episode already ended.", 0, True

        parts = []
        reward = 0

        for raw_action in actions:
            if self.done:
                break

            action = self._normalize_action(raw_action)
            if action is None:
                parts.append(
                    f"Invalid action '{raw_action}', skipped. "
                    f"Valid actions are {self.ACTIONS}."
                )
                continue

            before = self._state_summary()
            obs, raw_reward, terminated, truncated, info = self._env.step(
                self._ACTION_TO_ID[action]
            )
            self._last_obs = obs
            self._last_info = info
            reward = 1 if float(raw_reward) > 0 else 0
            self.done = bool(terminated or truncated)
            after = self._state_summary()

            feedback = (
                f"Action '{action}' executed. {before} -> {after}. "
                f"Reward: {reward}."
            )
            if float(raw_reward) != reward:
                feedback += f" Raw MiniGrid reward: {float(raw_reward):.6g}."
            if reward:
                feedback += " Task succeeded."
            elif terminated:
                feedback += " Episode terminated without success."
            elif truncated:
                feedback += " Maximum steps reached."
            parts.append(feedback)

        return self._render_current_grid(), " ".join(parts), reward, self.done

    @property
    def ROWS(self):
        return int(getattr(self._base_env(), "height", 0))

    @property
    def COLS(self):
        return int(getattr(self._base_env(), "width", 0))

    def _base_env(self):
        return self._env.unwrapped

    def _normalize_action(self, raw_action):
        candidate = str(raw_action).strip().casefold()
        for action in self.ACTIONS:
            if candidate == action.casefold():
                return action
        return None

    def _state_summary(self):
        base = self._base_env()
        direction = self._DIR_NAMES.get(int(base.agent_dir), str(base.agent_dir))
        return f"position={tuple(base.agent_pos)}, direction={direction}"

    def _format_carrying(self, obj):
        if obj is None:
            return "none"
        color = getattr(obj, "color", None)
        obj_type = getattr(obj, "type", type(obj).__name__)
        return f"{color}_{obj_type}" if color else str(obj_type)

    def _render_current_grid(self):
        if self._last_obs is None:
            return ""
        return self._render_agent_view(self._last_obs["image"])

    def _render_agent_view(self, image):
        width, height = int(image.shape[0]), int(image.shape[1])
        agent_x, agent_y = width // 2, height - 1
        rows = []
        for y in range(height):
            cells = []
            for x in range(width):
                if x == agent_x and y == agent_y:
                    cells.append("agent")
                else:
                    cells.append(self._tile_token(image[x, y]))
            rows.append(" ".join(cells))
        return "\n".join(rows)

    def _tile_token(self, encoded_tile):
        obj_idx, color_idx, state_idx = (int(v) for v in encoded_tile)
        obj = IDX_TO_OBJECT.get(obj_idx, f"object{obj_idx}")
        if obj == "unseen":
            return "unseen"
        if obj == "empty":
            return "."

        color = IDX_TO_COLOR.get(color_idx, f"color{color_idx}")
        token = f"{color}_{obj}"
        if obj == "door":
            state = {0: "open", 1: "closed", 2: "locked"}.get(
                state_idx,
                f"state{state_idx}",
            )
            token = f"{token}_{state}"
        return re.sub(r"\s+", "_", token)
