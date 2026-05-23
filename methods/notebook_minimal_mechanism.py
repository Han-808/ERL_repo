"""Mechanism-discovery updater variant for notebook_minimal."""

from common import call_lm
from methods.notebook_minimal import (
    NotebookMinimalMethod,
    apply_notebook_operations,
    extract_json_payload,
    number_lines,
    validate_operations,
)
from prompts import build_notebook_updater_prompt


MECHANISM_UPDATER_OBJECTIVE = """\
The notebook updater should prioritize learning reusable environment mechanics:
what actions are allowed, what blocks movement, what causes terminal
success/failure, what objects can be moved, and what preconditions make an
action effective.
"""


class NotebookMinimalMechanismMethod(NotebookMinimalMethod):
    """NotebookMinimalMethod with an environment-mechanism updater objective."""

    name = "notebook_minimal_mechanism"
    variant_name = "notebook_minimal_mechanism"
    updater_objective = MECHANISM_UPDATER_OBJECTIVE

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        suffix = "" if self.initial_notebook == "empty" else f"_{self.initial_notebook}"
        self.name = f"{self.variant_name}{suffix}"

    def _update_notebook(self, initial_obs, actions, feedback, reward):
        notebook_before = self.notebook
        prompt = build_notebook_updater_prompt(
            numbered_notebook=number_lines(self.notebook),
            initial_obs=initial_obs,
            actions=actions,
            feedback=feedback,
            reward=reward,
            reward_threshold=self.reward_threshold,
            updater_objective=self.updater_objective,
        )
        raw = call_lm(
            self.client, self.model, prompt,
            disable_thinking=self.disable_thinking,
        )
        update_info = {
            "raw_output": raw,
            "reasoning": "",
            "operations": [],
            "applied_operations": [],
            "parse_error": None,
            "notebook_before": notebook_before,
            "notebook_after": notebook_before,
            "notebook_changed": False,
            "updater_objective": self.updater_objective,
        }
        if not raw.strip():
            print("[notebook_minimal] empty updater response; no edit.")
            update_info["parse_error"] = "empty updater response"
            return update_info
        try:
            payload = extract_json_payload(raw)
        except Exception as exc:
            print(f"[notebook_minimal] JSON parse failed: {exc}")
            update_info["parse_error"] = str(exc)
            return update_info
        ops = validate_operations(payload.get("operations", []))
        reasoning = payload.get("reasoning", "")
        new_notebook, applied = apply_notebook_operations(self.notebook, ops)
        self.notebook = new_notebook
        update_info.update({
            "reasoning": reasoning,
            "operations": ops,
            "applied_operations": applied,
            "notebook_after": self.notebook,
            "notebook_changed": self.notebook != notebook_before,
        })
        if applied:
            print(f"[Notebook] {len(applied)} ops applied; now "
                  f"{len(self.notebook.splitlines())} lines.")
        else:
            print("[Notebook] no ops applied.")
        return update_info


class NotebookMinimalMechanismMiniGridMethod(NotebookMinimalMechanismMethod):
    """MiniGrid-compatible alias for notebook_minimal_mechanism."""

    name = "notebook_minimal_mechanism_minigrid"
    variant_name = "notebook_minimal_mechanism_minigrid"
