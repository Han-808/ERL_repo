"""Think-ahead updater variant for notebook_minimal."""

from common import call_lm
from methods.notebook_minimal import (
    NotebookMinimalMethod,
    apply_notebook_operations,
    extract_json_payload,
    number_lines,
    validate_operations,
)
from prompts import build_notebook_updater_prompt


THINKAHEAD_UPDATER_OBJECTIVE = """\
Update the notebook with rules that help future episodes plan over several steps.
Prefer notes about how an action changes future options, reachable positions,
remaining-step feasibility, or terminal risk, rather than only whether the
immediate action received reward.
"""


class NotebookMinimalThinkAheadMethod(NotebookMinimalMethod):
    """NotebookMinimalMethod with a multi-step planning updater objective."""

    name = "notebook_minimal_thinkahead"
    variant_name = "notebook_minimal_thinkahead"
    updater_objective = THINKAHEAD_UPDATER_OBJECTIVE

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        suffix = "" if self.initial_notebook == "empty" else f"_{self.initial_notebook}"
        self.name = f"{self.variant_name}{suffix}"

    def _update_notebook(self, initial_obs, actions, feedback, reward):
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
        if not raw.strip():
            print("[notebook_minimal] empty updater response; no edit.")
            return ""
        try:
            payload = extract_json_payload(raw)
        except Exception as exc:
            print(f"[notebook_minimal] JSON parse failed: {exc}")
            return ""
        ops = validate_operations(payload.get("operations", []))
        reasoning = payload.get("reasoning", "")
        new_notebook, applied = apply_notebook_operations(self.notebook, ops)
        self.notebook = new_notebook
        if applied:
            print(f"[Notebook] {len(applied)} ops applied; now "
                  f"{len(self.notebook.splitlines())} lines.")
        else:
            print("[Notebook] no ops applied.")
        return reasoning
