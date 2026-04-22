import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import run_experiment
from methods.ace import ACEModel
from methods.ace_aed import ACEAEDModel
from methods.ace_once import ACEOnceModel
from methods.hypothesis_v1 import HypothesisV1Model
from methods.hypothesis_v2 import HypothesisV2Model
from methods.hypothesis_v3 import HypothesisV3Model
from methods.hypothesis_v4 import HypothesisV4Model
from methods.notebook_minimal import NotebookMinimalModel
from methods.summary_delta_v1 import SummaryDeltaV1Model
from methods.summary_v1 import SummaryV1Model

METHODS = {
    "summary_v1": SummaryV1Model,
    "summary_delta_v1": SummaryDeltaV1Model,
    # ACE: default = full ACE (GT + test report + initial playbook)
    "ace": lambda: ACEModel(use_ground_truth=True, use_test_report=True, initial_playbook="default"),
    # ACE ablations: ground truth
    "ace_nogt": lambda: ACEModel(use_ground_truth=False, use_test_report=True, initial_playbook="default"),
    # ACE ablations: test report
    "ace_notest": lambda: ACEModel(use_ground_truth=True, use_test_report=False, initial_playbook="default"),
    # ACE ablations: initial playbook variants
    "ace_pb_empty": lambda: ACEModel(use_ground_truth=True, use_test_report=True, initial_playbook="empty"),
    "ace_pb_null": lambda: ACEModel(use_ground_truth=True, use_test_report=True, initial_playbook="null"),
    # ACE ablations: no GT + no test (minimal)
    "ace_nogt_notest": lambda: ACEModel(use_ground_truth=False, use_test_report=False, initial_playbook="default"),
    # ACE-AED: ADD/EDIT/DELETE curator
    "ace_aed": ACEAEDModel,
    # ace_once: reflector + curator merged into one call, ADD-only (ACE ablation)
    "ace_once": ACEOnceModel,
    "ace_once_nogt": lambda: ACEOnceModel(use_ground_truth=False),
    "ace_once_notest": lambda: ACEOnceModel(use_test_report=False),
    # notebook_minimal: free-form notebook with line-number editing
    "notebook_minimal": NotebookMinimalModel,
    "notebook_minimal_nogt": lambda: NotebookMinimalModel(use_ground_truth=False),
    "notebook_minimal_empty": lambda: NotebookMinimalModel(initial_notebook="empty"),
    "hypothesis_v1": HypothesisV1Model,
    "hypothesis_v2": HypothesisV2Model,
    "hypothesis_v3": lambda: HypothesisV3Model(use_ground_truth=True),
    "hypothesis_v3_nogt": lambda: HypothesisV3Model(use_ground_truth=False),
    "hypothesis_v4": lambda: HypothesisV4Model(use_ground_truth=True),
    "hypothesis_v4_nogt": lambda: HypothesisV4Model(use_ground_truth=False),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an AppWorld context-update experiment.")
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--model-name", "--model_name", dest="model_name", required=True)
    parser.add_argument("--context-model-name", "--context_model_name", dest="context_model_name", default=None)
    parser.add_argument("--max-steps", dest="max_steps", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", dest="top_p", type=float, default=1.0)
    parser.add_argument("--presence-penalty", dest="presence_penalty", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--dataset", default="train")
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--task-offset", type=int, default=0)
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument("--task-seed", type=int, default=100,
        help="If set, shuffle task order deterministically before applying offset/limit.",
    )
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--experiment-outputs-dir", default="./outputs")
    parser.add_argument("--context-save-every", type=int, default=10)
    parser.add_argument("--sglang-base-url", default=None)
    parser.add_argument("--skip-server-launch", action="store_true")
    parser.add_argument("--sglang-host", default="127.0.0.1")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--sglang-extra-args", nargs="*", default=None)
    parser.add_argument("--random-seed", type=int, default=100)
    parser.add_argument(
        "--disable-thinking", dest="disable_thinking", action="store_true",
        help="Disable chain-of-thought reasoning for Qwen3 models "
             "(sets chat_template_kwargs enable_thinking=False).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model = METHODS[args.method]()
    run_experiment(model, args)


if __name__ == "__main__":
    main()
