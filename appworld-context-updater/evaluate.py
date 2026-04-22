"""
Aggregate evaluation for AppWorld context-updater experiments.

Usage:
    # Evaluate specific experiments on the train split:
    python evaluate.py ace-qwen3-8b ace_nogt-qwen3-8b --dataset train

    # Evaluate all subdirs and export CSV table (rows=methods, columns=models):
    python evaluate.py --all --dataset train

Skips evaluation automatically if evaluations/{dataset}.json already exists.
"""

import argparse
import csv
import io
import json
import os
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
ACE_APPWORLD_ROOT = REPO_ROOT / "libs" / "ace-appworld"

os.environ.setdefault("APPWORLD_ROOT", str(ACE_APPWORLD_ROOT))
if str(ACE_APPWORLD_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ACE_APPWORLD_ROOT / "src"))


def run_evaluation(
    experiment_name: str,
    dataset: str,
    outputs_dir: Path,
) -> dict | None:
    from appworld.evaluator import evaluate_dataset

    # Point AppWorld at our outputs dir
    os.environ["APPWORLD_EXPERIMENT_OUTPUTS"] = str(outputs_dir)

    eval_json = outputs_dir / experiment_name / "evaluations" / f"{dataset}.json"
    if eval_json.exists():
        print(f"[skip] {experiment_name} — results already exist at {eval_json}")
        with open(eval_json) as f:
            return json.load(f)

    print(f"\n{'='*60}")
    print(f"Evaluating: {experiment_name}  (dataset={dataset})")
    print(f"{'='*60}")
    try:
        result = evaluate_dataset(
            experiment_name=experiment_name,
            dataset_name=dataset,
            suppress_errors=True,
            include_details=True,
            aggregate_only=False,
            save_reports=True,
            print_report=True,
        )
        return result
    except Exception as exc:
        print(f"ERROR evaluating {experiment_name}: {exc}")
        return None


def parse_experiment_name(name: str) -> tuple[str, str]:
    """Split 'method-modeltag' into (method, model). E.g. 'ace_nogt-qwen3-8b' -> ('ace_nogt', 'qwen3-8b')."""
    match = re.match(r"^(.+?)-(qwen3.+)$", name)
    if match:
        return match.group(1), match.group(2)
    return name, ""


def print_summary_table(results: dict[str, dict | None]) -> None:
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    header = f"{'experiment':<40} {'task_goal':>10} {'scenario_goal':>14}"
    print(header)
    print("-" * len(header))
    for name, result in sorted(results.items()):
        if result is None:
            print(f"{name:<40} {'ERROR':>10} {'ERROR':>14}")
            continue
        agg = result.get("aggregate", {})
        tgc = agg.get("task_goal_completion", "?")
        sgc = agg.get("scenario_goal_completion", "?")
        print(f"{name:<40} {str(tgc):>10} {str(sgc):>14}")
    print()


def export_csv(results: dict[str, dict | None], metric: str = "task_goal_completion") -> str:
    """Export a CSV table: rows=methods, columns=models."""
    methods: dict[str, dict[str, str]] = {}
    models: set[str] = set()
    for name, result in results.items():
        method, model = parse_experiment_name(name)
        if not model:
            continue
        models.add(model)
        if result is None:
            methods.setdefault(method, {})[model] = "ERROR"
        else:
            agg = result.get("aggregate", {})
            val = agg.get(metric, "?")
            methods.setdefault(method, {})[model] = str(val)

    sorted_models = sorted(models, key=lambda m: (re.search(r"\d+", m) and int(re.search(r"\d+", m).group()) or 0))
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["method"] + sorted_models)
    for method in sorted(methods):
        row = [method] + [methods[method].get(m, "") for m in sorted_models]
        writer.writerow(row)
    return buf.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AppWorld context-updater experiments.")
    parser.add_argument(
        "experiments",
        nargs="*",
        help="Experiment names to evaluate (subdirs under --outputs-dir). "
             "If omitted, use --all to evaluate everything.",
    )
    parser.add_argument("--dataset", default="train", help="Dataset split to evaluate on (default: train)")
    parser.add_argument(
        "--outputs-dir",
        default=str(SCRIPT_DIR / "outputs"),
        help="Base outputs directory (default: ./outputs)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all experiment subdirs found under --outputs-dir",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Only generate CSV from existing evaluation results (no new evaluations)",
    )
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir).resolve()
    if not outputs_dir.exists():
        print(f"Outputs dir does not exist: {outputs_dir}")
        sys.exit(1)

    if args.csv:
        # Only load existing eval JSONs and produce CSV
        results: dict[str, dict | None] = {}
        for d in sorted(outputs_dir.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            eval_json = d / "evaluations" / f"{args.dataset}.json"
            if eval_json.exists():
                with open(eval_json) as f:
                    results[d.name] = json.load(f)
        if not results:
            print(f"No existing evaluations found under {outputs_dir}")
            sys.exit(1)
        csv_text = export_csv(results)
        csv_path = outputs_dir / f"summary_{args.dataset}.csv"
        csv_path.write_text(csv_text, encoding="utf-8")
        print(f"CSV saved to {csv_path}")
        print(csv_text)
        return

    if args.all:
        experiment_names = sorted(
            d.name for d in outputs_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
        if not experiment_names:
            print(f"No experiment dirs found under {outputs_dir}")
            sys.exit(1)
    elif args.experiments:
        experiment_names = args.experiments
    else:
        parser.error("Provide experiment names or use --all or --csv")

    results: dict[str, dict | None] = {}
    for name in experiment_names:
        results[name] = run_evaluation(
            experiment_name=name,
            dataset=args.dataset,
            outputs_dir=outputs_dir,
        )

    print_summary_table(results)

    if args.all:
        csv_text = export_csv(results)
        csv_path = outputs_dir / f"summary_{args.dataset}.csv"
        csv_path.write_text(csv_text, encoding="utf-8")
        print(f"CSV saved to {csv_path}")
        print(csv_text)


if __name__ == "__main__":
    main()
