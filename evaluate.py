"""
Aggregate and compare results_<method>_<env>.json files.

Mirrors appworld-context-updater/evaluate.py in spirit: a small CLI
that loads one or more stored results JSON files and prints a
side-by-side success-rate summary.
"""

import argparse
import csv
from pathlib import Path

from common import load_results, results_path


def parse_result_file(path: Path) -> dict:
    name = path.stem  # results_<method>_<env>
    parts = name.split("_", 2)
    method = parts[1] if len(parts) > 1 else "?"
    env = parts[2] if len(parts) > 2 else "?"
    data = load_results(path)
    return {
        "method": method,
        "env": env,
        "episodes": len(data.get("logs", [])),
        "attempt1_rate": data.get("attempt1_rate"),
        "attempt2_rate": data.get("attempt2_rate"),
        "improvement": data.get("improvement"),
        "running_attempt1_rate": data.get("running_attempt1_rate", []),
        "running_attempt2_rate": data.get("running_attempt2_rate", []),
    }


def discover_results(outputs_dir: str) -> list:
    outputs = Path(outputs_dir)
    if not outputs.exists():
        return []
    return sorted(outputs.glob("results_*.json"))


def print_summary_table(rows: list) -> None:
    if not rows:
        print("(no results found)")
        return

    W = {"method": 10, "env": 12, "eps": 5, "a1": 10, "a2": 10, "imp": 10}

    def fmt_rate(r):
        return "-" if r is None else f"{r*100:.1f}%"

    def fmt_imp(r):
        return "-" if r is None else f"{r*100:+.1f}%"

    def line(left, mid, right, fill="─"):
        return left + mid.join(fill * (w + 2) for w in W.values()) + right

    def row(vals):
        return "│" + "│".join(
            f" {str(v).center(w)} " for v, w in zip(vals, W.values())
        ) + "│"

    print(line("┌", "┬", "┐"))
    print(row(["Method", "Env", "Eps", "Attempt 1", "Attempt 2", "Δ"]))
    print(line("├", "┼", "┤"))
    for r in rows:
        print(row([
            r["method"], r["env"], r["episodes"],
            fmt_rate(r["attempt1_rate"]),
            fmt_rate(r["attempt2_rate"]),
            fmt_imp(r["improvement"]),
        ]))
    print(line("└", "┴", "┘"))


def export_csv(rows: list, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=[
                "method", "env", "episodes",
                "attempt1_rate", "attempt2_rate", "improvement",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"CSV written to {out_path}")


def export_curve_csv(rows: list, out_path: Path) -> None:
    """Wide CSV with one row per K and one column per (method, env, attempt)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    max_k = max((r["episodes"] for r in rows), default=0)
    if max_k == 0:
        print("(no episodes to export)")
        return

    headers = ["K"]
    series = []
    for r in rows:
        tag = f"{r['method']}_{r['env']}"
        headers.append(f"{tag}_a1")
        headers.append(f"{tag}_a2")
        series.append(r["running_attempt1_rate"])
        series.append(r["running_attempt2_rate"])

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for k in range(max_k):
            line = [k + 1]
            for s in series:
                line.append(f"{s[k]:.4f}" if k < len(s) else "")
            writer.writerow(line)
    print(f"Curve CSV written to {out_path}")


def print_curve_waypoints(rows: list) -> None:
    """Print running attempt-1 pass rate at 25/50/75/100% of episodes."""
    if not rows:
        return
    print("\nRunning attempt-1 pass rate (online — first K episodes):")
    print(f"  {'method_env':<22}  {'K=25%':>7}  {'K=50%':>7}  "
          f"{'K=75%':>7}  {'K=100%':>7}")
    for r in rows:
        curve = r["running_attempt1_rate"]
        n = len(curve)
        if n == 0:
            continue
        tag = f"{r['method']}_{r['env']}"
        vals = []
        for frac in (0.25, 0.5, 0.75, 1.0):
            k = max(1, int(round(frac * n)))
            vals.append(f"{curve[k-1]*100:5.1f}%")
        print(f"  {tag:<22}  " + "  ".join(f"{v:>7}" for v in vals))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate experiment results.")
    parser.add_argument(
        "experiments", nargs="*",
        help="Explicit <method>_<env> pairs (e.g. ace_frozen_lake). "
             "If omitted, every results_*.json under --outputs-dir is scanned.",
    )
    parser.add_argument("--outputs-dir", dest="outputs_dir", default="./outputs")
    parser.add_argument("--all", action="store_true",
                        help="Include every results_*.json regardless of positional args.")
    parser.add_argument("--csv", type=str, default=None,
                        help="Optional summary CSV output path.")
    parser.add_argument("--curve", action="store_true",
                        help="Print running attempt-1 pass rate at "
                             "K=25/50/75/100%% waypoints.")
    parser.add_argument("--curve-csv", dest="curve_csv", default=None,
                        help="Optional full-curve CSV output path "
                             "(one row per K).")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.all or not args.experiments:
        paths = discover_results(args.outputs_dir)
    else:
        paths = []
        for spec in args.experiments:
            # spec may be "<method>_<env>" or a bare filename
            if "_" in spec:
                method, _, env = spec.partition("_")
                paths.append(results_path(args.outputs_dir, method, env))
            else:
                paths.append(Path(args.outputs_dir) / spec)

    rows = []
    for p in paths:
        if not p.exists():
            print(f"[skip] {p} not found")
            continue
        rows.append(parse_result_file(p))

    print_summary_table(rows)
    if args.curve:
        print_curve_waypoints(rows)
    if args.csv:
        export_csv(rows, Path(args.csv))
    if args.curve_csv:
        export_curve_csv(rows, Path(args.curve_csv))


if __name__ == "__main__":
    main()
