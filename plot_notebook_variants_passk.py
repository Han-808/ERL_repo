"""Merge and plot notebook-variant passk outputs.

This script uses only the Python standard library. It reads shard-level
rollouts.jsonl files, reconstructs per-sample pass rates, writes merged CSVs,
and emits SVG plots for:

* original-vs-updated violin distributions by state
* original-vs-updated median/std line charts by state
* cross-run updated median and updated-minus-original median comparisons
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from xml.sax.saxutils import escape


RUN_RE = re.compile(r"nothink-(origlogs|ownlogs)-(thinkahead|mechanism)-")
ENV_ORDER = ["frozen_lake", "sokoban"]
CONDITION_ORDER = ["original", "updated"]
RUN_ORDER = [
    ("origlogs", "thinkahead"),
    ("origlogs", "mechanism"),
    ("ownlogs", "thinkahead"),
    ("ownlogs", "mechanism"),
]
RUN_LABEL = {
    ("origlogs", "thinkahead"): "Original logs + Think-ahead",
    ("origlogs", "mechanism"): "Original logs + Mechanism",
    ("ownlogs", "thinkahead"): "Own logs + Think-ahead",
    ("ownlogs", "mechanism"): "Own logs + Mechanism",
}
RUN_COLOR = {
    ("origlogs", "thinkahead"): "#2563eb",
    ("origlogs", "mechanism"): "#dc2626",
    ("ownlogs", "thinkahead"): "#0891b2",
    ("ownlogs", "mechanism"): "#7c3aed",
}
COND_STYLE = {
    "original": {
        "label": "Original notebook",
        "fill": "#f97316",
        "stroke": "#c2410c",
        "median": "#7c2d12",
    },
    "updated": {
        "label": "Updated notebook",
        "fill": "#2563eb",
        "stroke": "#1d4ed8",
        "median": "#172554",
    },
}


def env_label(env: str) -> str:
    return {"frozen_lake": "FrozenLake", "sokoban": "Sokoban"}.get(env, env)


def run_key_from_dir(path: Path) -> tuple[str, str] | None:
    m = RUN_RE.search(path.name)
    if not m:
        return None
    return m.group(1), m.group(2)


def fmt(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def safe_stdev(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect(root: Path, expected_z: int) -> tuple[list[dict], list[dict], list[dict], bool]:
    sample_counts: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
    shard_rows: list[dict] = []
    all_complete = True

    shard_dirs = sorted([p for p in (root / "shards").iterdir() if p.is_dir()])
    for shard_dir in shard_dirs:
        run_key = run_key_from_dir(shard_dir)
        if run_key is None:
            continue
        source, variant = run_key
        rollouts_path = shard_dir / "rollouts.jsonl"
        config_path = shard_dir / "config.json"
        planned = 0
        if config_path.exists():
            planned = int(read_json(config_path).get("total_rollouts_planned", 0) or 0)
        actual = 0

        if not rollouts_path.exists():
            all_complete = False
            shard_rows.append(
                {
                    "source": source,
                    "variant": variant,
                    "shard": shard_dir.name,
                    "actual_rollouts": 0,
                    "planned_rollouts": planned,
                    "complete": False,
                }
            )
            continue

        with rollouts_path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                raw = raw.strip()
                if not raw:
                    continue
                row = json.loads(raw)
                actual += 1
                key = (
                    source,
                    variant,
                    row["env"],
                    int(row["state_x"]),
                    row.get("condition", "fixed"),
                    int(row["sample_y"]),
                )
                sample_counts[key][1] += 1
                if bool(row.get("success", False)):
                    sample_counts[key][0] += 1

        complete = bool(planned) and actual >= planned
        if not complete:
            all_complete = False
        shard_rows.append(
            {
                "source": source,
                "variant": variant,
                "shard": shard_dir.name,
                "actual_rollouts": actual,
                "planned_rollouts": planned,
                "complete": complete,
            }
        )

    sample_rows: list[dict] = []
    for key, (successes, total) in sorted(sample_counts.items()):
        source, variant, env, state_x, condition, sample_y = key
        sample_rows.append(
            {
                "source": source,
                "variant": variant,
                "run": f"{source}_{variant}",
                "env": env,
                "state_x": state_x,
                "condition": condition,
                "sample_y": sample_y,
                "num_games": total,
                "successes": successes,
                "pass_rate": successes / total if total else 0.0,
                "complete_sample": total >= expected_z,
            }
        )

    grouped: dict[tuple, list[float]] = defaultdict(list)
    games_by_state: dict[tuple, int] = defaultdict(int)
    complete_samples_by_state: dict[tuple, int] = defaultdict(int)
    for row in sample_rows:
        key = (
            row["source"],
            row["variant"],
            row["env"],
            row["state_x"],
            row["condition"],
        )
        games_by_state[key] += int(row["num_games"])
        if row["complete_sample"]:
            grouped[key].append(float(row["pass_rate"]))
            complete_samples_by_state[key] += 1

    state_rows: list[dict] = []
    for key, values in sorted(grouped.items()):
        source, variant, env, state_x, condition = key
        state_rows.append(
            {
                "source": source,
                "variant": variant,
                "run": f"{source}_{variant}",
                "env": env,
                "state_x": state_x,
                "condition": condition,
                "num_samples": len(values),
                "total_games": games_by_state[key],
                "mean_sample_pass_rate": mean(values) if values else 0.0,
                "median_sample_pass_rate": median(values) if values else 0.0,
                "std_sample_pass_rate": safe_stdev(values),
            }
        )

    return shard_rows, sample_rows, state_rows, all_complete


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def svg_header(width: int, height: int) -> list[str]:
    return [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        "<style>"
        "text{font-family:Arial,Helvetica,sans-serif;fill:#111827}"
        ".title{font-size:20px;font-weight:700}.subtitle{font-size:12px;fill:#4b5563}"
        ".panel{font-size:15px;font-weight:700}.tick{font-size:11px;fill:#4b5563}"
        ".label{font-size:13px;fill:#111827}.grid{stroke:#e5e7eb;stroke-width:1}"
        ".axis{stroke:#374151;stroke-width:1.2}.legend{font-size:12px}"
        "</style>",
    ]


def svg_finish(parts: list[str], path: Path) -> None:
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def scale_x(state: int, states: list[int], x0: float, w: float) -> float:
    lo, hi = min(states), max(states)
    if hi == lo:
        return x0 + w / 2
    return x0 + (state - lo) / (hi - lo) * w


def scale_y(value: float, y0: float, h: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return y0 + (hi - value) / (hi - lo) * h


def kernel_density(values: list[float], y: float, bandwidth: float = 0.075) -> float:
    return sum(math.exp(-0.5 * ((y - v) / bandwidth) ** 2) for v in values)


def violin_path(values: list[float], cx: float, y0: float, h: float, max_width: float) -> str:
    grid = [i / 50 for i in range(51)]
    dens = [kernel_density(values, y) for y in grid]
    max_d = max(dens) if dens else 1.0
    if max_d <= 0:
        max_d = 1.0
    left = []
    right = []
    for yv, d in zip(grid, dens):
        width = max_width * (d / max_d)
        py = scale_y(yv, y0, h)
        left.append((cx - width, py))
        right.append((cx + width, py))
    points = left + list(reversed(right))
    return "M " + " L ".join(f"{fmt(x,2)} {fmt(y,2)}" for x, y in points) + " Z"


def rows_for_group(sample_rows: list[dict], source: str, variant: str) -> list[dict]:
    return [r for r in sample_rows if r["source"] == source and r["variant"] == variant and r["complete_sample"]]


def plot_group_violin(sample_rows: list[dict], source: str, variant: str, out: Path, partial: bool) -> None:
    rows = rows_for_group(sample_rows, source, variant)
    width = 1560
    panel_h = 380
    margin_l, margin_r, margin_t, margin_b = 86, 44, 76, 70
    panel_gap = 46
    height = margin_t + margin_b + panel_h * 2 + panel_gap
    plot_w = width - margin_l - margin_r
    plot_h = panel_h - 84
    parts = svg_header(width, height)
    label = RUN_LABEL[(source, variant)]
    title = f"{label}: original vs updated sample accuracy by state"
    if partial:
        title += " (PARTIAL LOCAL DATA)"
    parts.append(f'<text x="{width/2:.1f}" y="32" text-anchor="middle" class="title">{escape(title)}</text>')
    parts.append(f'<text x="{width/2:.1f}" y="52" text-anchor="middle" class="subtitle">Each violin uses complete y-samples only; pass rate per sample is over z=20 test instances.</text>')

    lx = margin_l
    ly = 64
    for i, cond in enumerate(CONDITION_ORDER):
        st = COND_STYLE[cond]
        x = lx + i * 190
        parts.append(f'<rect x="{x}" y="{ly-10}" width="20" height="12" fill="{st["fill"]}" fill-opacity="0.62" stroke="{st["stroke"]}"/>')
        parts.append(f'<text x="{x+28}" y="{ly}" class="legend">{escape(st["label"])}</text>')

    for p_i, env in enumerate(ENV_ORDER):
        y_panel = margin_t + p_i * (panel_h + panel_gap)
        env_rows = [r for r in rows if r["env"] == env]
        states = sorted({int(r["state_x"]) for r in env_rows})
        if not states:
            continue
        parts.append(f'<text x="{margin_l}" y="{y_panel-16}" class="panel">{env_label(env)}</text>')
        for t in range(0, 11):
            val = t / 10
            y = scale_y(val, y_panel, plot_h)
            parts.append(f'<line x1="{margin_l}" y1="{fmt(y,2)}" x2="{margin_l+plot_w}" y2="{fmt(y,2)}" class="grid"/>')
            parts.append(f'<text x="{margin_l-10}" y="{fmt(y+4,2)}" text-anchor="end" class="tick">{t*10}%</text>')
        parts.append(f'<line x1="{margin_l}" y1="{y_panel}" x2="{margin_l}" y2="{y_panel+plot_h}" class="axis"/>')
        parts.append(f'<line x1="{margin_l}" y1="{y_panel+plot_h}" x2="{margin_l+plot_w}" y2="{y_panel+plot_h}" class="axis"/>')
        for tick in range(1, 41, 5):
            x = scale_x(tick, list(range(1, 41)), margin_l, plot_w)
            parts.append(f'<text x="{fmt(x,2)}" y="{y_panel+plot_h+22}" text-anchor="middle" class="tick">{tick}</text>')

        for state in states:
            cx = scale_x(state, list(range(1, 41)), margin_l, plot_w)
            for cond in ["updated", "original"]:
                vals = [
                    float(r["pass_rate"])
                    for r in env_rows
                    if int(r["state_x"]) == state and r["condition"] == cond
                ]
                if not vals:
                    continue
                st = COND_STYLE[cond]
                path = violin_path(vals, cx, y_panel, plot_h, max_width=11.5)
                opacity = 0.48 if cond == "updated" else 0.68
                parts.append(f'<path d="{path}" fill="{st["fill"]}" fill-opacity="{opacity}" stroke="{st["stroke"]}" stroke-width="1.0"/>')
                med = median(vals)
                my = scale_y(med, y_panel, plot_h)
                parts.append(f'<line x1="{fmt(cx-10,2)}" y1="{fmt(my,2)}" x2="{fmt(cx+10,2)}" y2="{fmt(my,2)}" stroke="{st["median"]}" stroke-width="2"/>')
        parts.append(f'<text x="{margin_l + plot_w/2:.1f}" y="{y_panel+plot_h+46}" text-anchor="middle" class="label">Notebook state</text>')
    svg_finish(parts, out)


def make_state_lookup(state_rows: list[dict]) -> dict[tuple, dict]:
    return {
        (r["source"], r["variant"], r["env"], int(r["state_x"]), r["condition"]): r
        for r in state_rows
    }


def plot_group_median_std(state_rows: list[dict], source: str, variant: str, out: Path, partial: bool) -> None:
    rows = [r for r in state_rows if r["source"] == source and r["variant"] == variant]
    width = 1560
    panel_h = 430
    margin_l, margin_r, margin_t, margin_b = 86, 44, 76, 70
    panel_gap = 50
    height = margin_t + margin_b + panel_h * 2 + panel_gap
    plot_w = (width - margin_l - margin_r - 60) / 2
    plot_h = panel_h - 92
    parts = svg_header(width, height)
    label = RUN_LABEL[(source, variant)]
    title = f"{label}: median and std by state"
    if partial:
        title += " (PARTIAL LOCAL DATA)"
    parts.append(f'<text x="{width/2:.1f}" y="32" text-anchor="middle" class="title">{escape(title)}</text>')
    parts.append(f'<text x="{width/2:.1f}" y="52" text-anchor="middle" class="subtitle">Median/std are computed across complete y-sample pass rates at each state.</text>')

    for p_i, env in enumerate(ENV_ORDER):
        y_panel = margin_t + p_i * (panel_h + panel_gap)
        parts.append(f'<text x="{margin_l}" y="{y_panel-16}" class="panel">{env_label(env)}</text>')
        for metric_i, metric in enumerate(["median_sample_pass_rate", "std_sample_pass_rate"]):
            x0 = margin_l + metric_i * (plot_w + 60)
            ymax = 1.0 if metric_i == 0 else 0.55
            metric_label = "Median accuracy" if metric_i == 0 else "Std across samples"
            for t in range(0, 6 if metric_i else 11):
                val = t / (10 if metric_i == 0 else 10)
                if val > ymax:
                    continue
                y = scale_y(val, y_panel, plot_h, 0, ymax)
                parts.append(f'<line x1="{x0}" y1="{fmt(y,2)}" x2="{x0+plot_w}" y2="{fmt(y,2)}" class="grid"/>')
                parts.append(f'<text x="{x0-10}" y="{fmt(y+4,2)}" text-anchor="end" class="tick">{int(val*100)}%</text>')
            parts.append(f'<line x1="{x0}" y1="{y_panel}" x2="{x0}" y2="{y_panel+plot_h}" class="axis"/>')
            parts.append(f'<line x1="{x0}" y1="{y_panel+plot_h}" x2="{x0+plot_w}" y2="{y_panel+plot_h}" class="axis"/>')
            parts.append(f'<text x="{x0+plot_w/2:.1f}" y="{y_panel-2}" text-anchor="middle" class="label">{metric_label}</text>')

            for cond in CONDITION_ORDER:
                vals_by_state = [
                    (int(r["state_x"]), float(r[metric]))
                    for r in rows
                    if r["env"] == env and r["condition"] == cond
                ]
                vals_by_state.sort()
                if not vals_by_state:
                    continue
                avg = mean(v for _, v in vals_by_state)
                avg_y = scale_y(avg, y_panel, plot_h, 0, ymax)
                st = COND_STYLE[cond]
                parts.append(f'<line x1="{x0}" y1="{fmt(avg_y,2)}" x2="{x0+plot_w}" y2="{fmt(avg_y,2)}" stroke="{st["stroke"]}" stroke-width="1.2" stroke-dasharray="6 5" opacity="0.65"/>')
                points = []
                for state, val in vals_by_state:
                    x = scale_x(state, list(range(1, 41)), x0, plot_w)
                    y = scale_y(val, y_panel, plot_h, 0, ymax)
                    points.append(f"{fmt(x,2)},{fmt(y,2)}")
                parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{st["stroke"]}" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>')
        parts.append(f'<text x="{margin_l}" y="{y_panel+plot_h+38}" class="legend" fill="{COND_STYLE["original"]["stroke"]}">Original notebook</text>')
        parts.append(f'<text x="{margin_l+160}" y="{y_panel+plot_h+38}" class="legend" fill="{COND_STYLE["updated"]["stroke"]}">Updated notebook</text>')
    svg_finish(parts, out)


def plot_cross_lines(state_rows: list[dict], out: Path, metric: str, title: str, partial: bool) -> None:
    width = 1560
    panel_h = 380
    margin_l, margin_r, margin_t, margin_b = 86, 44, 82, 70
    panel_gap = 46
    height = margin_t + margin_b + panel_h * 2 + panel_gap
    plot_w = width - margin_l - margin_r
    plot_h = panel_h - 84
    parts = svg_header(width, height)
    if partial:
        title += " (PARTIAL LOCAL DATA)"
    parts.append(f'<text x="{width/2:.1f}" y="32" text-anchor="middle" class="title">{escape(title)}</text>')
    parts.append(f'<text x="{width/2:.1f}" y="52" text-anchor="middle" class="subtitle">Lines compare the four variant-evaluation settings.</text>')

    legend_y = 68
    legend_x = margin_l
    for i, rk in enumerate(RUN_ORDER):
        x = legend_x + i * 300
        parts.append(f'<line x1="{x}" y1="{legend_y}" x2="{x+34}" y2="{legend_y}" stroke="{RUN_COLOR[rk]}" stroke-width="4"/>')
        parts.append(f'<text x="{x+42}" y="{legend_y+4}" class="legend">{escape(RUN_LABEL[rk])}</text>')

    lookup = make_state_lookup(state_rows)
    for p_i, env in enumerate(ENV_ORDER):
        y_panel = margin_t + p_i * (panel_h + panel_gap)
        all_vals: list[float] = []
        series: dict[tuple[str, str], list[tuple[int, float]]] = {}
        for rk in RUN_ORDER:
            source, variant = rk
            vals = []
            for state in range(1, 41):
                if metric == "delta_median":
                    u = lookup.get((source, variant, env, state, "updated"))
                    o = lookup.get((source, variant, env, state, "original"))
                    if u and o:
                        val = float(u["median_sample_pass_rate"]) - float(o["median_sample_pass_rate"])
                    else:
                        continue
                else:
                    row = lookup.get((source, variant, env, state, "updated"))
                    if not row:
                        continue
                    val = float(row[metric])
                vals.append((state, val))
                all_vals.append(val)
            series[rk] = vals
        if metric == "delta_median":
            ymin, ymax = -0.55, 0.55
        elif metric == "std_sample_pass_rate":
            ymin, ymax = 0.0, 0.55
        else:
            ymin, ymax = 0.0, 1.0
        parts.append(f'<text x="{margin_l}" y="{y_panel-16}" class="panel">{env_label(env)}</text>')
        for t in range(0, 11):
            val = ymin + (ymax - ymin) * t / 10
            y = scale_y(val, y_panel, plot_h, ymin, ymax)
            parts.append(f'<line x1="{margin_l}" y1="{fmt(y,2)}" x2="{margin_l+plot_w}" y2="{fmt(y,2)}" class="grid"/>')
            parts.append(f'<text x="{margin_l-10}" y="{fmt(y+4,2)}" text-anchor="end" class="tick">{int(val*100)}%</text>')
        if ymin < 0 < ymax:
            zy = scale_y(0, y_panel, plot_h, ymin, ymax)
            parts.append(f'<line x1="{margin_l}" y1="{fmt(zy,2)}" x2="{margin_l+plot_w}" y2="{fmt(zy,2)}" stroke="#111827" stroke-width="1.2" stroke-dasharray="5 5"/>')
        parts.append(f'<line x1="{margin_l}" y1="{y_panel}" x2="{margin_l}" y2="{y_panel+plot_h}" class="axis"/>')
        parts.append(f'<line x1="{margin_l}" y1="{y_panel+plot_h}" x2="{margin_l+plot_w}" y2="{y_panel+plot_h}" class="axis"/>')
        for rk, vals in series.items():
            if not vals:
                continue
            pts = []
            for state, val in vals:
                x = scale_x(state, list(range(1, 41)), margin_l, plot_w)
                y = scale_y(val, y_panel, plot_h, ymin, ymax)
                pts.append(f"{fmt(x,2)},{fmt(y,2)}")
            parts.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{RUN_COLOR[rk]}" stroke-width="2.6" stroke-linejoin="round" stroke-linecap="round"/>')
    svg_finish(parts, out)


def write_overall_summary(state_rows: list[dict], out: Path) -> None:
    rows = []
    for source, variant in RUN_ORDER:
        for env in ENV_ORDER:
            for condition in CONDITION_ORDER:
                vals = [
                    float(r["median_sample_pass_rate"])
                    for r in state_rows
                    if r["source"] == source
                    and r["variant"] == variant
                    and r["env"] == env
                    and r["condition"] == condition
                ]
                std_vals = [
                    float(r["std_sample_pass_rate"])
                    for r in state_rows
                    if r["source"] == source
                    and r["variant"] == variant
                    and r["env"] == env
                    and r["condition"] == condition
                ]
                rows.append(
                    {
                        "source": source,
                        "variant": variant,
                        "env": env,
                        "condition": condition,
                        "num_states": len(vals),
                        "mean_state_median": mean(vals) if vals else "",
                        "mean_state_std": mean(std_vals) if std_vals else "",
                    }
                )
    write_csv(out, rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="notebook_variants_passk_y8z20_seed20260509_complete",
        help="Extracted package root containing shards/.",
    )
    parser.add_argument("--expected-z", type=int, default=20)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    if not (root / "shards").exists():
        raise SystemExit(f"Missing shards directory under {root}")

    tmp_out = Path(args.out or "notebook_variants_passk_y8z20_seed20260509_plots")
    shard_rows, sample_rows, state_rows, all_complete = collect(root, args.expected_z)
    out = tmp_out if all_complete else Path(str(tmp_out) + "_PARTIAL_LOCAL_DATA")
    out.mkdir(parents=True, exist_ok=True)

    write_csv(out / "completeness_by_shard.csv", shard_rows)
    write_csv(out / "merged_sample_summary_from_rollouts.csv", sample_rows)
    write_csv(out / "merged_state_summary_from_rollouts.csv", state_rows)
    write_overall_summary(state_rows, out / "overall_state_summary.csv")

    complete_sample_rows = [r for r in sample_rows if r["complete_sample"]]
    for source, variant in RUN_ORDER:
        if any(r["source"] == source and r["variant"] == variant for r in complete_sample_rows):
            plot_group_violin(
                complete_sample_rows,
                source,
                variant,
                out / f"{source}_{variant}_violin_original_vs_updated.svg",
                partial=not all_complete,
            )
            plot_group_median_std(
                state_rows,
                source,
                variant,
                out / f"{source}_{variant}_median_std_original_vs_updated.svg",
                partial=not all_complete,
            )

    plot_cross_lines(
        state_rows,
        out / "crossrun_updated_median_by_state.svg",
        "median_sample_pass_rate",
        "Updated notebook median accuracy by state",
        partial=not all_complete,
    )
    plot_cross_lines(
        state_rows,
        out / "crossrun_updated_std_by_state.svg",
        "std_sample_pass_rate",
        "Updated notebook std across samples by state",
        partial=not all_complete,
    )
    plot_cross_lines(
        state_rows,
        out / "crossrun_delta_median_updated_minus_original.svg",
        "delta_median",
        "Median accuracy delta: updated minus original",
        partial=not all_complete,
    )

    print(f"Wrote {out}")
    print(f"All shards complete: {all_complete}")
    for row in shard_rows:
        print(
            f"{row['actual_rollouts']:5d} / {row['planned_rollouts']:5d} "
            f"{row['shard']}"
        )


if __name__ == "__main__":
    main()
