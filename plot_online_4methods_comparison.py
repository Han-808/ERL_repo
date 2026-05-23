#!/usr/bin/env python3
"""
Compare online accuracy for four grid-game methods.

Methods:
  - ace_once
  - notebook_minimal
  - notebook_minimal_thinkahead
  - notebook_minimal_mechanism

The script searches for results_<method>_<env>.json files, picks the best
candidate for each method/env pair, and writes a unified SVG with two panels
(FrozenLake and Sokoban), each containing the four method lines.

Usage:
  python plot_online_4methods_comparison.py
  python plot_online_4methods_comparison.py --input-root runs --expected-episodes 80
  python plot_online_4methods_comparison.py --prefer-substring online-qwen3-14b-nothink-4methods-k80
  python plot_online_4methods_comparison.py --x-limit 90
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


METHODS = [
    ("ace_once", "ACE-once", "#059669"),
    ("notebook_minimal", "Minimal notebook", "#111827"),
    ("notebook_minimal_thinkahead", "Think-ahead", "#2563eb"),
    ("notebook_minimal_mechanism", "Mechanism", "#dc2626"),
]

ENVS = [
    ("frozen_lake", "FrozenLake"),
    ("sokoban", "Sokoban"),
]

SVG_FONT = "Segoe UI, Helvetica Neue, Arial, sans-serif"


@dataclass
class ResultSeries:
    method: str
    label: str
    color: str
    env: str
    env_label: str
    path: Path
    rewards: list[float]
    running_accuracy: list[float]

    @property
    def episodes(self) -> int:
        return len(self.rewards)

    @property
    def successes(self) -> int:
        return sum(1 for reward in self.rewards if reward >= 1.0)

    @property
    def pass_rate(self) -> float:
        return self.successes / self.episodes if self.episodes else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot online running accuracy for ACE-once and notebook variants."
    )
    parser.add_argument(
        "--input-root",
        default=".",
        help="Directory to search recursively for result JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default="online_4methods_comparison_visuals",
        help="Directory where SVGs and CSV summary will be written.",
    )
    parser.add_argument(
        "--expected-episodes",
        type=int,
        default=80,
        help="Expected episode count per env; used for warnings and chart ticks.",
    )
    parser.add_argument(
        "--x-limit",
        type=int,
        default=None,
        help=(
            "Maximum episode shown on the x-axis. Defaults to the larger of "
            "expected episodes and the loaded result length."
        ),
    )
    parser.add_argument(
        "--prefer-substring",
        default="online-qwen3-14b-nothink-4methods-k80",
        help=(
            "Prefer result paths containing this substring when several runs exist. "
            "Use an empty string to disable this preference."
        ),
    )
    return parser.parse_args()


def load_result(path: Path) -> tuple[list[float], dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    logs = data.get("logs", [])
    rewards: list[float] = []
    for row in logs:
        if not isinstance(row, dict):
            continue
        try:
            rewards.append(float(row.get("reward1", 0.0)))
        except (TypeError, ValueError):
            rewards.append(0.0)
    return rewards, data


def running_accuracy(rewards: Iterable[float]) -> list[float]:
    out: list[float] = []
    hits = 0
    for i, reward in enumerate(rewards, start=1):
        if reward >= 1.0:
            hits += 1
        out.append(hits / i)
    return out


def candidate_score(
    path: Path,
    rewards: list[float],
    expected_episodes: int,
    prefer_substring: str,
) -> tuple[int, int, int, float]:
    path_text = str(path).replace("\\", "/")
    preferred = 1 if prefer_substring and prefer_substring in path_text else 0
    complete = 1 if len(rewards) >= expected_episodes else 0
    return (preferred, complete, len(rewards), path.stat().st_mtime)


def find_best_result(
    root: Path,
    method: str,
    env: str,
    expected_episodes: int,
    prefer_substring: str,
) -> ResultSeries | None:
    name = f"results_{method}_{env}.json"
    candidates = []
    for path in root.rglob(name):
        if "code_snapshot" in path.parts:
            continue
        try:
            rewards, _ = load_result(path)
        except Exception:
            continue
        score = candidate_score(path, rewards, expected_episodes, prefer_substring)
        candidates.append((score, path, rewards))
    if not candidates:
        return None
    _, path, rewards = max(candidates, key=lambda item: item[0])
    method_label, color = next((label, color) for key, label, color in METHODS if key == method)
    env_label = next(label for key, label in ENVS if key == env)
    return ResultSeries(
        method=method,
        label=method_label,
        color=color,
        env=env,
        env_label=env_label,
        path=path,
        rewards=rewards,
        running_accuracy=running_accuracy(rewards),
    )


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def svg_text(text: str) -> str:
    return html.escape(text, quote=True)


def points_for_values(
    values: list[float],
    x0: float,
    y0: float,
    plot_w: float,
    plot_h: float,
    max_episode: int,
) -> str:
    points = []
    denom = max(1, max_episode - 1)
    for i, value in enumerate(values[:max_episode], start=1):
        x = x0 + ((i - 1) / denom) * plot_w
        y = y0 + (1.0 - value) * plot_h
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def episode_ticks(max_episode: int) -> list[int]:
    base = [1, 10, 20, 40, 60, 80, 100, 120]
    ticks = [tick for tick in base if tick <= max_episode]
    if max_episode not in ticks:
        ticks.append(max_episode)
    return sorted(set(ticks))


def draw_panel(
    svg: list[str],
    title: str,
    series: list[ResultSeries],
    x0: float,
    y0: float,
    plot_w: float,
    plot_h: float,
    max_episode: int,
) -> None:
    svg.append(
        f"<text x='{x0:.2f}' y='{y0 - 42:.2f}' font-family='{SVG_FONT}' "
        f"font-size='20' font-weight='700' fill='#111827'>{svg_text(title)}</text>"
    )
    svg.append(
        f"<text x='{x0:.2f}' y='{y0 - 18:.2f}' font-family='{SVG_FONT}' "
        f"font-size='12' fill='#4b5563'>Running accuracy</text>"
    )

    for t in range(0, 11):
        acc = t / 10.0
        y = y0 + (1.0 - acc) * plot_h
        stroke = "#111827" if t == 0 else "#e5e7eb"
        svg.append(
            f"<line x1='{x0:.2f}' y1='{y:.2f}' x2='{x0 + plot_w:.2f}' y2='{y:.2f}' "
            f"stroke='{stroke}' stroke-width='1'/>"
        )
        svg.append(
            f"<text x='{x0 - 10:.2f}' y='{y + 4:.2f}' text-anchor='end' "
            f"font-family='{SVG_FONT}' font-size='12' fill='#4b5563'>{t * 10}%</text>"
        )

    for tick in episode_ticks(max_episode):
        x = x0 + ((tick - 1) / max(1, max_episode - 1)) * plot_w
        svg.append(
            f"<line x1='{x:.2f}' y1='{y0:.2f}' x2='{x:.2f}' y2='{y0 + plot_h:.2f}' "
            f"stroke='#f3f4f6' stroke-width='1'/>"
        )
        svg.append(
            f"<text x='{x:.2f}' y='{y0 + plot_h + 24:.2f}' text-anchor='middle' "
            f"font-family='{SVG_FONT}' font-size='12' fill='#4b5563'>{tick}</text>"
        )

    svg.append(
        f"<line x1='{x0:.2f}' y1='{y0:.2f}' x2='{x0:.2f}' y2='{y0 + plot_h:.2f}' "
        "stroke='#111827' stroke-width='1.3'/>"
    )
    svg.append(
        f"<line x1='{x0:.2f}' y1='{y0 + plot_h:.2f}' x2='{x0 + plot_w:.2f}' "
        f"y2='{y0 + plot_h:.2f}' stroke='#111827' stroke-width='1.3'/>"
    )

    for item in series:
        if not item.running_accuracy:
            continue
        plotted_values = item.running_accuracy[:max_episode]
        if not plotted_values:
            continue
        points = points_for_values(
            plotted_values, x0, y0, plot_w, plot_h, max_episode
        )
        svg.append(
            f"<polyline points='{points}' fill='none' stroke='{item.color}' "
            "stroke-width='3.2' stroke-linejoin='round' stroke-linecap='round'/>"
        )
        last_x = x0 + ((len(plotted_values) - 1) / max(1, max_episode - 1)) * plot_w
        last_y = y0 + (1.0 - plotted_values[-1]) * plot_h
        svg.append(
            f"<circle cx='{last_x:.2f}' cy='{last_y:.2f}' r='4.0' "
            f"fill='#ffffff' stroke='{item.color}' stroke-width='2.2'/>"
        )


def draw_unified_svg(
    series_by_env: dict[str, list[ResultSeries]],
    out_path: Path,
    expected_episodes: int,
    x_limit: int | None = None,
) -> None:
    loaded_max_episode = max(
        [expected_episodes]
        + [s.episodes for series in series_by_env.values() for s in series]
    )
    max_episode = x_limit or loaded_max_episode
    width = 1600
    height = 820
    top = 156
    plot_h = 500
    left_panel_x = 106
    panel_gap = 96
    plot_w = (width - left_panel_x * 2 - panel_gap) / 2
    right_panel_x = left_panel_x + plot_w + panel_gap

    svg: list[str] = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='#ffffff'/>",
        (
            f"<text x='106' y='48' font-family='{SVG_FONT}' font-size='24' "
            "font-weight='800' fill='#111827'>Online Accuracy by Episode</text>"
        ),
        (
            f"<text x='106' y='76' font-family='{SVG_FONT}' font-size='13' "
            f"fill='#4b5563'>Qwen3-14B no-thinking, expected {expected_episodes} episodes per environment; "
            "each panel shows four method lines.</text>"
        ),
    ]

    legend_x = 106
    legend_y = 116
    for i, (_, label, color) in enumerate(METHODS):
        x = legend_x + i * 282
        svg.append(
            f"<line x1='{x}' y1='{legend_y}' x2='{x + 40}' y2='{legend_y}' "
            f"stroke='{color}' stroke-width='4.5' stroke-linecap='round'/>"
        )
        svg.append(
            f"<text x='{x + 50}' y='{legend_y + 5}' font-family='{SVG_FONT}' "
            f"font-size='14' fill='#111827'>{svg_text(label)}</text>"
        )

    draw_panel(
        svg,
        "FrozenLake",
        series_by_env.get("frozen_lake", []),
        left_panel_x,
        top,
        plot_w,
        plot_h,
        max_episode,
    )
    draw_panel(
        svg,
        "Sokoban",
        series_by_env.get("sokoban", []),
        right_panel_x,
        top,
        plot_w,
        plot_h,
        max_episode,
    )

    svg.append(
        f"<text x='{width / 2:.2f}' y='{height - 48}' text-anchor='middle' "
        f"font-family='{SVG_FONT}' font-size='15' fill='#111827'>Episode</text>"
    )
    svg.append(
        "<text transform='translate(31 406) rotate(-90)' text-anchor='middle' "
        f"font-family='{SVG_FONT}' font-size='15' fill='#111827'>Accuracy</text>"
    )
    svg.append("</svg>")
    out_path.write_text("\n".join(svg), encoding="utf-8")


def draw_single_env_svg(
    env_label: str,
    series: list[ResultSeries],
    out_path: Path,
    expected_episodes: int,
    x_limit: int | None = None,
) -> None:
    loaded_max_episode = max([expected_episodes] + [s.episodes for s in series])
    max_episode = x_limit or loaded_max_episode
    width = 1180
    height = 700
    x0 = 96
    y0 = 116
    plot_w = 1010
    plot_h = 430
    svg = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='#ffffff'/>",
    ]
    draw_panel(svg, env_label, series, x0, y0, plot_w, plot_h, max_episode)
    legend_y = 596
    for i, item in enumerate(series):
        x = 96 + i * 260
        svg.append(
            f"<line x1='{x}' y1='{legend_y}' x2='{x + 34}' y2='{legend_y}' "
            f"stroke='{item.color}' stroke-width='4.5' stroke-linecap='round'/>"
        )
        svg.append(
            f"<text x='{x + 44}' y='{legend_y + 5}' font-family='{SVG_FONT}' "
            f"font-size='13' fill='#111827'>{svg_text(item.label)} ({pct(item.pass_rate)})</text>"
        )
    svg.append(
        f"<text x='{width / 2:.2f}' y='{height - 30}' text-anchor='middle' "
        f"font-family='{SVG_FONT}' font-size='15' fill='#111827'>Episode</text>"
    )
    svg.append("</svg>")
    out_path.write_text("\n".join(svg), encoding="utf-8")


def window_rate(rewards: list[float], start: int, end: int) -> float:
    window = rewards[start:end]
    if not window:
        return 0.0
    return sum(1 for reward in window if reward >= 1.0) / len(window)


def write_summary_csv(series: list[ResultSeries], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "env",
                "episodes",
                "successes",
                "pass_rate",
                "first20",
                "last20",
                "first_half",
                "second_half",
                "result_path",
            ],
        )
        writer.writeheader()
        for item in series:
            half = item.episodes // 2
            writer.writerow(
                {
                    "method": item.method,
                    "env": item.env,
                    "episodes": item.episodes,
                    "successes": item.successes,
                    "pass_rate": f"{item.pass_rate:.6f}",
                    "first20": f"{window_rate(item.rewards, 0, 20):.6f}",
                    "last20": f"{window_rate(item.rewards, max(0, item.episodes - 20), item.episodes):.6f}",
                    "first_half": f"{window_rate(item.rewards, 0, half):.6f}",
                    "second_half": f"{window_rate(item.rewards, half, item.episodes):.6f}",
                    "result_path": str(item.path),
                }
            )


def main() -> None:
    args = parse_args()
    root = Path(args.input_root).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    series_by_env: dict[str, list[ResultSeries]] = {env: [] for env, _ in ENVS}
    all_series: list[ResultSeries] = []
    missing: list[tuple[str, str]] = []

    for env, _ in ENVS:
        for method, _, _ in METHODS:
            result = find_best_result(
                root=root,
                method=method,
                env=env,
                expected_episodes=args.expected_episodes,
                prefer_substring=args.prefer_substring,
            )
            if result is None:
                missing.append((method, env))
                continue
            series_by_env[env].append(result)
            all_series.append(result)

    draw_unified_svg(
        series_by_env,
        out_dir / "online_accuracy_4methods_k80.svg",
        args.expected_episodes,
        args.x_limit,
    )
    for env, env_label in ENVS:
        draw_single_env_svg(
            env_label,
            series_by_env.get(env, []),
            out_dir / f"{env}_online_accuracy_4methods.svg",
            args.expected_episodes,
            args.x_limit,
        )
    write_summary_csv(all_series, out_dir / "online_4methods_summary.csv")

    print(f"Wrote: {out_dir}")
    for item in all_series:
        warning = "" if item.episodes >= args.expected_episodes else "  WARNING: fewer than expected"
        print(
            f"{item.env:12s} {item.method:30s} "
            f"{item.successes:3d}/{item.episodes:<3d} {pct(item.pass_rate):>7s}  "
            f"{item.path}{warning}"
        )
    if missing:
        print("\nMissing result files:")
        for method, env in missing:
            print(f"  {method} / {env}")


if __name__ == "__main__":
    main()
