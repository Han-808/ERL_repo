from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ENV_LABELS = {
    "frozen_lake": "FrozenLake",
    "sokoban": "Sokoban",
}


def load_state_summary(csv_path: Path) -> dict[str, list[dict[str, float]]]:
    rows_by_env: dict[str, list[dict[str, float]]] = defaultdict(list)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"env", "state_x", "mean_sample_pass_rate", "std_sample_pass_rate"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            missing_cols = ", ".join(sorted(missing))
            raise ValueError(f"Missing required columns in {csv_path}: {missing_cols}")

        for row in reader:
            env = row["env"]
            rows_by_env[env].append(
                {
                    "state_x": float(row["state_x"]),
                    "mean_acc": float(row["mean_sample_pass_rate"]),
                    "std_acc": float(row["std_sample_pass_rate"]),
                }
            )

    for rows in rows_by_env.values():
        rows.sort(key=lambda item: item["state_x"])

    return rows_by_env


def plot_state_summary(csv_path: Path, output_path: Path) -> None:
    rows_by_env = load_state_summary(csv_path)
    envs = [env for env in ("sokoban", "frozen_lake") if env in rows_by_env]
    envs.extend(sorted(set(rows_by_env) - set(envs)))

    fig, axes = plt.subplots(
        len(envs),
        1,
        figsize=(10, 4 * len(envs)),
        sharex=True,
        constrained_layout=True,
    )
    if len(envs) == 1:
        axes = [axes]

    for ax, env in zip(axes, envs):
        rows = rows_by_env[env]
        states = [row["state_x"] for row in rows]
        mean_acc = [row["mean_acc"] for row in rows]
        std_acc = [row["std_acc"] for row in rows]

        ax.plot(states, mean_acc, marker="o", linewidth=2, label="mean_acc")
        ax.plot(states, std_acc, marker="s", linewidth=2, label="std_acc")
        ax.set_title(ENV_LABELS.get(env, env))
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, max(1.0, max(mean_acc + std_acc) * 1.1))
        ax.grid(True, alpha=0.25)
        ax.legend()

    axes[-1].set_xlabel("States")
    fig.suptitle("Mean and Std Accuracy by State", fontsize=14)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot mean_acc and std_acc against state_x for each environment."
    )
    parser.add_argument("--csv", type=Path, default=Path("state_summary.csv"))
    parser.add_argument("--out", type=Path, default=Path("plots/state_accuracy_by_env.png"))
    args = parser.parse_args()

    plot_state_summary(args.csv, args.out)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
