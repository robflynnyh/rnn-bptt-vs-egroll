#!/usr/bin/env python3
"""Score the predeclared bounded EGGROLL screening runs."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "artifacts" / "eggroll_tuning" / "screening"
SCORE_STEPS = (1600, 1700, 1800, 1900, 2000)
CONTROL_NAME = "screen-zscore-s5e-3-lr3e-1-control"


def validation_row(entry: dict) -> dict:
    matches = [
        row
        for row in entry["grid"]
        if row["sequence_length"] == 16 and row["num_kv_pairs"] == 1
    ]
    if len(matches) != 1:
        raise ValueError("expected one length-16, one-pair validation row")
    return matches[0]


def score_run(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    by_step = {entry["step"]: validation_row(entry) for entry in result["validation_history"]}
    missing = [step for step in SCORE_STEPS if step not in by_step]
    if missing:
        raise ValueError(f"{path}: missing score steps {missing}")
    losses = [by_step[step]["loss"] for step in SCORE_STEPS]
    accuracies = [by_step[step]["accuracy"] for step in SCORE_STEPS]
    updates = result["update_history"][-500:]
    metric_names = (
        "parameter_update_rms",
        "update_to_parameter_rms_ratio",
        "raw_gradient_rms",
        "gradient_rms",
    )
    update_means = {
        name: sum(row["metrics"][name] for row in updates) / len(updates)
        for name in metric_names
    }
    config = result["config"]
    return {
        "name": path.parent.name,
        "update_rule": config["eggroll_update_rule"],
        "sigma": config["sigma"],
        "learning_rate": config["eggroll_learning_rate"],
        "elite_count": config["elite_count"],
        "elite_commit_scale": config["elite_commit_scale"],
        "score_steps": list(SCORE_STEPS),
        "validation_losses": losses,
        "validation_accuracies": accuracies,
        "screening_score": sum(losses) / len(losses),
        "final_validation_loss": losses[-1],
        "final_validation_accuracy": accuracies[-1],
        "last_500_update_means": update_means,
        "training_seconds": result["timing_seconds"]["training"],
    }


def main() -> None:
    rows = [
        score_run(path)
        for path in sorted(RESULT_ROOT.glob("*/metrics.json"))
    ]
    if not rows:
        raise SystemExit("no completed screening results found")
    controls = [row for row in rows if row["name"] == CONTROL_NAME]
    if len(controls) != 1:
        raise SystemExit("the completed results must contain exactly one control")
    control_score = controls[0]["screening_score"]
    for row in rows:
        row["relative_improvement"] = (
            control_score - row["screening_score"]
        ) / control_score
        row["promotion_eligible"] = row["relative_improvement"] >= 0.01
    rows.sort(key=lambda row: row["screening_score"])
    output = {
        "control": CONTROL_NAME,
        "control_score": control_score,
        "promotion_threshold": 0.01,
        "score_steps": list(SCORE_STEPS),
        "runs": rows,
    }
    output_path = RESULT_ROOT.parent / "screening_results.json"
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    for row in rows:
        print(
            f"{row['screening_score']:.6f} "
            f"{100 * row['relative_improvement']:+.2f}% {row['name']}"
        )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
