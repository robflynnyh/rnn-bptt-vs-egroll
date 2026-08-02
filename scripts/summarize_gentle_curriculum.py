#!/usr/bin/env python3
"""Summarize the predeclared gentle-curriculum comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMMON_MILESTONES = (
    (16, 1),
    (32, 2),
    (64, 4),
    (128, 8),
    (256, 16),
    (512, 32),
    (1_024, 64),
)
THRESHOLD = 0.9


def task_from_curriculum(entry: dict) -> tuple[int, int]:
    curriculum = entry["curriculum"]
    return (
        curriculum["sequence_length_before_probe"],
        curriculum["num_kv_pairs_before_probe"],
    )


def summarize_run(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    config = result["config"]
    if config["generations"] != 20_000:
        raise ValueError(f"{path}: expected exactly 20,000 generations")

    first_mastered: dict[tuple[int, int], int] = {}
    reached: set[tuple[int, int]] = set()
    for entry in result["validation_history"]:
        task = task_from_curriculum(entry)
        if task in COMMON_MILESTONES:
            reached.add(task)
            accuracy = entry["curriculum"].get("frontier_accuracy")
            if accuracy is not None and accuracy >= THRESHOLD:
                first_mastered.setdefault(task, entry["step"])

    mastered = [task for task in COMMON_MILESTONES if task in first_mastered]
    final_generation = config["generations"]
    late_updates = [
        row for row in result["update_history"]
        if row["generation"] > final_generation - 500
    ]
    diagnostic_names = (
        "mean_model_batch_loss",
        "parameter_update_rms",
        "update_to_parameter_rms_ratio",
    )
    final_probe = {
        (row["sequence_length"], row["num_kv_pairs"]): row
        for row in result["final_curriculum_probe"]["grid"]
    }
    return {
        "schedule": config["curriculum_schedule"],
        "seed": config["seed"],
        "initial_checksum": result["model"]["initial_checksum"],
        "highest_common_milestone_mastered": mastered[-1] if mastered else None,
        "highest_common_milestone_index": (
            COMMON_MILESTONES.index(mastered[-1]) if mastered else -1
        ),
        "first_mastered_generation": {
            f"{length},{pairs}": generation
            for (length, pairs), generation in first_mastered.items()
        },
        "common_milestones_reached": [
            list(task) for task in COMMON_MILESTONES if task in reached
        ],
        "final_probe": {
            f"{length},{pairs}": {
                "accuracy": final_probe[(length, pairs)]["accuracy"],
                "loss": final_probe[(length, pairs)]["loss"],
            }
            for length, pairs in COMMON_MILESTONES
        },
        "training_seconds": result["timing_seconds"]["training"],
        "experiment_seconds": result["timing_seconds"]["experiment"],
        "last_500_update_means": {
            name: sum(row["metrics"][name] for row in late_updates)
            / len(late_updates)
            for name in diagnostic_names
        },
        "transition_count": len(result["curriculum"]["transitions"]),
        "transitions": result["curriculum"]["transitions"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(7, 8), required=True)
    args = parser.parse_args()
    output_root = ROOT / "artifacts" / "gentle_curriculum" / f"seed-{args.seed}"
    runs = {
        schedule: summarize_run(output_root / schedule / "metrics.json")
        for schedule in ("reference", "gentle")
    }
    if runs["reference"]["initial_checksum"] != runs["gentle"]["initial_checksum"]:
        raise ValueError("matched schedules did not start from the same model")
    gentle_wins = (
        runs["gentle"]["highest_common_milestone_index"]
        > runs["reference"]["highest_common_milestone_index"]
    )
    summary = {
        "seed": args.seed,
        "threshold": THRESHOLD,
        "common_milestones": [list(task) for task in COMMON_MILESTONES],
        "strict_higher_milestone_for_gentle": gentle_wins,
        "runs": runs,
    }
    output_path = output_root / "summary.json"
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    for schedule, row in runs.items():
        print(
            f"{schedule}: highest={row['highest_common_milestone_mastered']} "
            f"runtime={row['training_seconds']:.1f}s"
        )
    print(f"strict_higher_milestone_for_gentle={gentle_wins}")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
