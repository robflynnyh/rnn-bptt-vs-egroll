import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / (
    "summarize_gentle_curriculum.py"
)
SPEC = importlib.util.spec_from_file_location("gentle_summary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_summary_uses_one_passing_probe_and_common_milestones(tmp_path) -> None:
    final_grid = [
        {
            "sequence_length": length,
            "num_kv_pairs": pairs,
            "accuracy": 0.25,
            "loss": 1.5,
        }
        for length, pairs in MODULE.COMMON_MILESTONES
    ]
    result = {
        "config": {
            "generations": 20_000,
            "curriculum_schedule": "gentle",
            "seed": 7,
        },
        "model": {"initial_checksum": "same-init"},
        "validation_history": [
            {
                "step": 100,
                "curriculum": {
                    "sequence_length_before_probe": 16,
                    "num_kv_pairs_before_probe": 1,
                    "frontier_accuracy": 0.89,
                },
            },
            {
                "step": 200,
                "curriculum": {
                    "sequence_length_before_probe": 16,
                    "num_kv_pairs_before_probe": 1,
                    "frontier_accuracy": 0.91,
                },
            },
            {
                "step": 300,
                "curriculum": {
                    "sequence_length_before_probe": 32,
                    "num_kv_pairs_before_probe": 2,
                    "frontier_accuracy": 0.9,
                },
            },
        ],
        "update_history": [
            {
                "generation": 19_600,
                "metrics": {
                    "mean_model_batch_loss": 1.0,
                    "parameter_update_rms": 0.2,
                    "update_to_parameter_rms_ratio": 0.3,
                },
            },
            {
                "generation": 20_000,
                "metrics": {
                    "mean_model_batch_loss": 2.0,
                    "parameter_update_rms": 0.4,
                    "update_to_parameter_rms_ratio": 0.5,
                },
            },
        ],
        "final_curriculum_probe": {"grid": final_grid},
        "timing_seconds": {"training": 10.0, "experiment": 12.0},
        "curriculum": {"transitions": [{"generation": 200}]},
    }
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(result), encoding="utf-8")

    summary = MODULE.summarize_run(path)

    assert summary["highest_common_milestone_mastered"] == (32, 2)
    assert summary["first_mastered_generation"] == {"16,1": 200, "32,2": 300}
    assert summary["last_500_update_means"] == pytest.approx({
        "mean_model_batch_loss": 1.5,
        "parameter_update_rms": 0.3,
        "update_to_parameter_rms_ratio": 0.4,
    })
