from dataclasses import replace

import torch

from rnn_bptt_vs_eggroll.experiment import (
    CurriculumState,
    run_experiment,
    smoke_config,
    update_curriculum,
)


def test_curriculum_advances_after_one_passing_probe() -> None:
    config = replace(
        smoke_config(), curriculum_delays=(0, 4), curriculum_accuracy_threshold=0.75,
    )
    state = CurriculumState()
    assert update_curriculum(state, 0.7, config, generation=1) is None
    transition = update_curriculum(state, 0.8, config, generation=2,)
    assert transition is not None
    assert state.current_max_delay(config) == 4


def test_smoke_experiment_writes_reproducible_outputs(tmp_path) -> None:
    base_config = replace(
        smoke_config(seed=13),
        generations=2,
        evaluation_interval=1,
        evaluation_examples=8,
        test_examples=8,
        evaluation_batch_size=4,
        evaluation_pairs=(2,),
        evaluation_delays=(0, 4, 8),
        population_size=4,
        population_chunk_size=4,
        curriculum_delays=(0, 4),
        curriculum_accuracy_threshold=0.0,
        curriculum_probe_examples=4,
    )
    results = {}
    for method in ("bptt", "eggroll"):
        output_dir = tmp_path / method
        result = run_experiment(
            output_dir,
            device=torch.device("cpu"),
            config=replace(base_config, method=method),
        )
        assert result is not None
        results[method] = result
        assert result["method"] == method
        assert result["budgets"]["unique_training_sequences"] == 16
        expected_candidate_forwards = 64 if method == "eggroll" else 0
        assert (
            result["budgets"]["eggroll_candidate_forward_sequences"]
            == expected_candidate_forwards
        )
        assert len(result["test"]["grid"]) == 3
        transitions = result["curriculum"]["transitions"]
        assert len(transitions) == 1
        assert transitions[0]["generation"] == 1
        assert transitions[0]["from_max_delay"] == 0
        assert transitions[0]["to_max_delay"] == 4
        assert (output_dir / "metrics.json").is_file()
        assert (output_dir / "model.pt").is_file()

    assert (
        results["bptt"]["model"]["initial_checksum"]
        == results["eggroll"]["model"]["initial_checksum"]
    )
