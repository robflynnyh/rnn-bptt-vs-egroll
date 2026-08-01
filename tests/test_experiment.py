from dataclasses import replace

import torch

from rnn_bptt_vs_eggroll.experiment import (
    CurriculumState,
    run_experiment,
    smoke_config,
    update_curriculum,
)


def test_curriculum_waits_for_both_models_and_consecutive_probes() -> None:
    config = replace(
        smoke_config(),
        curriculum_delays=(0, 4),
        curriculum_accuracy_threshold=0.75,
        curriculum_consecutive_probes=2,
    )
    state = CurriculumState()
    assert update_curriculum(
        state,
        {"bptt": 0.8, "eggroll": 0.7},
        config,
        generation=1,
    ) is None
    assert state.consecutive_passes == 0
    assert update_curriculum(
        state,
        {"bptt": 0.8, "eggroll": 0.8},
        config,
        generation=2,
    ) is None
    transition = update_curriculum(
        state,
        {"bptt": 0.9, "eggroll": 0.8},
        config,
        generation=3,
    )
    assert transition is not None
    assert state.current_max_delay(config) == 4


def test_smoke_experiment_writes_reproducible_outputs(tmp_path) -> None:
    config = replace(
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
        curriculum_consecutive_probes=1,
        curriculum_probe_examples=4,
    )
    results = run_experiment(tmp_path, device=torch.device("cpu"), config=config)
    assert results is not None
    assert results["model"]["initial_checksums"]["bptt"] == results["model"][
        "initial_checksums"
    ]["eggroll"]
    assert results["budgets"]["unique_training_sequences"] == 16
    assert results["budgets"]["eggroll_candidate_forward_sequences"] == 64
    assert len(results["test"]["grid"]) == 6
    transitions = results["curriculum"]["transitions"]
    assert len(transitions) == 1
    assert transitions[0]["generation"] == 1
    assert transitions[0]["from_max_delay"] == 0
    assert transitions[0]["to_max_delay"] == 4
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "bptt.pt").is_file()
    assert (tmp_path / "eggroll.pt").is_file()
