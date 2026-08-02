from dataclasses import replace
import sys

import pytest
import torch

from rnn_bptt_vs_eggroll.experiment import (
    CurriculumState,
    run_experiment,
    scheduled_eggroll_learning_rate,
    smoke_config,
    update_curriculum,
)


def test_curriculum_advances_after_one_passing_probe() -> None:
    config = replace(smoke_config(), curriculum_accuracy_threshold=0.75,)
    state = CurriculumState()
    assert update_curriculum(state, 0.7, config, generation=1) is None
    transition = update_curriculum(state, 0.8, config, generation=2,)
    assert transition is not None
    assert state.current_task(config) == (16, 2)


def test_eggroll_learning_rate_holds_then_cosine_decays() -> None:
    config = replace(
        smoke_config(),
        generations=100,
        eggroll_learning_rate=0.3,
        eggroll_learning_rate_final=0.01,
        eggroll_learning_rate_decay_start=20,
    )

    assert scheduled_eggroll_learning_rate(config, 1) == 0.3
    assert scheduled_eggroll_learning_rate(config, 20) == 0.3
    assert scheduled_eggroll_learning_rate(config, 60) == pytest.approx(0.155)
    assert scheduled_eggroll_learning_rate(config, 100) == pytest.approx(0.01)


def test_smoke_experiment_writes_reproducible_outputs(tmp_path) -> None:
    base_config = replace(
        smoke_config(seed=13),
        generations=2,
        evaluation_interval=1,
        evaluation_examples=8,
        test_examples=8,
        evaluation_batch_size=4,
        population_size=4,
        population_chunk_size=2,
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
        assert len(result["test"]["grid"]) == 2
        assert {
            row["sequence_length"] for row in result["validation_history"][0]["grid"]
        } == {8}
        assert {
            row["sequence_length"] for row in result["validation_history"][-1]["grid"]
        } == {8, 16,}
        transitions = result["curriculum"]["transitions"]
        assert len(transitions) == 1
        assert transitions[0]["generation"] == 1
        assert transitions[0]["from_sequence_length"] == 8
        assert transitions[0]["to_sequence_length"] == 16
        assert (output_dir / "metrics.json").is_file()
        assert (output_dir / "model.pt").is_file()

    assert (
        results["bptt"]["model"]["initial_checksum"]
        == results["eggroll"]["model"]["initial_checksum"]
    )


def test_wandb_tracks_full_single_method_run(tmp_path, monkeypatch) -> None:
    class FakeRun:
        def __init__(self) -> None:
            self.defined_metrics = []
            self.logged = []
            self.saved = []
            self.summary = {}
            self.finished = False

        def define_metric(self, *args, **kwargs) -> None:
            self.defined_metrics.append((args, kwargs))

        def log(self, metrics) -> None:
            self.logged.append(metrics)

        def save(self, path, **kwargs) -> None:
            self.saved.append((path, kwargs))

        def finish(self) -> None:
            self.finished = True

    class FakeWandb:
        def __init__(self) -> None:
            self.run = FakeRun()
            self.init_kwargs = None

        def init(self, **kwargs):
            self.init_kwargs = kwargs
            return self.run

    fake_wandb = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    config = replace(
        smoke_config(seed=17),
        method="bptt",
        generations=2,
        evaluation_interval=2,
        log_interval=1,
        evaluation_examples=4,
        test_examples=4,
        curriculum_probe_examples=4,
        wandb_enabled=True,
        wandb_run_name="test-bptt",
    )

    result = run_experiment(tmp_path, device=torch.device("cpu"), config=config)

    assert result is not None
    assert fake_wandb.init_kwargs["config"]["method"] == "bptt"
    logged_keys = {key for row in fake_wandb.run.logged for key in row}
    assert "train/batch_loss" in logged_keys
    assert "curriculum/frontier_accuracy" in logged_keys
    assert "validation_grid/seq_len_8/kv_pairs_1/accuracy" in logged_keys
    assert "test_grid/seq_len_8/kv_pairs_1/accuracy" in logged_keys
    assert sum("train/batch_loss" in row for row in fake_wandb.run.logged) == 2
    assert {path.rsplit("/", 1)[-1] for path, _ in fake_wandb.run.saved} == {
        "metrics.json",
        "model.pt",
    }
    assert fake_wandb.run.finished
