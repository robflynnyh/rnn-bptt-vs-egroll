from dataclasses import replace
import json
import sys

import pytest
import torch

from rnn_bptt_vs_eggroll.experiment import (
    DENSE_RECALL_CURRICULUM,
    DENSE_RECALL_FROM_TWO_CURRICULUM,
    GENTLE_CURRICULUM,
    REFERENCE_CURRICULUM,
    CurriculumState,
    ExperimentConfig,
    apply_curriculum_schedule,
    logical_query_span,
    run_experiment,
    scheduled_eggroll_learning_rate,
    smoke_config,
    update_curriculum,
)


def test_logical_query_span_counts_available_query_slots() -> None:
    assert logical_query_span(4, 1) == 1
    assert logical_query_span(6, 1) == 2
    assert logical_query_span(10, 1) == 4
    assert logical_query_span(32, 2) == 14

    with pytest.raises(ValueError, match="positive integer query span"):
        logical_query_span(3, 1)


def test_named_curriculum_schedules_are_exact() -> None:
    reference = ExperimentConfig()
    assert reference.curriculum_schedule == "reference"
    assert tuple(zip(
        reference.curriculum_sequence_lengths,
        reference.curriculum_num_kv_pairs,
    )) == REFERENCE_CURRICULUM

    gentle = apply_curriculum_schedule(reference, "gentle")
    assert gentle.curriculum_schedule == "gentle"
    assert tuple(zip(
        gentle.curriculum_sequence_lengths,
        gentle.curriculum_num_kv_pairs,
    )) == GENTLE_CURRICULUM
    assert GENTLE_CURRICULUM[:7] == (
        (4, 1), (6, 1), (8, 1), (10, 1), (12, 1), (14, 1), (16, 1),
    )
    assert GENTLE_CURRICULUM[7:] == REFERENCE_CURRICULUM[1:]

    dense = apply_curriculum_schedule(reference, "dense-recall")
    assert dense.task == "dense-recall"
    assert dense.curriculum_schedule == "dense-recall"
    assert len(DENSE_RECALL_CURRICULUM) == 1_024
    assert DENSE_RECALL_CURRICULUM[0] == (4, 1)
    assert DENSE_RECALL_CURRICULUM[-1] == (2_050, 1_024)

    dense_from_two = apply_curriculum_schedule(reference, "dense-recall-from-2")
    assert dense_from_two.task == "dense-recall"
    assert tuple(zip(
        dense_from_two.curriculum_sequence_lengths,
        dense_from_two.curriculum_num_kv_pairs,
    )) == DENSE_RECALL_FROM_TWO_CURRICULUM
    assert DENSE_RECALL_FROM_TWO_CURRICULUM[0] == (6, 2)
    assert DENSE_RECALL_FROM_TWO_CURRICULUM[-1] == (2_050, 1_024)


def test_curriculum_advances_after_one_passing_probe() -> None:
    config = replace(smoke_config(), curriculum_accuracy_threshold=0.75,)
    state = CurriculumState()
    assert update_curriculum(state, 0.7, config, generation=1) is None
    transition = update_curriculum(state, 0.8, config, generation=2,)
    assert transition is not None
    assert state.current_task(config) == (16, 2)
    assert transition["from_logical_query_span"] == 3
    assert transition["to_logical_query_span"] == 6


def test_dense_recall_smoke_advances_by_one_pair(tmp_path) -> None:
    config = replace(
        smoke_config(seed=41),
        method="bptt",
        task="dense-recall",
        curriculum_schedule="custom",
        curriculum_sequence_lengths=(4, 6),
        curriculum_num_kv_pairs=(1, 2),
        curriculum_accuracy_threshold=0.0,
        curriculum_frontier_probability=1.0,
        curriculum_probe_examples=4,
        evaluation_frontier_only=True,
        evaluation_interval=1,
        evaluation_examples=4,
        test_examples=4,
        evaluation_batch_size=4,
        generations=2,
    )

    result = run_experiment(tmp_path, device=torch.device("cpu"), config=config)

    assert result is not None
    transition = result["curriculum"]["transitions"][0]
    assert transition["from_sequence_length"] == 4
    assert transition["to_sequence_length"] == 6
    assert transition["from_task_size"] == 1
    assert transition["to_task_size"] == 2
    assert transition["from_logical_query_span"] is None
    assert transition["to_logical_query_span"] is None
    assert result["update_history"][0]["sampled_sequence_length"] == 4
    assert result["update_history"][0]["sampled_task_size"] == 1
    assert all(
        row["logical_query_span"] is None
        for entry in result["validation_history"]
        for row in entry["grid"]
    )


def test_product_vocabulary_dense_recall_uses_exact_component_accuracy(tmp_path) -> None:
    config = replace(
        smoke_config(seed=47),
        method="bptt",
        task="dense-recall",
        architecture="lstm",
        curriculum_schedule="custom",
        curriculum_sequence_lengths=(4, 6),
        curriculum_num_kv_pairs=(1, 2),
        curriculum_accuracy_threshold=0.0,
        curriculum_frontier_probability=1.0,
        curriculum_probe_examples=4,
        evaluation_frontier_only=True,
        evaluation_interval=1,
        evaluation_examples=4,
        test_examples=4,
        evaluation_batch_size=4,
        generations=2,
        vocab_size=32,
        hidden_size=8,
        tie_input_output=True,
        product_vocab_codebooks=2,
        product_vocab_codebook_size=4,
    )

    result = run_experiment(tmp_path, device=torch.device("cpu"), config=config)

    assert result is not None
    assert result["model"]["architecture"] == "single_layer_product_vocab_lstm"
    assert result["model"]["product_vocab_codebooks"] == 2
    assert "batch_component_accuracy" in result["update_history"][-1]["metrics"]
    test_row = result["test"]["grid"][0]
    assert 0 <= test_row["accuracy"] <= test_row["component_accuracy"] <= 1


def test_dense_recall_stops_after_stage_update_budget(tmp_path) -> None:
    config = replace(
        smoke_config(seed=43),
        method="bptt",
        task="dense-recall",
        architecture="lstm",
        curriculum_schedule="custom",
        curriculum_sequence_lengths=(6,),
        curriculum_num_kv_pairs=(2,),
        curriculum_accuracy_threshold=1.0,
        curriculum_max_updates_per_stage=1,
        dense_recall_coupled_vocab=True,
        curriculum_frontier_probability=1.0,
        curriculum_probe_examples=128,
        evaluation_frontier_only=True,
        evaluation_interval=1,
        evaluation_examples=128,
        test_examples=4,
        evaluation_batch_size=128,
        generations=10,
        checkpoint_interval=100,
    )

    result = run_experiment(tmp_path, device=torch.device("cpu"), config=config)

    assert result is not None
    assert result["budgets"]["completed_generations"] == 1
    assert result["stopping"] == {
        "status": "stage_budget_exhausted",
        "generation": 1,
        "stage": 0,
        "num_kv_pairs": 2,
        "input_sequence_length": 5,
        "full_sequence_length": 6,
        "updates_at_stage": 1,
        "latest_accuracy": 0.0,
        "best_accuracy": 0.0,
        "accuracy_threshold": 1.0,
    }
    assert result["terminal_checkpoint"].endswith("generation_0000000001.pt")
    assert (tmp_path / "checkpoints" / "generation_0000000001.pt").is_file()


def test_eggroll_learning_rate_holds_then_cosine_decays() -> None:
    config = replace(
        smoke_config(),
        generations=100,
        eggroll_learning_rate=0.3,
        eggroll_learning_rate_final=0.01,
        eggroll_learning_rate_decay_start=20,
        eggroll_learning_rate_decay_end=100,
    )

    assert scheduled_eggroll_learning_rate(config, 1) == 0.3
    assert scheduled_eggroll_learning_rate(config, 20) == 0.3
    assert scheduled_eggroll_learning_rate(config, 60) == pytest.approx(0.155)
    assert scheduled_eggroll_learning_rate(config, 100) == pytest.approx(0.01)


def test_lstm_is_available_for_both_optimizers() -> None:
    lstm = replace(smoke_config(), method="bptt", architecture="lstm")
    assert lstm.architecture == "lstm"
    eggroll = replace(smoke_config(), method="eggroll", architecture="lstm")
    assert eggroll.architecture == "lstm"


def test_eggroll_learning_rate_holds_at_floor_after_decay() -> None:
    config = replace(
        smoke_config(),
        generations=200,
        eggroll_learning_rate=0.3,
        eggroll_learning_rate_final=0.01,
        eggroll_learning_rate_decay_start=0,
        eggroll_learning_rate_decay_end=100,
    )

    assert scheduled_eggroll_learning_rate(config, 50) == pytest.approx(0.155)
    assert scheduled_eggroll_learning_rate(config, 100) == pytest.approx(0.01)
    assert scheduled_eggroll_learning_rate(config, 200) == pytest.approx(0.01)


def test_short_screen_can_use_prefix_of_long_run_schedule() -> None:
    config = replace(
        smoke_config(),
        generations=20,
        eggroll_learning_rate=0.3,
        eggroll_learning_rate_final=0.01,
        eggroll_learning_rate_decay_start=0,
        eggroll_learning_rate_decay_end=100_000,
    )

    assert scheduled_eggroll_learning_rate(config, 20) == pytest.approx(
        0.2999999714,
    )


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
        final_full_curriculum_probe=True,
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
        assert result["final_curriculum_probe"]["enabled"]
        assert len(result["final_curriculum_probe"]["grid"]) == 2
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
        assert transitions[0]["from_logical_query_span"] == 3
        assert transitions[0]["to_sequence_length"] == 16
        assert transitions[0]["to_logical_query_span"] == 6
        assert result["curriculum"]["last_trained_logical_query_span"] == 6
        assert result["curriculum"]["next_logical_query_span"] == 6
        assert all(
            row["logical_query_span"]
            == logical_query_span(row["sequence_length"], row["num_kv_pairs"])
            for entry in result["validation_history"]
            for row in entry["grid"]
        )
        assert (output_dir / "metrics.json").is_file()
        assert (output_dir / "model.pt").is_file()

    assert (
        results["bptt"]["model"]["initial_checksum"]
        == results["eggroll"]["model"]["initial_checksum"]
    )


def test_elite_experiment_logs_selection_and_actual_update(tmp_path) -> None:
    config = replace(
        smoke_config(seed=19),
        method="eggroll",
        generations=1,
        evaluation_interval=1,
        evaluation_examples=4,
        test_examples=4,
        curriculum_probe_examples=4,
        population_size=4,
        population_chunk_size=2,
        eggroll_update_rule="elite-centroid",
        elite_count=1,
        elite_commit_scale=0.2,
    )

    result = run_experiment(tmp_path, device=torch.device("cpu"), config=config)

    assert result is not None
    metrics = result["update_history"][0]["metrics"]
    assert metrics["update_rule_standardized"] == 0
    assert metrics["update_rule_elite_centroid"] == 1
    assert metrics["selected_elite_count"] == 1
    assert metrics["elite_fraction"] == 0.5
    assert 0 <= metrics["elite_positive_fraction"] <= 1
    assert metrics["parameter_update_rms"] > 0
    assert metrics["update_to_parameter_rms_ratio"] > 0


def test_frontier_only_evaluation_omits_earlier_stages(tmp_path) -> None:
    config = replace(
        smoke_config(seed=21),
        method="bptt",
        generations=1,
        curriculum_enabled=False,
        evaluation_frontier_only=True,
        evaluation_interval=1,
        evaluation_examples=4,
        curriculum_probe_examples=4,
        test_examples=4,
        evaluation_batch_size=4,
        final_full_curriculum_probe=True,
    )

    result = run_experiment(tmp_path, device=torch.device("cpu"), config=config)

    assert result is not None
    assert all(
        [row["sequence_length"] for row in entry["grid"]] == [16]
        for entry in result["validation_history"]
    )
    assert [row["sequence_length"] for row in result["test"]["grid"]] == [16]
    assert [
        row["sequence_length"] for row in result["final_curriculum_probe"]["grid"]
    ] == [16]


def test_adaptive_mutation_scales_are_updated_and_recorded(tmp_path) -> None:
    config = replace(
        smoke_config(seed=23),
        method="eggroll",
        generations=2,
        evaluation_interval=2,
        evaluation_examples=4,
        test_examples=4,
        curriculum_probe_examples=4,
        population_size=8,
        population_chunk_size=4,
        adaptive_mutation_scales=True,
        mutation_scale_learning_rate=2.0,
    )

    result = run_experiment(tmp_path, device=torch.device("cpu"), config=config)

    assert result is not None
    metrics = result["update_history"][-1]["metrics"]
    assert metrics["adaptive_mutation_scales"] == 1
    scales = result["final_mutation_scales"]
    assert set(scales) == {
        "input_weight", "recurrent_weight", "hidden_bias", "output_weight",
        "output_bias",
    }
    assert any(scale != pytest.approx(1.0) for scale in scales.values())
    assert all(
        f"mutation_scale/{name}" in metrics
        and f"mutation_scale_gradient/{name}" in metrics
        for name in scales
    )


def test_adaptive_mutation_scales_reject_elite_update() -> None:
    with pytest.raises(ValueError, match="require standardized"):
        replace(
            smoke_config(),
            adaptive_mutation_scales=True,
            eggroll_update_rule="elite-centroid",
        )


def test_resume_and_bootstrap_options_are_validated() -> None:
    with pytest.raises(ValueError, match="set together"):
        replace(smoke_config(), bootstrap_model_path="model.pt")
    with pytest.raises(ValueError, match="mutually exclusive"):
        replace(
            smoke_config(),
            bootstrap_model_path="model.pt",
            bootstrap_metrics_path="metrics.json",
            resume_checkpoint="checkpoint.pt",
        )
    with pytest.raises(ValueError, match="wandb_run_id"):
        replace(smoke_config(), wandb_resume="allow")
    with pytest.raises(ValueError, match="positive"):
        replace(smoke_config(), checkpoint_interval=0)


def test_full_state_checkpoint_resume_matches_uninterrupted_training(tmp_path) -> None:
    config = replace(
        smoke_config(seed=29),
        method="bptt",
        generations=4,
        curriculum_enabled=False,
        evaluation_interval=2,
        evaluation_examples=4,
        test_examples=4,
        evaluation_batch_size=4,
        checkpoint_interval=2,
    )
    uninterrupted_dir = tmp_path / "uninterrupted"
    interrupted_dir = tmp_path / "interrupted"
    resumed_dir = tmp_path / "resumed"
    uninterrupted = run_experiment(
        uninterrupted_dir, device=torch.device("cpu"), config=config,
    )
    run_experiment(
        interrupted_dir,
        device=torch.device("cpu"),
        config=replace(config, generations=2),
    )
    checkpoint = interrupted_dir / "checkpoints" / "generation_0000000002.pt"
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_payload["config_signature"]["generations"] = 2
    checkpoint_payload["config_signature"].pop("product_vocab_codebooks")
    checkpoint_payload["config_signature"].pop("product_vocab_codebook_size")
    torch.save(checkpoint_payload, checkpoint)

    resumed = run_experiment(
        resumed_dir,
        device=torch.device("cpu"),
        config=replace(config, resume_checkpoint=str(checkpoint)),
    )

    assert uninterrupted is not None
    assert resumed is not None
    assert resumed["start"] == {
        "kind": "checkpoint",
        "generation": 2,
        "checkpoint_path": str(checkpoint),
        "rng_continuity": True,
    }
    uninterrupted_state = torch.load(
        uninterrupted_dir / "model.pt", map_location="cpu", weights_only=True,
    )
    resumed_state = torch.load(
        resumed_dir / "model.pt", map_location="cpu", weights_only=True,
    )
    assert uninterrupted_state.keys() == resumed_state.keys()
    assert all(
        torch.equal(uninterrupted_state[name], resumed_state[name])
        for name in uninterrupted_state
    )
    assert not list((resumed_dir / "checkpoints").glob("*.tmp"))


def test_completed_run_can_bootstrap_a_longer_target(tmp_path) -> None:
    parent_config = replace(
        smoke_config(seed=31),
        method="bptt",
        generations=2,
        evaluation_interval=1,
        evaluation_examples=4,
        test_examples=4,
        evaluation_batch_size=4,
        curriculum_probe_examples=4,
        curriculum_accuracy_threshold=0.0,
    )
    parent_dir = tmp_path / "parent"
    run_experiment(parent_dir, device=torch.device("cpu"), config=parent_config)
    metrics_path = parent_dir / "metrics.json"
    legacy_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    for name in (
        "adaptive_mutation_scales",
        "mutation_scale_learning_rate",
        "mutation_scale_min",
        "mutation_scale_max",
    ):
        legacy_metrics["config"].pop(name)
    metrics_path.write_text(json.dumps(legacy_metrics), encoding="utf-8")
    continuation_config = replace(
        parent_config,
        generations=3,
        bptt_learning_rate=1e-3,
        curriculum_frontier_probability=1.0,
        bootstrap_model_path=str(parent_dir / "model.pt"),
        bootstrap_metrics_path=str(metrics_path),
    )
    with pytest.raises(ValueError, match="bootstrap configuration mismatch"):
        run_experiment(
            tmp_path / "rejected",
            device=torch.device("cpu"),
            config=continuation_config,
        )
    with pytest.raises(ValueError, match="bootstrap configuration mismatch"):
        run_experiment(
            tmp_path / "optimizer-only-override",
            device=torch.device("cpu"),
            config=replace(
                continuation_config,
                bootstrap_allow_optimizer_override=True,
            ),
        )
    continuation = run_experiment(
        tmp_path / "continuation",
        device=torch.device("cpu"),
        config=replace(
            continuation_config,
            bootstrap_allow_optimizer_override=True,
            bootstrap_allow_curriculum_sampling_override=True,
        ),
    )

    assert continuation is not None
    assert continuation["start"]["kind"] == "bootstrap"
    assert continuation["start"]["generation"] == 2
    assert continuation["start"]["rng_continuity"] is False
    assert continuation["update_history"][0]["generation"] == 3
    assert continuation["budgets"]["continuation_training_sequences"] == 8


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
        generations=3,
        evaluation_interval=3,
        log_interval=3,
        wandb_log_interval=1,
        evaluation_examples=4,
        test_examples=4,
        curriculum_probe_examples=4,
        final_full_curriculum_probe=True,
        wandb_enabled=True,
        wandb_run_name="test-bptt",
        wandb_run_id="resume-me",
        wandb_resume="allow",
    )

    result = run_experiment(tmp_path, device=torch.device("cpu"), config=config)

    assert result is not None
    assert fake_wandb.init_kwargs["config"]["method"] == "bptt"
    assert fake_wandb.init_kwargs["id"] == "resume-me"
    assert fake_wandb.init_kwargs["resume"] == "allow"
    logged_keys = {key for row in fake_wandb.run.logged for key in row}
    assert "train/batch_loss" in logged_keys
    assert "curriculum/frontier_accuracy" in logged_keys
    assert "curriculum/logical_query_span" in logged_keys
    assert "train/sampled_logical_query_span" in logged_keys
    assert "train/curriculum_logical_query_span" in logged_keys
    assert "validation_grid/seq_len_8/kv_pairs_1/accuracy" in logged_keys
    assert "validation_grid/seq_len_8/kv_pairs_1/logical_query_span" in logged_keys
    assert "final_curriculum_probe/seq_len_8/kv_pairs_1/accuracy" in logged_keys
    assert "test_grid/seq_len_8/kv_pairs_1/accuracy" in logged_keys
    assert sum("train/batch_loss" in row for row in fake_wandb.run.logged) == 3
    assert len(result["update_history"]) == 2
    assert {path.rsplit("/", 1)[-1] for path, _ in fake_wandb.run.saved} == {
        "metrics.json",
        "model.pt",
    }
    assert fake_wandb.run.finished
