"""Train one RNN with BPTT or forward-only EGGROLL on associative recall."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

from .eggroll import (
    assign_maximization_gradients,
    estimate_reward_gradients,
    evaluate_population,
    gradient_rms,
    sample_antithetic_noise,
    shape_fitness,
)
from .model import VanillaRNN
from .task import IGNORE_INDEX, MQARConfig, sample_batch


@dataclass(frozen=True)
class ExperimentConfig:
    method: str = "bptt"
    seed: int = 7
    generations: int = 3_000
    batch_size: int = 256
    curriculum_enabled: bool = True
    curriculum_sequence_lengths: tuple[int, ...] = (
        16,
        32,
        64,
        128,
        256,
        512,
        1_024,
    )
    curriculum_num_kv_pairs: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    curriculum_accuracy_threshold: float = 0.9
    curriculum_frontier_probability: float = 0.5
    curriculum_probe_examples: int = 512
    evaluation_examples: int = 1_024
    test_examples: int = 4_096
    evaluation_batch_size: int = 256
    evaluation_interval: int = 250
    vocab_size: int = 8_192
    query_power_a: float = 0.01
    random_non_queries: bool = False
    hidden_size: int = 64
    recurrent_radius: float = 0.9
    population_size: int = 16_384
    population_chunk_size: int = 2
    perturbation_rank: int = 1
    sigma: float = 0.005
    sigma_decay: float = 1.0
    fitness_shaping: str = "zscore"
    eggroll_learning_rate: float = 0.3
    eggroll_learning_rate_decay: float = 1.0
    eggroll_weight_decay: float = 1e-3
    eggroll_momentum: float = 0.0
    bptt_learning_rate: float = 3e-3
    bptt_learning_rate_decay: float = 1.0
    bptt_weight_decay: float = 1e-3
    bptt_gradient_clip: float = 1.0
    wandb_enabled: bool = False
    wandb_project: str = "rnn-bptt-vs-eggroll"
    wandb_entity: str | None = "wobrob101"
    wandb_run_name: str | None = None
    wandb_group: str | None = None
    log_progress: bool = False

    def __post_init__(self) -> None:
        if self.method not in {"bptt", "eggroll"}:
            raise ValueError("method must be 'bptt' or 'eggroll'")
        if self.generations < 1 or self.batch_size < 1:
            raise ValueError("generations and batch_size must be positive")
        if not self.curriculum_sequence_lengths:
            raise ValueError("curriculum must contain at least one stage")
        if len(self.curriculum_sequence_lengths) != len(self.curriculum_num_kv_pairs):
            raise ValueError("curriculum lengths and pair counts must align")
        if tuple(sorted(set(self.curriculum_sequence_lengths))) != (
            self.curriculum_sequence_lengths
        ):
            raise ValueError(
                "curriculum sequence lengths must be unique and increasing"
            )
        if any(length < 4 or length % 2 for length in self.curriculum_sequence_lengths):
            raise ValueError("curriculum sequence lengths must be positive and even")
        if any(pairs < 1 for pairs in self.curriculum_num_kv_pairs):
            raise ValueError("curriculum pair counts must be positive")
        if any(
            4 * pairs > length
            for length, pairs in zip(
                self.curriculum_sequence_lengths, self.curriculum_num_kv_pairs,
            )
        ):
            raise ValueError("every curriculum stage needs context and query slots")
        if not 0 <= self.curriculum_accuracy_threshold <= 1:
            raise ValueError("curriculum_accuracy_threshold must be in [0, 1]")
        if not 0 <= self.curriculum_frontier_probability <= 1:
            raise ValueError("curriculum_frontier_probability must be in [0, 1]")
        if self.curriculum_probe_examples < 1:
            raise ValueError("curriculum_probe_examples must be positive")
        if self.vocab_size <= max(self.curriculum_sequence_lengths):
            raise ValueError(
                "Zoology MQAR requires vocab_size above every sequence length"
            )
        if max(self.curriculum_num_kv_pairs) >= self.vocab_size // 2:
            raise ValueError("pair count exceeds the available key vocabulary")
        if self.query_power_a <= 0:
            raise ValueError("query_power_a must be positive")
        counts = (
            self.evaluation_examples,
            self.test_examples,
            self.evaluation_batch_size,
            self.evaluation_interval,
            self.hidden_size,
        )
        if any(value < 1 for value in counts):
            raise ValueError("evaluation and model dimensions must be positive")
        if self.recurrent_radius <= 0:
            raise ValueError("recurrent_radius must be positive")
        if self.population_size < 2 or self.population_size % 2:
            raise ValueError("population_size must be positive and even")
        if self.population_chunk_size < 2 or self.population_chunk_size % 2:
            raise ValueError("population_chunk_size must be positive and even")
        if self.perturbation_rank < 1:
            raise ValueError("perturbation_rank must be positive")
        if self.sigma <= 0 or not 0 < self.sigma_decay <= 1:
            raise ValueError("sigma settings are invalid")
        if self.fitness_shaping not in {"zscore", "centered-rank", "centered"}:
            raise ValueError("unknown fitness shaping")
        rates = (self.eggroll_learning_rate, self.bptt_learning_rate)
        if any(value <= 0 for value in rates):
            raise ValueError("learning rates must be positive")
        decays = (
            self.eggroll_learning_rate_decay,
            self.bptt_learning_rate_decay,
        )
        if any(not 0 < value <= 1 for value in decays):
            raise ValueError("learning-rate decays must be in (0, 1]")
        if min(self.eggroll_weight_decay, self.bptt_weight_decay) < 0:
            raise ValueError("weight decay must be non-negative")
        if not 0 <= self.eggroll_momentum < 1:
            raise ValueError("eggroll_momentum must be in [0, 1)")
        if self.bptt_gradient_clip <= 0:
            raise ValueError("bptt_gradient_clip must be positive")
        if not self.wandb_project:
            raise ValueError("wandb_project must be non-empty")


def smoke_config(seed: int = 7) -> ExperimentConfig:
    """A structural integration check, deliberately too small for conclusions."""

    return ExperimentConfig(
        seed=seed,
        generations=4,
        batch_size=8,
        curriculum_sequence_lengths=(8, 16),
        curriculum_num_kv_pairs=(1, 2),
        curriculum_probe_examples=16,
        evaluation_examples=32,
        test_examples=32,
        evaluation_batch_size=16,
        evaluation_interval=2,
        vocab_size=32,
        hidden_size=8,
        population_size=16,
        population_chunk_size=2,
    )


@dataclass
class CurriculumState:
    """Mutable state shared across workers while configuration stays immutable."""

    stage: int = 0
    transitions: list[dict[str, Any]] = field(default_factory=list)

    def current_stage(self, config: ExperimentConfig) -> int:
        if not config.curriculum_enabled:
            return len(config.curriculum_sequence_lengths) - 1
        return self.stage

    def current_task(self, config: ExperimentConfig) -> tuple[int, int]:
        stage = self.current_stage(config)
        return (
            config.curriculum_sequence_lengths[stage],
            config.curriculum_num_kv_pairs[stage],
        )


def _world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _is_primary() -> bool:
    return _rank() == 0


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _broadcast_model(model: VanillaRNN) -> None:
    if not dist.is_initialized():
        return
    for value in model.state_dict().values():
        dist.broadcast(value, src=0)


def _broadcast_curriculum_state(state: CurriculumState, device: torch.device,) -> None:
    if not dist.is_initialized():
        return
    values = torch.tensor([state.stage], device=device, dtype=torch.long)
    dist.broadcast(values, src=0)
    state.stage = int(values.item())


def _gather_equal_shards(values: Tensor) -> Tensor:
    if not dist.is_initialized():
        return values
    gathered = [torch.empty_like(values) for _ in range(_world_size())]
    dist.all_gather(gathered, values.contiguous())
    return torch.cat(gathered)


def _average_gradients(gradients: dict[str, Tensor]) -> None:
    if not dist.is_initialized():
        return
    for value in gradients.values():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value.div_(_world_size())


def _state_checksum(model: VanillaRNN) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("ascii"))
        raw_bytes = (
            value.detach().cpu().contiguous().view(torch.uint8).flatten().tolist()
        )
        digest.update(bytes(raw_bytes))
    return digest.hexdigest()


def _initialize_wandb(config: ExperimentConfig, output_dir: Path,) -> Any | None:
    if not config.wandb_enabled or not _is_primary():
        return None
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("W&B tracking requires the 'wandb' package") from error
    run = wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.wandb_run_name or output_dir.name,
        group=config.wandb_group,
        config=asdict(config),
    )
    run.define_metric("generation")
    run.define_metric("*", step_metric="generation")
    return run


def _validation_wandb_metrics(entry: dict[str, Any], method: str,) -> dict[str, float]:
    metrics: dict[str, float] = {
        "generation": float(entry["step"]),
        "model/parameter_l2_norm": entry["parameter_l2_norm"],
    }
    for name, value in entry["summary"][method].items():
        if value is not None:
            metrics[f"validation/{name}"] = value
    curriculum = entry["curriculum"]
    if curriculum["enabled"]:
        metrics.update(
            {
                "curriculum/stage": float(curriculum["stage_after_probe"]),
                "curriculum/sequence_length": float(
                    curriculum["sequence_length_after_probe"]
                ),
                "curriculum/num_kv_pairs": float(
                    curriculum["num_kv_pairs_after_probe"]
                ),
                "curriculum/frontier_accuracy": curriculum["frontier_accuracy"],
                "curriculum/advanced": float(curriculum["transition"] is not None),
            }
        )
    for row in entry["grid"]:
        prefix = (
            f"validation_grid/seq_len_{row['sequence_length']}"
            f"/kv_pairs_{row['num_kv_pairs']}"
        )
        metrics[f"{prefix}/accuracy"] = row["accuracy"]
        metrics[f"{prefix}/loss"] = row["loss"]
    return metrics


def _update_wandb_metrics(entry: dict[str, Any]) -> dict[str, float]:
    metrics = {
        "generation": float(entry["generation"]),
        "train/sampled_stage": float(entry["sampled_stage"]),
        "train/sampled_sequence_length": float(entry["sampled_sequence_length"]),
        "train/sampled_num_kv_pairs": float(entry["sampled_num_kv_pairs"]),
        "train/curriculum_stage": float(entry["curriculum_stage"]),
        "train/curriculum_sequence_length": float(entry["curriculum_sequence_length"]),
        "train/curriculum_num_kv_pairs": float(entry["curriculum_num_kv_pairs"]),
        "train/unique_sequences_seen": float(entry["unique_training_sequences_seen"]),
        "train/learning_rate": entry["learning_rate"],
    }
    if "sigma" in entry:
        metrics["train/sigma"] = entry["sigma"]
    metrics.update({f"train/{name}": value for name, value in entry["metrics"].items()})
    return metrics


def _test_wandb_metrics(results: dict[str, Any],) -> dict[str, float]:
    method = results["method"]
    metrics: dict[str, float] = {
        "generation": float(results["config"]["generations"]),
        "timing/experiment_seconds": results["timing_seconds"]["experiment"],
        "timing/training_seconds": results["timing_seconds"]["training"],
        "curriculum/final_trained_sequence_length": float(
            results["curriculum"]["last_trained_sequence_length"]
        ),
        "curriculum/final_trained_num_kv_pairs": float(
            results["curriculum"]["last_trained_num_kv_pairs"]
        ),
    }
    for name, value in results["test"]["summary"][method].items():
        if value is not None:
            metrics[f"test/{name}"] = value
    for row in results["test"]["grid"]:
        prefix = (
            f"test_grid/seq_len_{row['sequence_length']}"
            f"/kv_pairs_{row['num_kv_pairs']}"
        )
        metrics[f"{prefix}/accuracy"] = row["accuracy"]
        metrics[f"{prefix}/loss"] = row["loss"]
    return metrics


def _finish_wandb(run: Any, output_dir: Path, results: dict[str, Any],) -> None:
    final_metrics = _test_wandb_metrics(results)
    run.log(final_metrics)
    run.summary.update(
        {key: value for key, value in final_metrics.items() if key != "generation"}
    )
    run.save(
        str(output_dir / "metrics.json"), base_path=str(output_dir), policy="now",
    )
    run.save(
        str(output_dir / "model.pt"), base_path=str(output_dir), policy="now",
    )
    run.finish()


def _parameter_l2_norm(model: VanillaRNN) -> float:
    return float(
        torch.sqrt(
            sum(
                parameter.detach().float().square().sum()
                for parameter in model.parameters()
            )
        )
    )


def _model_gradient_rms(model: VanillaRNN) -> float:
    gradients = {
        name: parameter.grad
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return gradient_rms(gradients)


def _sample_training_stage(
    config: ExperimentConfig, state: CurriculumState, *, generator: torch.Generator,
) -> int:
    """Sample a reached Zoology stage, with explicit frontier rehearsal."""

    frontier = state.current_stage(config)
    if frontier == 0:
        return 0
    if config.curriculum_enabled and float(torch.rand((), generator=generator)) < (
        config.curriculum_frontier_probability
    ):
        return frontier
    upper_bound = frontier if config.curriculum_enabled else frontier + 1
    return int(torch.randint(0, upper_bound, (1,), generator=generator))


def update_curriculum(
    state: CurriculumState,
    accuracy: float,
    config: ExperimentConfig,
    *,
    generation: int,
) -> dict[str, Any] | None:
    """Advance one stage when the selected gate passes its frontier probe."""

    if not config.curriculum_enabled:
        return None
    if accuracy < config.curriculum_accuracy_threshold:
        return None
    if state.stage == len(config.curriculum_sequence_lengths) - 1:
        return None
    previous_length, previous_pairs = state.current_task(config)
    state.stage += 1
    next_length, next_pairs = state.current_task(config)
    transition = {
        "generation": generation,
        "from_sequence_length": previous_length,
        "from_num_kv_pairs": previous_pairs,
        "to_sequence_length": next_length,
        "to_num_kv_pairs": next_pairs,
        "frontier_accuracy": accuracy,
        "threshold": config.curriculum_accuracy_threshold,
    }
    state.transitions.append(transition)
    return transition


def _bptt_update(
    model: VanillaRNN,
    optimizer: torch.optim.Optimizer,
    inputs: Tensor,
    targets: Tensor,
    *,
    gradient_clip: float,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    readout_mask = targets.ne(IGNORE_INDEX)
    supervised_targets = targets[readout_mask]
    logits = model(inputs, readout_mask=readout_mask)
    assert isinstance(logits, Tensor)
    loss = F.cross_entropy(logits, supervised_targets)
    loss.backward()
    unclipped_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    clipped_gradient_rms = _model_gradient_rms(model)
    optimizer.step()
    return {
        "batch_loss": float(loss.detach()),
        "batch_accuracy": float(
            logits.argmax(dim=-1).eq(supervised_targets).float().mean()
        ),
        "gradient_l2_norm_before_clip": float(unclipped_norm),
        "gradient_rms_after_clip": clipped_gradient_rms,
    }


def _eggroll_update(
    model: VanillaRNN,
    optimizer: torch.optim.Optimizer,
    inputs: Tensor,
    targets: Tensor,
    *,
    local_population_size: int,
    global_population_size: int,
    candidate_chunk_size: int,
    perturbation_rank: int,
    sigma: float,
    fitness_shaping: str,
    noise_generator: torch.Generator,
) -> dict[str, float]:
    noise = sample_antithetic_noise(
        model, local_population_size, perturbation_rank, generator=noise_generator,
    )
    model.eval()
    with torch.no_grad():
        local_losses, local_accuracies = evaluate_population(
            model,
            inputs,
            targets,
            noise,
            sigma,
            candidate_chunk_size=min(candidate_chunk_size, local_population_size),
        )
        global_losses = _gather_equal_shards(local_losses)
        global_accuracies = _gather_equal_shards(local_accuracies)
        global_pair_gaps = _gather_equal_shards(
            (local_losses[: noise.pair_count] - local_losses[noise.pair_count :]).abs()
        )
        global_fitness = shape_fitness(global_losses, fitness_shaping)
        shard_start = _rank() * local_population_size
        local_fitness = global_fitness[
            shard_start : shard_start + local_population_size
        ]
        raw_reward_gradients = estimate_reward_gradients(noise, local_fitness)
        _average_gradients(raw_reward_gradients)
        # This is the update scaling used by the working spiral implementation.
        gradient_scale = sigma * math.sqrt(global_population_size)
        reward_gradients = {
            name: value * gradient_scale for name, value in raw_reward_gradients.items()
        }
        readout_mask = targets.ne(IGNORE_INDEX)
        supervised_targets = targets[readout_mask]
        mean_logits = model(inputs, readout_mask=readout_mask)
        assert isinstance(mean_logits, Tensor)
        mean_loss_before = float(F.cross_entropy(mean_logits, supervised_targets))
        mean_accuracy_before = float(
            mean_logits.argmax(dim=-1).eq(supervised_targets).float().mean()
        )

    optimizer.zero_grad(set_to_none=True)
    assign_maximization_gradients(model, reward_gradients)
    optimizer.step()
    return {
        "mean_model_batch_loss": mean_loss_before,
        "mean_model_batch_accuracy": mean_accuracy_before,
        "candidate_loss_mean": float(global_losses.mean()),
        "candidate_loss_std": float(global_losses.std(unbiased=False)),
        "candidate_accuracy_mean": float(global_accuracies.mean()),
        "fitness_std": float(global_fitness.std(unbiased=False)),
        "antithetic_loss_gap_abs_mean": float(global_pair_gaps.mean()),
        "gradient_scale": gradient_scale,
        "raw_gradient_rms": gradient_rms(raw_reward_gradients),
        "gradient_rms": gradient_rms(reward_gradients),
    }


def _evaluation_seed(
    base_seed: int, split: str, sequence_length: int, num_kv_pairs: int
) -> int:
    split_offset = {"validation": 100_003, "test": 200_003}[split]
    return base_seed + split_offset + 1_009 * num_kv_pairs + 9_176 * sequence_length


@torch.no_grad()
def evaluate_grid(
    models: dict[str, VanillaRNN],
    task_config: MQARConfig,
    config: ExperimentConfig,
    *,
    split: str,
    example_count: int,
    device: torch.device,
    evaluation_stages: tuple[int, ...] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate each model on byte-identical deterministic grid examples."""

    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    rows: list[dict[str, Any]] = []
    stages = (
        tuple(range(len(config.curriculum_sequence_lengths)))
        if evaluation_stages is None
        else evaluation_stages
    )
    for stage in stages:
        sequence_length = config.curriculum_sequence_lengths[stage]
        num_kv_pairs = config.curriculum_num_kv_pairs[stage]
        generator = torch.Generator().manual_seed(
            _evaluation_seed(config.seed, split, sequence_length, num_kv_pairs)
        )
        totals = {name: {"loss": 0.0, "correct": 0} for name in models}
        seen = 0
        while seen < example_count:
            current_size = min(config.evaluation_batch_size, example_count - seen)
            batch = sample_batch(
                current_size,
                sequence_length,
                num_kv_pairs,
                task_config,
                generator=generator,
            ).to(device)
            readout_mask = batch.targets.ne(IGNORE_INDEX)
            supervised_targets = batch.targets[readout_mask]
            for name, model in models.items():
                model.eval()
                logits = model(batch.inputs, readout_mask=readout_mask)
                assert isinstance(logits, Tensor)
                totals[name]["loss"] += float(
                    F.cross_entropy(logits, supervised_targets, reduction="sum")
                )
                totals[name]["correct"] += int(
                    logits.argmax(dim=-1).eq(supervised_targets).sum()
                )
            seen += current_size
        supervised_count = example_count * num_kv_pairs
        for name in models:
            rows.append(
                {
                    "method": name,
                    "stage": stage,
                    "sequence_length": sequence_length,
                    "num_kv_pairs": num_kv_pairs,
                    "examples": example_count,
                    "supervised_queries": supervised_count,
                    "loss": totals[name]["loss"] / supervised_count,
                    "accuracy": totals[name]["correct"] / supervised_count,
                }
            )
    return rows


def summarize_grid(rows: list[dict[str, Any]],) -> dict[str, dict[str, float | None]]:
    summaries: dict[str, dict[str, float | None]] = {}
    for method in sorted({row["method"] for row in rows}):
        method_rows = [row for row in rows if row["method"] == method]
        summaries[method] = {
            "reached_stage_accuracy_mean": (
                sum(row["accuracy"] for row in method_rows) / len(method_rows)
                if method_rows
                else None
            ),
        }
    return summaries


def run_experiment(
    output_dir: Path, *, device: torch.device, config: ExperimentConfig,
) -> dict[str, Any] | None:
    """Train one method with its own persistent Zoology MQAR curriculum."""

    world_size = _world_size()
    rank = _rank()
    if config.method == "bptt" and world_size != 1:
        raise ValueError("BPTT runs use one process; torchrun is only for EGGROLL")
    local_population_size = 0
    if config.method == "eggroll":
        if config.population_size % world_size:
            raise ValueError("population_size must be divisible by the worker count")
        local_population_size = config.population_size // world_size
        if local_population_size < 2 or local_population_size % 2:
            raise ValueError("each worker needs a positive even population shard")
    if _is_primary():
        output_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()
    wandb_run = _initialize_wandb(config, output_dir)

    task_config = MQARConfig(
        vocab_size=config.vocab_size,
        power_a=config.query_power_a,
        random_non_queries=config.random_non_queries,
    )
    torch.manual_seed(config.seed)
    model = VanillaRNN(
        config.vocab_size, config.hidden_size, recurrent_radius=config.recurrent_radius,
    ).to(device)
    _broadcast_model(model)
    initial_checksum = _state_checksum(model)

    if config.method == "bptt":
        optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.bptt_learning_rate,
            weight_decay=config.bptt_weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.eggroll_learning_rate,
            momentum=config.eggroll_momentum,
            weight_decay=config.eggroll_weight_decay,
        )
    data_generator = torch.Generator().manual_seed(config.seed + 30_001)
    stage_generator = torch.Generator().manual_seed(config.seed + 40_001)
    generator_device = device if device.type != "mps" else torch.device("cpu")
    noise_generator = torch.Generator(device=generator_device)
    noise_generator.manual_seed(config.seed + 60_001 + rank * 1_000_003)

    validation_history: list[dict[str, Any]] = []
    update_history: list[dict[str, Any]] = []
    method_seconds = 0.0
    curriculum_state = CurriculumState()
    last_training_stage = curriculum_state.current_stage(config)

    def record(step: int, *, allow_curriculum_advance: bool) -> None:
        frontier_stage = curriculum_state.current_stage(config)
        frontier_length, frontier_pairs = curriculum_state.current_task(config)
        if _is_primary():
            grid = evaluate_grid(
                {config.method: model},
                task_config,
                config,
                split="validation",
                example_count=config.evaluation_examples,
                device=device,
                evaluation_stages=tuple(range(frontier_stage + 1)),
            )
            curriculum_metrics: dict[str, Any] = {
                "enabled": config.curriculum_enabled,
                "stage_before_probe": frontier_stage,
                "sequence_length_before_probe": frontier_length,
                "num_kv_pairs_before_probe": frontier_pairs,
            }
            transition = None
            if config.curriculum_enabled:
                probe_grid = evaluate_grid(
                    {config.method: model},
                    task_config,
                    config,
                    split="validation",
                    example_count=config.curriculum_probe_examples,
                    device=device,
                    evaluation_stages=(frontier_stage,),
                )
                probe_accuracy = probe_grid[0]["accuracy"]
                if allow_curriculum_advance:
                    transition = update_curriculum(
                        curriculum_state, probe_accuracy, config, generation=step,
                    )
                curriculum_metrics.update(
                    {
                        "probe_examples": config.curriculum_probe_examples,
                        "frontier_accuracy": probe_accuracy,
                        "transition": transition,
                        "stage_after_probe": curriculum_state.stage,
                        "sequence_length_after_probe": (
                            curriculum_state.current_task(config)[0]
                        ),
                        "num_kv_pairs_after_probe": (
                            curriculum_state.current_task(config)[1]
                        ),
                    }
                )
            entry = {
                "step": step,
                "summary": summarize_grid(grid),
                "grid": grid,
                "curriculum": curriculum_metrics,
                "parameter_l2_norm": _parameter_l2_norm(model),
            }
            validation_history.append(entry)
            if wandb_run is not None:
                wandb_run.log(_validation_wandb_metrics(entry, config.method))
            if config.log_progress:
                print(
                    json.dumps(
                        {
                            "validation": entry["summary"],
                            "curriculum": curriculum_metrics,
                            "step": step,
                        }
                    ),
                    flush=True,
                )
        _broadcast_curriculum_state(curriculum_state, device)

    record(0, allow_curriculum_advance=False)
    current_sigma = config.sigma
    experiment_start = time.perf_counter()
    for generation in range(1, config.generations + 1):
        training_frontier = curriculum_state.current_stage(config)
        last_training_stage = training_frontier
        sampled_stage = _sample_training_stage(
            config, curriculum_state, generator=stage_generator,
        )
        sequence_length = config.curriculum_sequence_lengths[sampled_stage]
        num_kv_pairs = config.curriculum_num_kv_pairs[sampled_stage]
        batch = sample_batch(
            config.batch_size,
            sequence_length,
            num_kv_pairs,
            task_config,
            generator=data_generator,
        ).to(device)

        _synchronize(device)
        start = time.perf_counter()
        if config.method == "eggroll":
            method_metrics = _eggroll_update(
                model,
                optimizer,
                batch.inputs,
                batch.targets,
                local_population_size=local_population_size,
                global_population_size=config.population_size,
                candidate_chunk_size=config.population_chunk_size,
                perturbation_rank=config.perturbation_rank,
                sigma=current_sigma,
                fitness_shaping=config.fitness_shaping,
                noise_generator=noise_generator,
            )
        else:
            method_metrics = _bptt_update(
                model,
                optimizer,
                batch.inputs,
                batch.targets,
                gradient_clip=config.bptt_gradient_clip,
            )
        _synchronize(device)
        update_seconds = time.perf_counter() - start
        method_seconds += update_seconds

        should_record = (
            generation == 1
            or generation % config.evaluation_interval == 0
            or generation == config.generations
        )
        if should_record and _is_primary():
            update_entry = {
                "generation": generation,
                "sampled_stage": sampled_stage,
                "sampled_sequence_length": sequence_length,
                "sampled_num_kv_pairs": num_kv_pairs,
                "curriculum_stage": curriculum_state.stage,
                "curriculum_sequence_length": (
                    config.curriculum_sequence_lengths[training_frontier]
                ),
                "curriculum_num_kv_pairs": (
                    config.curriculum_num_kv_pairs[training_frontier]
                ),
                "unique_training_sequences_seen": generation * config.batch_size,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "method": config.method,
                "metrics": {**method_metrics, "seconds": update_seconds},
            }
            if config.method == "eggroll":
                update_entry["sigma"] = current_sigma
            update_history.append(update_entry)
            if wandb_run is not None:
                wandb_run.log(_update_wandb_metrics(update_entry))
            if config.log_progress:
                print(json.dumps({"update": update_entry}), flush=True)

        current_sigma *= config.sigma_decay
        optimizer.param_groups[0]["lr"] *= (
            config.eggroll_learning_rate_decay
            if config.method == "eggroll"
            else config.bptt_learning_rate_decay
        )
        if (
            generation % config.evaluation_interval == 0
            or generation == config.generations
        ):
            record(generation, allow_curriculum_advance=True)

    _synchronize(device)
    experiment_seconds = time.perf_counter() - experiment_start
    if not _is_primary():
        if dist.is_initialized():
            dist.barrier()
        return None

    test_grid = evaluate_grid(
        {config.method: model},
        task_config,
        config,
        split="test",
        example_count=config.test_examples,
        device=device,
        evaluation_stages=tuple(range(last_training_stage + 1)),
    )
    results: dict[str, Any] = {
        "experiment": "single_method_zoology_mqar_curriculum",
        "method": config.method,
        "config": asdict(config),
        "model": {
            "architecture": "single_layer_tanh_elman_rnn",
            "vocab_size": task_config.vocab_size,
            "hidden_size": config.hidden_size,
            "parameter_count": model.parameter_count,
            "initial_checksum": initial_checksum,
        },
        "distributed": {
            "world_size": world_size,
            "global_population": (
                config.population_size if config.method == "eggroll" else None
            ),
            "population_per_worker": (
                local_population_size if config.method == "eggroll" else None
            ),
        },
        "budgets": {
            "unique_training_sequences": config.generations * config.batch_size,
            "training_sequences": config.generations * config.batch_size,
            "eggroll_candidate_forward_sequences": (
                config.generations * config.batch_size * config.population_size
                if config.method == "eggroll"
                else 0
            ),
        },
        "timing_seconds": {
            "experiment": experiment_seconds,
            "training": method_seconds,
        },
        "validation_history": validation_history,
        "update_history": update_history,
        "curriculum": {
            "enabled": config.curriculum_enabled,
            "last_trained_stage": last_training_stage,
            "last_trained_sequence_length": (
                config.curriculum_sequence_lengths[last_training_stage]
            ),
            "last_trained_num_kv_pairs": (
                config.curriculum_num_kv_pairs[last_training_stage]
            ),
            "next_stage": curriculum_state.current_stage(config),
            "next_sequence_length": curriculum_state.current_task(config)[0],
            "next_num_kv_pairs": curriculum_state.current_task(config)[1],
            "final_stage": curriculum_state.stage,
            "transitions": curriculum_state.transitions,
        },
        "test": {"summary": summarize_grid(test_grid), "grid": test_grid,},
        "final_parameter_l2_norm": _parameter_l2_norm(model),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    torch.save(model.state_dict(), output_dir / "model.pt")
    if wandb_run is not None:
        _finish_wandb(wandb_run, output_dir, results)
    if dist.is_initialized():
        dist.barrier()
    return results


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("bptt", "eggroll"), required=True)
    parser.add_argument("--preset", choices=("smoke", "reference"), default="smoke")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/smoke"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--generations", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--population-size", type=int)
    parser.add_argument("--population-chunk-size", type=int)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--recurrent-radius", type=float)
    parser.add_argument(
        "--curriculum", action=argparse.BooleanOptionalAction, default=None,
    )
    parser.add_argument("--curriculum-sequence-lengths", type=_parse_int_tuple)
    parser.add_argument("--curriculum-num-kv-pairs", type=_parse_int_tuple)
    parser.add_argument("--curriculum-accuracy-threshold", type=float)
    parser.add_argument("--curriculum-frontier-probability", type=float)
    parser.add_argument("--curriculum-probe-examples", type=int)
    parser.add_argument("--evaluation-examples", type=int)
    parser.add_argument("--test-examples", type=int)
    parser.add_argument("--evaluation-interval", type=int)
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--sigma-decay", type=float)
    parser.add_argument("--perturbation-rank", type=int)
    parser.add_argument(
        "--fitness-shaping", choices=("zscore", "centered-rank", "centered"),
    )
    parser.add_argument("--eggroll-learning-rate", type=float)
    parser.add_argument("--eggroll-learning-rate-decay", type=float)
    parser.add_argument("--eggroll-weight-decay", type=float)
    parser.add_argument("--eggroll-momentum", type=float)
    parser.add_argument("--bptt-learning-rate", type=float)
    parser.add_argument("--bptt-learning-rate-decay", type=float)
    parser.add_argument("--bptt-weight-decay", type=float)
    parser.add_argument("--bptt-gradient-clip", type=float)
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--query-power-a", type=float)
    parser.add_argument(
        "--random-non-queries", action=argparse.BooleanOptionalAction, default=None,
    )
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-group")
    parser.add_argument("--log-progress", action="store_true")
    return parser


def _apply_cli_overrides(
    config: ExperimentConfig, args: argparse.Namespace
) -> ExperimentConfig:
    names = (
        "method",
        "seed",
        "generations",
        "batch_size",
        "population_size",
        "population_chunk_size",
        "hidden_size",
        "recurrent_radius",
        "curriculum_sequence_lengths",
        "curriculum_num_kv_pairs",
        "curriculum_accuracy_threshold",
        "curriculum_frontier_probability",
        "curriculum_probe_examples",
        "evaluation_examples",
        "test_examples",
        "evaluation_interval",
        "sigma",
        "sigma_decay",
        "perturbation_rank",
        "fitness_shaping",
        "eggroll_learning_rate",
        "eggroll_learning_rate_decay",
        "eggroll_weight_decay",
        "eggroll_momentum",
        "bptt_learning_rate",
        "bptt_learning_rate_decay",
        "bptt_weight_decay",
        "bptt_gradient_clip",
        "vocab_size",
        "query_power_a",
        "wandb_project",
        "wandb_entity",
        "wandb_run_name",
        "wandb_group",
    )
    overrides = {
        name: getattr(args, name) for name in names if getattr(args, name) is not None
    }
    if args.curriculum is not None:
        overrides["curriculum_enabled"] = args.curriculum
    if args.random_non_queries is not None:
        overrides["random_non_queries"] = args.random_non_queries
    if args.wandb is not None:
        overrides["wandb_enabled"] = args.wandb
    overrides["log_progress"] = args.log_progress
    return replace(config, **overrides)


def main() -> None:
    args = _parser().parse_args()
    config = smoke_config() if args.preset == "smoke" else ExperimentConfig()
    config = _apply_cli_overrides(config, args)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and config.method != "eggroll":
        raise RuntimeError("torchrun is only supported for EGGROLL runs")
    if world_size > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        if not torch.cuda.is_available():
            raise RuntimeError("multi-worker population evaluation requires CUDA")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl")
    else:
        device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if args.device == "auto"
            else torch.device(args.device)
        )
    try:
        results = run_experiment(args.output_dir, device=device, config=config)
        if results is not None:
            print(
                json.dumps(
                    {
                        "output_dir": str(args.output_dir),
                        "test_summary": results["test"]["summary"],
                    }
                )
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
