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
from .task import AssociativeRecallConfig, sample_batch


@dataclass(frozen=True)
class ExperimentConfig:
    method: str = "bptt"
    seed: int = 7
    generations: int = 3_000
    batch_size: int = 256
    train_num_pairs: int = 2
    train_min_delay: int = 0
    train_max_delay: int = 32
    curriculum_enabled: bool = True
    curriculum_delays: tuple[int, ...] = (0, 2, 4, 8, 16, 32)
    curriculum_accuracy_threshold: float = 0.9
    curriculum_frontier_probability: float = 0.5
    curriculum_probe_examples: int = 512
    evaluation_pairs: tuple[int, ...] = (1, 2, 4)
    evaluation_delays: tuple[int, ...] = (0, 2, 4, 8, 16, 32, 64, 128, 256)
    evaluation_examples: int = 1_024
    test_examples: int = 4_096
    evaluation_batch_size: int = 256
    evaluation_interval: int = 250
    num_keys: int = 4
    num_values: int = 4
    distractor_std: float = 0.0
    hidden_size: int = 32
    recurrent_radius: float = 0.9
    population_size: int = 16_384
    population_chunk_size: int = 2_048
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
    log_progress: bool = False

    def __post_init__(self) -> None:
        if self.method not in {"bptt", "eggroll"}:
            raise ValueError("method must be 'bptt' or 'eggroll'")
        if self.generations < 1 or self.batch_size < 1:
            raise ValueError("generations and batch_size must be positive")
        if self.train_num_pairs < 1:
            raise ValueError("train_num_pairs must be positive")
        if not 0 <= self.train_min_delay <= self.train_max_delay:
            raise ValueError("training delay bounds are invalid")
        if not self.curriculum_delays:
            raise ValueError("curriculum_delays must be non-empty")
        if tuple(sorted(set(self.curriculum_delays))) != self.curriculum_delays:
            raise ValueError("curriculum_delays must be unique and increasing")
        if self.curriculum_enabled and not (
            self.train_min_delay <= self.curriculum_delays[0]
            and self.curriculum_delays[-1] <= self.train_max_delay
        ):
            raise ValueError("curriculum delays must lie inside the training range")
        if not 0 <= self.curriculum_accuracy_threshold <= 1:
            raise ValueError("curriculum_accuracy_threshold must be in [0, 1]")
        if not 0 <= self.curriculum_frontier_probability <= 1:
            raise ValueError("curriculum_frontier_probability must be in [0, 1]")
        if self.curriculum_probe_examples < 1:
            raise ValueError("curriculum_probe_examples must be positive")
        if not self.evaluation_pairs or min(self.evaluation_pairs) < 1:
            raise ValueError("evaluation_pairs must be non-empty and positive")
        if tuple(sorted(set(self.evaluation_pairs))) != self.evaluation_pairs:
            raise ValueError("evaluation_pairs must be unique and increasing")
        if not self.evaluation_delays or min(self.evaluation_delays) < 0:
            raise ValueError("evaluation_delays must be non-empty and non-negative")
        if tuple(sorted(set(self.evaluation_delays))) != self.evaluation_delays:
            raise ValueError("evaluation_delays must be unique and increasing")
        vocabulary_limit = min(self.num_keys, self.num_values)
        if max((*self.evaluation_pairs, self.train_num_pairs)) > vocabulary_limit:
            raise ValueError("pair counts cannot exceed either vocabulary size")
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


def smoke_config(seed: int = 7) -> ExperimentConfig:
    """A structural integration check, deliberately too small for conclusions."""

    return ExperimentConfig(
        seed=seed,
        generations=4,
        batch_size=8,
        train_num_pairs=2,
        train_min_delay=0,
        train_max_delay=4,
        curriculum_delays=(0, 2, 4),
        curriculum_probe_examples=16,
        evaluation_pairs=(1, 2),
        evaluation_delays=(0, 2, 4, 8),
        evaluation_examples=32,
        test_examples=32,
        evaluation_batch_size=16,
        evaluation_interval=2,
        num_keys=4,
        num_values=4,
        hidden_size=8,
        population_size=16,
        population_chunk_size=8,
    )


@dataclass
class CurriculumState:
    """Mutable state shared across workers while configuration stays immutable."""

    stage: int = 0
    transitions: list[dict[str, Any]] = field(default_factory=list)

    def current_max_delay(self, config: ExperimentConfig) -> int:
        if not config.curriculum_enabled:
            return config.train_max_delay
        return config.curriculum_delays[self.stage]


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


def _sample_training_delay(
    config: ExperimentConfig, state: CurriculumState, *, generator: torch.Generator,
) -> int:
    """Sample uniformly below the frontier, with explicit frontier rehearsal."""

    frontier = state.current_max_delay(config)
    if frontier <= config.train_min_delay:
        return frontier
    if config.curriculum_enabled and float(torch.rand((), generator=generator)) < (
        config.curriculum_frontier_probability
    ):
        return frontier
    # The frontier has exactly the configured probability; the remaining mass
    # is spread over previously introduced delays.
    upper_bound = frontier if config.curriculum_enabled else frontier + 1
    return int(
        torch.randint(config.train_min_delay, upper_bound, (1,), generator=generator,)
    )


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
    if state.stage == len(config.curriculum_delays) - 1:
        return None
    previous_delay = config.curriculum_delays[state.stage]
    state.stage += 1
    transition = {
        "generation": generation,
        "from_max_delay": previous_delay,
        "to_max_delay": config.curriculum_delays[state.stage],
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
    logits = model(inputs)
    assert isinstance(logits, Tensor)
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    unclipped_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    clipped_gradient_rms = _model_gradient_rms(model)
    optimizer.step()
    return {
        "batch_loss": float(loss.detach()),
        "batch_accuracy": float(logits.argmax(dim=-1).eq(targets).float().mean()),
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
        mean_logits = model(inputs)
        assert isinstance(mean_logits, Tensor)
        mean_loss_before = float(F.cross_entropy(mean_logits, targets))
        mean_accuracy_before = float(
            mean_logits.argmax(dim=-1).eq(targets).float().mean()
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


def _evaluation_seed(base_seed: int, split: str, num_pairs: int, delay: int) -> int:
    split_offset = {"validation": 100_003, "test": 200_003}[split]
    return base_seed + split_offset + 1_009 * num_pairs + 9_176 * delay


@torch.no_grad()
def evaluate_grid(
    models: dict[str, VanillaRNN],
    task_config: AssociativeRecallConfig,
    config: ExperimentConfig,
    *,
    split: str,
    example_count: int,
    device: torch.device,
    evaluation_pairs: tuple[int, ...] | None = None,
    evaluation_delays: tuple[int, ...] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate each model on byte-identical deterministic grid examples."""

    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    rows: list[dict[str, Any]] = []
    pairs = config.evaluation_pairs if evaluation_pairs is None else evaluation_pairs
    delays = (
        config.evaluation_delays if evaluation_delays is None else evaluation_delays
    )
    for num_pairs in pairs:
        for delay in delays:
            generator = torch.Generator().manual_seed(
                _evaluation_seed(config.seed, split, num_pairs, delay)
            )
            totals = {name: {"loss": 0.0, "correct": 0} for name in models}
            seen = 0
            while seen < example_count:
                current_size = min(config.evaluation_batch_size, example_count - seen,)
                batch = sample_batch(
                    current_size, num_pairs, delay, task_config, generator=generator,
                ).to(device)
                for name, model in models.items():
                    model.eval()
                    logits = model(batch.inputs)
                    assert isinstance(logits, Tensor)
                    totals[name]["loss"] += float(
                        F.cross_entropy(logits, batch.targets, reduction="sum")
                    )
                    totals[name]["correct"] += int(
                        logits.argmax(dim=-1).eq(batch.targets).sum()
                    )
                seen += current_size
            for name in models:
                rows.append(
                    {
                        "method": name,
                        "num_pairs": num_pairs,
                        "delay": delay,
                        "examples": example_count,
                        "loss": totals[name]["loss"] / example_count,
                        "accuracy": totals[name]["correct"] / example_count,
                    }
                )
    return rows


def summarize_grid(
    rows: list[dict[str, Any]],
    config: ExperimentConfig,
    *,
    trained_max_delay: int | None = None,
) -> dict[str, dict[str, float | None]]:
    if trained_max_delay is None:
        trained_max_delay = config.train_max_delay
    summaries: dict[str, dict[str, float | None]] = {}
    for method in sorted({row["method"] for row in rows}):
        method_rows = [row for row in rows if row["method"] == method]
        in_distribution = [
            row["accuracy"]
            for row in method_rows
            if row["num_pairs"] == config.train_num_pairs
            and config.train_min_delay <= row["delay"] <= trained_max_delay
        ]
        extrapolation = [
            row["accuracy"]
            for row in method_rows
            if row["num_pairs"] == config.train_num_pairs
            and row["delay"] > trained_max_delay
        ]
        summaries[method] = {
            "in_distribution_accuracy_mean": (
                sum(in_distribution) / len(in_distribution) if in_distribution else None
            ),
            "delay_extrapolation_accuracy_mean": (
                sum(extrapolation) / len(extrapolation) if extrapolation else None
            ),
        }
    return summaries


def run_experiment(
    output_dir: Path, *, device: torch.device, config: ExperimentConfig,
) -> dict[str, Any] | None:
    """Train one method with its own persistent delay curriculum."""

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

    task_config = AssociativeRecallConfig(
        num_keys=config.num_keys,
        num_values=config.num_values,
        distractor_std=config.distractor_std,
    )
    torch.manual_seed(config.seed)
    model = VanillaRNN(
        task_config.input_size,
        config.hidden_size,
        task_config.num_values,
        recurrent_radius=config.recurrent_radius,
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
    delay_generator = torch.Generator().manual_seed(config.seed + 40_001)
    generator_device = device if device.type != "mps" else torch.device("cpu")
    noise_generator = torch.Generator(device=generator_device)
    noise_generator.manual_seed(config.seed + 60_001 + rank * 1_000_003)

    validation_history: list[dict[str, Any]] = []
    update_history: list[dict[str, Any]] = []
    method_seconds = 0.0
    curriculum_state = CurriculumState()
    last_training_frontier = curriculum_state.current_max_delay(config)

    def record(step: int, *, allow_curriculum_advance: bool) -> None:
        frontier_before_probe = curriculum_state.current_max_delay(config)
        if _is_primary():
            grid = evaluate_grid(
                {config.method: model},
                task_config,
                config,
                split="validation",
                example_count=config.evaluation_examples,
                device=device,
            )
            curriculum_metrics: dict[str, Any] = {
                "enabled": config.curriculum_enabled,
                "stage_before_probe": curriculum_state.stage,
                "max_delay_before_probe": frontier_before_probe,
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
                    evaluation_pairs=(config.train_num_pairs,),
                    evaluation_delays=(frontier_before_probe,),
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
                        "max_delay_after_probe": (
                            curriculum_state.current_max_delay(config)
                        ),
                    }
                )
            entry = {
                "step": step,
                "summary": summarize_grid(
                    grid, config, trained_max_delay=frontier_before_probe,
                ),
                "grid": grid,
                "curriculum": curriculum_metrics,
                "parameter_l2_norm": _parameter_l2_norm(model),
            }
            validation_history.append(entry)
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
        training_frontier = curriculum_state.current_max_delay(config)
        last_training_frontier = training_frontier
        delay = _sample_training_delay(
            config, curriculum_state, generator=delay_generator,
        )
        batch = sample_batch(
            config.batch_size,
            config.train_num_pairs,
            delay,
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
                "sampled_delay": delay,
                "curriculum_stage": curriculum_state.stage,
                "curriculum_max_delay": training_frontier,
                "unique_training_sequences_seen": generation * config.batch_size,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "method": config.method,
                "metrics": {**method_metrics, "seconds": update_seconds},
            }
            if config.method == "eggroll":
                update_entry["sigma"] = current_sigma
            update_history.append(update_entry)
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
    )
    results: dict[str, Any] = {
        "experiment": "single_method_associative_recall_memory_horizon",
        "method": config.method,
        "config": asdict(config),
        "model": {
            "architecture": "single_layer_tanh_elman_rnn",
            "input_size": task_config.input_size,
            "hidden_size": config.hidden_size,
            "output_size": task_config.num_values,
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
            "last_trained_max_delay": last_training_frontier,
            "next_max_delay": curriculum_state.current_max_delay(config),
            "final_stage": curriculum_state.stage,
            "transitions": curriculum_state.transitions,
        },
        "test": {
            "summary": summarize_grid(
                test_grid, config, trained_max_delay=last_training_frontier,
            ),
            "grid": test_grid,
        },
        "final_parameter_l2_norm": _parameter_l2_norm(model),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    torch.save(model.state_dict(), output_dir / "model.pt")
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
    parser.add_argument("--train-num-pairs", type=int)
    parser.add_argument("--train-min-delay", type=int)
    parser.add_argument("--train-max-delay", type=int)
    parser.add_argument(
        "--curriculum", action=argparse.BooleanOptionalAction, default=None,
    )
    parser.add_argument("--curriculum-delays", type=_parse_int_tuple)
    parser.add_argument("--curriculum-accuracy-threshold", type=float)
    parser.add_argument("--curriculum-frontier-probability", type=float)
    parser.add_argument("--curriculum-probe-examples", type=int)
    parser.add_argument("--evaluation-pairs", type=_parse_int_tuple)
    parser.add_argument("--evaluation-delays", type=_parse_int_tuple)
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
    parser.add_argument("--distractor-std", type=float)
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
        "train_num_pairs",
        "train_min_delay",
        "train_max_delay",
        "curriculum_delays",
        "curriculum_accuracy_threshold",
        "curriculum_frontier_probability",
        "curriculum_probe_examples",
        "evaluation_pairs",
        "evaluation_delays",
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
        "distractor_std",
    )
    overrides = {
        name: getattr(args, name) for name in names if getattr(args, name) is not None
    }
    if args.curriculum is not None:
        overrides["curriculum_enabled"] = args.curriculum
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
