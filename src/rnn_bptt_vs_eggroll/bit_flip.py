"""BPTT curriculum for a constant-memory one-bit state-machine task."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


START_ZERO = 0
START_ONE = 1
HOLD = 2
FLIP = 3
QUERY = 4
INPUT_VOCAB_SIZE = 5
OUTPUT_SIZE = 2


@dataclass(frozen=True)
class BitFlipBatch:
    inputs: Tensor
    targets: Tensor
    operation_count: int

    def to(self, device: torch.device | str) -> "BitFlipBatch":
        return BitFlipBatch(
            inputs=self.inputs.to(device),
            targets=self.targets.to(device),
            operation_count=self.operation_count,
        )


def sample_bit_flip_batch(
    batch_size: int,
    operation_count: int,
    *,
    generator: torch.Generator,
) -> BitFlipBatch:
    """Sample start bits followed by random HOLD/FLIP operations and QUERY."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if operation_count < 1:
        raise ValueError("operation_count must be positive")
    start_bits = torch.randint(2, (batch_size,), generator=generator)
    flips = torch.randint(2, (batch_size, operation_count), generator=generator)
    inputs = torch.empty(batch_size, operation_count + 2, dtype=torch.long)
    inputs[:, 0] = start_bits
    inputs[:, 1:-1] = flips + HOLD
    inputs[:, -1] = QUERY
    targets = start_bits.bitwise_xor(flips.sum(dim=1).remainder(2))
    return BitFlipBatch(
        inputs=inputs,
        targets=targets,
        operation_count=operation_count,
    )


def build_curriculum(
    *,
    dense_until: int,
    maximum: int,
    growth: float,
) -> tuple[int, ...]:
    """Use unit increments first, then bounded geometric growth."""

    if not 1 <= dense_until <= maximum:
        raise ValueError("dense_until must lie between one and maximum")
    if growth <= 1:
        raise ValueError("growth must exceed one")
    lengths = list(range(1, dense_until + 1))
    while lengths[-1] < maximum:
        next_length = max(lengths[-1] + 1, math.ceil(lengths[-1] * growth))
        lengths.append(min(next_length, maximum))
    return tuple(lengths)


class BitFlipRNN(nn.Module):
    """A fused single-layer tanh RNN with a binary terminal readout."""

    def __init__(self, hidden_size: int, *, recurrent_radius: float = 0.9) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if recurrent_radius <= 0:
            raise ValueError("recurrent_radius must be positive")
        self.hidden_size = hidden_size
        self.recurrent_radius = recurrent_radius
        self.embedding = nn.Embedding(INPUT_VOCAB_SIZE, hidden_size)
        self.recurrent = nn.RNN(
            hidden_size,
            hidden_size,
            nonlinearity="tanh",
            batch_first=True,
        )
        self.readout = nn.Linear(hidden_size, OUTPUT_SIZE)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.embedding.weight, std=1 / math.sqrt(self.hidden_size))
        nn.init.xavier_uniform_(self.recurrent.weight_ih_l0)
        nn.init.orthogonal_(self.recurrent.weight_hh_l0)
        with torch.no_grad():
            self.recurrent.weight_hh_l0.mul_(self.recurrent_radius)
        nn.init.zeros_(self.recurrent.bias_ih_l0)
        nn.init.zeros_(self.recurrent.bias_hh_l0)
        nn.init.xavier_uniform_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 2 or inputs.dtype != torch.long:
            raise ValueError("inputs must have integer shape [batch, time]")
        states, _ = self.recurrent(self.embedding(inputs))
        return self.readout(states[:, -1])

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


@dataclass(frozen=True)
class BitFlipConfig:
    seed: int = 7
    max_updates: int = 2_000_000
    batch_size: int = 256
    hidden_size: int = 16
    recurrent_radius: float = 0.9
    learning_rate: float = 3e-3
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    promotion_accuracy: float = 0.95
    evaluation_interval: int = 100
    evaluation_examples: int = 2_048
    evaluation_batch_size: int = 256
    stage_patience: int = 100_000
    curriculum_dense_until: int = 32
    curriculum_max_operations: int = 16_384
    curriculum_growth: float = 1.25
    checkpoint_interval: int = 10_000
    wandb_enabled: bool = False
    wandb_project: str = "rnn-bptt-vs-eggroll"
    wandb_entity: str | None = "wobrob101"
    wandb_run_name: str | None = None
    wandb_group: str | None = "bit-flip-bptt"
    wandb_run_id: str | None = None
    wandb_resume: str | None = None

    def __post_init__(self) -> None:
        positive_ints = (
            self.max_updates,
            self.batch_size,
            self.hidden_size,
            self.evaluation_interval,
            self.evaluation_examples,
            self.evaluation_batch_size,
            self.stage_patience,
            self.checkpoint_interval,
        )
        if any(value < 1 for value in positive_ints):
            raise ValueError("integer budgets and dimensions must be positive")
        if min(
            self.recurrent_radius,
            self.learning_rate,
            self.gradient_clip,
        ) <= 0:
            raise ValueError("model and optimizer scales must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if not 0 < self.promotion_accuracy <= 1:
            raise ValueError("promotion_accuracy must be in (0, 1]")
        build_curriculum(
            dense_until=self.curriculum_dense_until,
            maximum=self.curriculum_max_operations,
            growth=self.curriculum_growth,
        )
        if self.wandb_resume not in {None, "allow", "must", "never"}:
            raise ValueError("invalid W&B resume policy")
        if self.wandb_resume is not None and self.wandb_run_id is None:
            raise ValueError("W&B resume requires a run ID")


def _evaluation_seed(seed: int, operation_count: int) -> int:
    return seed + 100_003 + 9_176 * operation_count


@torch.no_grad()
def evaluate(
    model: BitFlipRNN,
    *,
    operation_count: int,
    config: BitFlipConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    generator = torch.Generator().manual_seed(
        _evaluation_seed(config.seed, operation_count)
    )
    total_loss = 0.0
    total_correct = 0
    seen = 0
    while seen < config.evaluation_examples:
        current_size = min(
            config.evaluation_batch_size,
            config.evaluation_examples - seen,
        )
        batch = sample_bit_flip_batch(
            current_size,
            operation_count,
            generator=generator,
        ).to(device)
        logits = model(batch.inputs)
        total_loss += float(F.cross_entropy(logits, batch.targets, reduction="sum"))
        total_correct += int(logits.argmax(dim=-1).eq(batch.targets).sum())
        seen += current_size
    return {
        "loss": total_loss / seen,
        "accuracy": total_correct / seen,
    }


def _parameter_l2_norm(model: nn.Module) -> float:
    return float(
        torch.sqrt(
            sum(
                parameter.detach().float().square().sum()
                for parameter in model.parameters()
            )
        )
    )


def _save_checkpoint(
    path: Path,
    *,
    config: BitFlipConfig,
    model: BitFlipRNN,
    optimizer: torch.optim.Optimizer,
    step: int,
    stage: int,
    stage_started_at: int,
    data_generator: torch.Generator,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(
        {
            "format_version": 1,
            "config": asdict(config),
            "step": step,
            "stage": stage,
            "stage_started_at": stage_started_at,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "data_generator_state": data_generator.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            ),
        },
        temporary,
    )
    os.replace(temporary, path)


def run(
    output_dir: Path,
    *,
    config: BitFlipConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Train BPTT on a persistent operation-count curriculum."""

    output_dir.mkdir(parents=True, exist_ok=True)
    curriculum = build_curriculum(
        dense_until=config.curriculum_dense_until,
        maximum=config.curriculum_max_operations,
        growth=config.curriculum_growth,
    )
    torch.manual_seed(config.seed)
    model = BitFlipRNN(
        config.hidden_size,
        recurrent_radius=config.recurrent_radius,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    data_generator = torch.Generator().manual_seed(config.seed + 30_001)
    stage = 0
    stage_started_at = 0
    checkpoint_path = output_dir / "checkpoint.pt"

    wandb_run = None
    if config.wandb_enabled:
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError("W&B tracking requires wandb") from error
        wandb_run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.wandb_run_name or output_dir.name,
            group=config.wandb_group,
            id=config.wandb_run_id,
            resume=config.wandb_resume,
            config={**asdict(config), "curriculum": curriculum},
        )
        wandb_run.define_metric("update")
        wandb_run.define_metric("*", step_metric="update")

    validation_history: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    stopping = "max_updates"
    start_time = time.perf_counter()
    completed_step = 0

    initial = evaluate(
        model,
        operation_count=curriculum[stage],
        config=config,
        device=device,
    )
    validation_history.append(
        {"update": 0, "stage": stage, "operation_count": curriculum[stage], **initial}
    )
    print(json.dumps({"validation": validation_history[-1]}), flush=True)

    for step in range(1, config.max_updates + 1):
        completed_step = step
        operation_count = curriculum[stage]
        batch = sample_bit_flip_batch(
            config.batch_size,
            operation_count,
            generator=data_generator,
        ).to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        update_start = time.perf_counter()
        logits = model(batch.inputs)
        loss = F.cross_entropy(logits, batch.targets)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip,
        )
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        update_seconds = time.perf_counter() - update_start
        train_accuracy = float(logits.argmax(dim=-1).eq(batch.targets).float().mean())

        if wandb_run is not None:
            wandb_run.log(
                {
                    "update": step,
                    "train/loss": float(loss.detach()),
                    "train/accuracy": train_accuracy,
                    "train/gradient_l2_norm_before_clip": float(gradient_norm),
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "train/operation_count": float(operation_count),
                    "train/input_sequence_length": float(operation_count + 2),
                    "train/stage": float(stage),
                    "timing/update_seconds": update_seconds,
                }
            )

        if step % config.evaluation_interval == 0:
            validation = evaluate(
                model,
                operation_count=operation_count,
                config=config,
                device=device,
            )
            entry = {
                "update": step,
                "stage": stage,
                "operation_count": operation_count,
                "input_sequence_length": operation_count + 2,
                "updates_at_stage": step - stage_started_at,
                "parameter_l2_norm": _parameter_l2_norm(model),
                **validation,
            }
            validation_history.append(entry)
            advanced = validation["accuracy"] >= config.promotion_accuracy
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "update": step,
                        "validation/frontier_loss": validation["loss"],
                        "validation/frontier_accuracy": validation["accuracy"],
                        "model/parameter_l2_norm": entry["parameter_l2_norm"],
                        "curriculum/operation_count": float(operation_count),
                        "curriculum/input_sequence_length": float(operation_count + 2),
                        "curriculum/stage": float(stage),
                        "curriculum/advanced": float(advanced),
                    }
                )
            print(json.dumps({"validation": entry}), flush=True)
            if advanced:
                if stage == len(curriculum) - 1:
                    stopping = "maximum_operation_count_passed"
                    break
                previous = operation_count
                stage += 1
                stage_started_at = step
                transition = {
                    "update": step,
                    "from_operation_count": previous,
                    "to_operation_count": curriculum[stage],
                    "accuracy": validation["accuracy"],
                    "threshold": config.promotion_accuracy,
                }
                transitions.append(transition)
                print(json.dumps({"curriculum_transition": transition}), flush=True)
            elif step - stage_started_at >= config.stage_patience:
                stopping = "stage_patience_exhausted"
                break

        if step % config.checkpoint_interval == 0:
            _save_checkpoint(
                checkpoint_path,
                config=config,
                model=model,
                optimizer=optimizer,
                step=step,
                stage=stage,
                stage_started_at=stage_started_at,
                data_generator=data_generator,
            )

    _save_checkpoint(
        checkpoint_path,
        config=config,
        model=model,
        optimizer=optimizer,
        step=completed_step,
        stage=stage,
        stage_started_at=stage_started_at,
        data_generator=data_generator,
    )
    torch.save(model.state_dict(), output_dir / "model.pt")
    results = {
        "experiment": "bit_flip_bptt_curriculum",
        "config": asdict(config),
        "curriculum": curriculum,
        "model": {
            "architecture": "single_layer_tanh_rnn",
            "hidden_size": config.hidden_size,
            "parameter_count": model.parameter_count,
        },
        "completed_updates": completed_step,
        "final_stage": stage,
        "final_operation_count": curriculum[stage],
        "transitions": transitions,
        "validation_history": validation_history,
        "stopping_reason": stopping,
        "elapsed_seconds": time.perf_counter() - start_time,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if wandb_run is not None:
        wandb_run.summary.update(
            {
                "completed_updates": completed_step,
                "final_stage": stage,
                "final_operation_count": curriculum[stage],
                "stopping_reason": stopping,
            }
        )
        wandb_run.finish()
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "completed_updates": completed_step,
                "final_operation_count": curriculum[stage],
                "stopping_reason": stopping,
            }
        ),
        flush=True,
    )
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-updates", type=int, default=2_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--recurrent-radius", type=float, default=0.9)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--promotion-accuracy", type=float, default=0.95)
    parser.add_argument("--evaluation-interval", type=int, default=100)
    parser.add_argument("--evaluation-examples", type=int, default=2_048)
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    parser.add_argument("--stage-patience", type=int, default=100_000)
    parser.add_argument("--curriculum-dense-until", type=int, default=32)
    parser.add_argument("--curriculum-max-operations", type=int, default=16_384)
    parser.add_argument("--curriculum-growth", type=float, default=1.25)
    parser.add_argument("--checkpoint-interval", type=int, default=10_000)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb-project", default="rnn-bptt-vs-eggroll")
    parser.add_argument("--wandb-entity", default="wobrob101")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-group", default="bit-flip-bptt")
    parser.add_argument("--wandb-run-id")
    parser.add_argument("--wandb-resume", choices=("allow", "must", "never"))
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = BitFlipConfig(
        seed=args.seed,
        max_updates=args.max_updates,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        recurrent_radius=args.recurrent_radius,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        promotion_accuracy=args.promotion_accuracy,
        evaluation_interval=args.evaluation_interval,
        evaluation_examples=args.evaluation_examples,
        evaluation_batch_size=args.evaluation_batch_size,
        stage_patience=args.stage_patience,
        curriculum_dense_until=args.curriculum_dense_until,
        curriculum_max_operations=args.curriculum_max_operations,
        curriculum_growth=args.curriculum_growth,
        checkpoint_interval=args.checkpoint_interval,
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_group=args.wandb_group,
        wandb_run_id=args.wandb_run_id,
        wandb_resume=args.wandb_resume,
    )
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    run(args.output_dir, config=config, device=device)


if __name__ == "__main__":
    main()
