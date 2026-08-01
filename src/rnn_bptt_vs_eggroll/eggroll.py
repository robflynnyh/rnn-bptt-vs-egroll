"""Forward-only, low-rank antithetic EGGROLL for the benchmark RNN."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .model import VanillaRNN


@dataclass(frozen=True)
class MatrixFactors:
    left: Tensor
    right: Tensor


@dataclass(frozen=True)
class AntitheticNoise:
    matrices: dict[str, MatrixFactors]
    vectors: dict[str, Tensor]
    rank: int

    @property
    def pair_count(self) -> int:
        values = [
            *(factors.left for factors in self.matrices.values()),
            *self.vectors.values(),
        ]
        if not values:
            raise ValueError("noise contains no parameters")
        return values[0].shape[0]

    @property
    def population_size(self) -> int:
        return 2 * self.pair_count


def sample_antithetic_noise(
    model: nn.Module,
    population_size: int,
    rank: int,
    *,
    generator: torch.Generator,
) -> AntitheticNoise:
    """Sample rank-r matrix noise and full vector noise in +/- pairs."""

    if population_size < 2 or population_size % 2:
        raise ValueError("population_size must be a positive even number")
    if rank < 1:
        raise ValueError("rank must be positive")
    pair_count = population_size // 2
    matrices: dict[str, MatrixFactors] = {}
    vectors: dict[str, Tensor] = {}
    for name, parameter in model.named_parameters():
        if parameter.ndim == 2:
            matrices[name] = MatrixFactors(
                left=torch.randn(
                    pair_count,
                    parameter.shape[0],
                    rank,
                    device=parameter.device,
                    dtype=parameter.dtype,
                    generator=generator,
                ),
                right=torch.randn(
                    pair_count,
                    parameter.shape[1],
                    rank,
                    device=parameter.device,
                    dtype=parameter.dtype,
                    generator=generator,
                ),
            )
        else:
            vectors[name] = torch.randn(
                pair_count,
                *parameter.shape,
                device=parameter.device,
                dtype=parameter.dtype,
                generator=generator,
            )
    return AntitheticNoise(matrices=matrices, vectors=vectors, rank=rank)


def _signed_pair(values: Tensor) -> Tensor:
    return torch.cat((values, -values), dim=0)


def _repeated_pair(values: Tensor) -> Tensor:
    return torch.cat((values, values), dim=0)


def _population_affine(
    inputs: Tensor,
    weight: Tensor,
    weight_name: str,
    noise: AntitheticNoise,
    sigma: float,
    *,
    bias: Tensor | None = None,
    bias_name: str | None = None,
) -> Tensor:
    factors = noise.matrices[weight_name]
    left = _signed_pair(factors.left)
    right = _repeated_pair(factors.right)
    base = F.linear(inputs, weight, bias)
    if inputs.ndim == 2:
        base = base.unsqueeze(0).expand(noise.population_size, -1, -1)
        projected = torch.einsum("bi,pir->pbr", inputs, right)
    elif inputs.ndim == 3:
        projected = torch.einsum("pbi,pir->pbr", inputs, right)
    else:
        raise ValueError("population affine input must be [B, D] or [P, B, D]")
    perturbation = torch.einsum("pbr,por->pbo", projected, left)
    outputs = base + (sigma / math.sqrt(noise.rank)) * perturbation
    if bias is not None:
        if bias_name is None:
            raise ValueError("bias_name is required when a bias is provided")
        outputs = outputs + sigma * _signed_pair(noise.vectors[bias_name])[:, None]
    return outputs


@torch.no_grad()
def population_forward(
    model: VanillaRNN,
    inputs: Tensor,
    noise: AntitheticNoise,
    sigma: float,
) -> Tensor:
    """Evaluate a population without materialising full perturbed matrices."""

    if inputs.ndim != 3:
        raise ValueError("inputs must have shape [batch, time, input_size]")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    hidden = inputs.new_zeros(
        noise.population_size,
        inputs.shape[0],
        model.hidden_size,
    )
    for step in inputs.unbind(dim=1):
        input_term = _population_affine(
            step,
            model.input_weight,
            "input_weight",
            noise,
            sigma,
        )
        recurrent_term = _population_affine(
            hidden,
            model.recurrent_weight,
            "recurrent_weight",
            noise,
            sigma,
            bias=model.hidden_bias,
            bias_name="hidden_bias",
        )
        hidden = torch.tanh(input_term + recurrent_term)
    return _population_affine(
        hidden,
        model.output_weight,
        "output_weight",
        noise,
        sigma,
        bias=model.output_bias,
        bias_name="output_bias",
    )


def slice_pair_noise(noise: AntitheticNoise, start: int, end: int) -> AntitheticNoise:
    """Select complete antithetic pairs for memory-bounded evaluation."""

    if not 0 <= start < end <= noise.pair_count:
        raise ValueError("invalid pair slice")
    return AntitheticNoise(
        matrices={
            name: MatrixFactors(
                left=factors.left[start:end],
                right=factors.right[start:end],
            )
            for name, factors in noise.matrices.items()
        },
        vectors={name: values[start:end] for name, values in noise.vectors.items()},
        rank=noise.rank,
    )


@torch.no_grad()
def evaluate_population(
    model: VanillaRNN,
    inputs: Tensor,
    targets: Tensor,
    noise: AntitheticNoise,
    sigma: float,
    *,
    candidate_chunk_size: int,
) -> tuple[Tensor, Tensor]:
    """Return per-candidate loss and accuracy, chunking complete +/- pairs."""

    if candidate_chunk_size < 2 or candidate_chunk_size % 2:
        raise ValueError("candidate_chunk_size must be positive and even")
    pair_chunk_size = candidate_chunk_size // 2
    positive_losses = []
    negative_losses = []
    positive_accuracies = []
    negative_accuracies = []
    for start in range(0, noise.pair_count, pair_chunk_size):
        chunk = slice_pair_noise(
            noise,
            start,
            min(start + pair_chunk_size, noise.pair_count),
        )
        logits = population_forward(model, inputs, chunk, sigma)
        losses = -logits.log_softmax(dim=-1).gather(
            dim=-1,
            index=targets[None, :, None].expand(chunk.population_size, -1, 1),
        ).squeeze(-1).mean(dim=-1)
        accuracies = logits.argmax(dim=-1).eq(targets).float().mean(dim=-1)
        positive_losses.append(losses[: chunk.pair_count])
        negative_losses.append(losses[chunk.pair_count :])
        positive_accuracies.append(accuracies[: chunk.pair_count])
        negative_accuracies.append(accuracies[chunk.pair_count :])
    return (
        torch.cat((*positive_losses, *negative_losses)),
        torch.cat((*positive_accuracies, *negative_accuracies)),
    )


def shape_fitness(losses: Tensor, mode: str = "zscore") -> Tensor:
    """Convert candidate losses into zero-centred maximization fitness."""

    if losses.ndim != 1 or losses.numel() < 2:
        raise ValueError("losses must contain a candidate population")
    rewards = -losses.float()
    if mode == "zscore":
        centered = rewards - rewards.mean()
        return centered / torch.sqrt(rewards.var(unbiased=False) + 1e-8)
    if mode == "centered-rank":
        order = rewards.argsort().argsort().to(rewards.dtype)
        return order / (len(rewards) - 1) - 0.5
    if mode == "centered":
        return rewards - rewards.mean()
    raise ValueError("mode must be 'zscore', 'centered-rank', or 'centered'")


def estimate_reward_gradients(
    noise: AntitheticNoise,
    fitness: Tensor,
) -> dict[str, Tensor]:
    """Estimate the reference fitness-weighted forward perturbation."""

    if fitness.shape != (noise.population_size,):
        raise ValueError("fitness must have one value per population member")
    pair_fitness = fitness[: noise.pair_count] - fitness[noise.pair_count :]
    denominator = noise.population_size
    gradients: dict[str, Tensor] = {}
    for name, factors in noise.matrices.items():
        gradients[name] = torch.einsum(
            "p,por,pir->oi",
            pair_fitness.to(factors.left.dtype),
            factors.left,
            factors.right,
        ) / (denominator * math.sqrt(noise.rank))
    for name, values in noise.vectors.items():
        broadcast_shape = (noise.pair_count,) + (1,) * (values.ndim - 1)
        gradients[name] = (
            pair_fitness.to(values.dtype).reshape(broadcast_shape) * values
        ).sum(dim=0) / denominator
    return gradients


def assign_maximization_gradients(
    model: nn.Module,
    reward_gradients: dict[str, Tensor],
) -> None:
    """Assign negative gradients so a standard optimizer maximizes fitness."""

    parameters = dict(model.named_parameters())
    if set(parameters) != set(reward_gradients):
        raise ValueError("gradient estimate does not match model parameters")
    for name, parameter in parameters.items():
        parameter.grad = -reward_gradients[name].detach()


def gradient_rms(gradients: dict[str, Tensor]) -> float:
    squared_sum = sum(value.float().square().sum() for value in gradients.values())
    count = sum(value.numel() for value in gradients.values())
    return float(torch.sqrt(squared_sum / count))


@torch.no_grad()
def materialize_candidate_parameters(
    model: nn.Module,
    noise: AntitheticNoise,
    candidate: int,
    sigma: float,
) -> dict[str, Tensor]:
    """Materialise one candidate for tests and mechanistic inspection."""

    if not 0 <= candidate < noise.population_size:
        raise ValueError("candidate index is outside the population")
    sign = 1.0 if candidate < noise.pair_count else -1.0
    pair = candidate % noise.pair_count
    result = {}
    for name, parameter in model.named_parameters():
        if name in noise.matrices:
            factors = noise.matrices[name]
            delta = factors.left[pair] @ factors.right[pair].transpose(0, 1)
            delta = delta / math.sqrt(noise.rank)
        else:
            delta = noise.vectors[name][pair]
        result[name] = parameter.detach() + sign * sigma * delta
    return result
