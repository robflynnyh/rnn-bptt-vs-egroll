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
    model: nn.Module, population_size: int, rank: int, *, generator: torch.Generator,
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
        outputs = base.unsqueeze(0).expand(noise.population_size, -1, -1).clone()
        projected = torch.einsum("bi,pir->pbr", inputs, right)
    elif inputs.ndim == 3:
        outputs = base
        projected = torch.einsum("pbi,pir->pbr", inputs, right)
    else:
        raise ValueError("population affine input must be [B, D] or [P, B, D]")
    scale = sigma / math.sqrt(noise.rank)
    for rank_index in range(noise.rank):
        outputs.addcmul_(
            projected[:, :, rank_index, None],
            left[:, None, :, rank_index],
            value=scale,
        )
    if bias is not None:
        if bias_name is None:
            raise ValueError("bias_name is required when a bias is provided")
        outputs.add_(_signed_pair(noise.vectors[bias_name])[:, None], alpha=sigma)
    return outputs


def _population_embedding(
    token_ids: Tensor,
    weight: Tensor,
    weight_name: str,
    noise: AntitheticNoise,
    sigma: float,
    *,
    candidate_inputs: bool = False,
) -> Tensor:
    """Embed token IDs for every low-rank population member."""

    if token_ids.ndim != 1 or token_ids.dtype != torch.long:
        raise ValueError("token_ids must have shape [batch]")
    factors = noise.matrices[weight_name]
    left = _signed_pair(factors.left)
    right = _repeated_pair(factors.right)
    base = F.embedding(token_ids, weight.transpose(0, 1))
    if candidate_inputs:
        if token_ids.shape[0] != noise.population_size:
            raise ValueError("candidate token inputs must match the population")
        rows = torch.arange(noise.population_size, device=token_ids.device)
        selected_right = right[rows, token_ids, :]
        outputs = base[:, None, :]
    else:
        selected_right = right[:, token_ids, :]
        outputs = base.unsqueeze(0).expand(noise.population_size, -1, -1).clone()
    scale = sigma / math.sqrt(noise.rank)
    for rank_index in range(noise.rank):
        if candidate_inputs:
            outputs.addcmul_(
                selected_right[:, rank_index, None, None],
                left[:, None, :, rank_index],
                value=scale,
            )
        else:
            outputs.addcmul_(
                selected_right[:, :, rank_index, None],
                left[:, None, :, rank_index],
                value=scale,
            )
    return outputs


@torch.no_grad()
def population_forward(
    model: VanillaRNN,
    inputs: Tensor,
    noise: AntitheticNoise,
    sigma: float,
    *,
    readout_mask: Tensor | None = None,
) -> Tensor:
    """Evaluate population logits at the final or selected MQAR positions."""

    if inputs.ndim != 2 or inputs.dtype != torch.long:
        raise ValueError("inputs must be integer token IDs with shape [batch, time]")
    if readout_mask is not None and readout_mask.shape != inputs.shape:
        raise ValueError("readout_mask must have the same shape as inputs")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    hidden = model.input_weight.new_zeros(
        noise.population_size, inputs.shape[0], model.hidden_size,
    )
    selected_logits = []
    for time, token_ids in enumerate(inputs.unbind(dim=1)):
        input_term = _population_embedding(
            token_ids, model.input_weight, "input_weight", noise, sigma,
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
        if readout_mask is not None and readout_mask[:, time].any():
            selected_logits.append(
                _population_affine(
                    hidden[:, readout_mask[:, time]],
                    model.output_weight,
                    "output_weight",
                    noise,
                    sigma,
                    bias=model.output_bias,
                    bias_name="output_bias",
                )
            )
    if readout_mask is not None:
        if not selected_logits:
            raise ValueError("readout_mask must select at least one position")
        return torch.cat(selected_logits, dim=1)
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
                left=factors.left[start:end], right=factors.right[start:end],
            )
            for name, factors in noise.matrices.items()
        },
        vectors={name: values[start:end] for name, values in noise.vectors.items()},
        rank=noise.rank,
    )


def _cast_noise(noise: AntitheticNoise, dtype: torch.dtype) -> AntitheticNoise:
    return AntitheticNoise(
        matrices={
            name: MatrixFactors(
                left=factors.left.to(dtype), right=factors.right.to(dtype),
            )
            for name, factors in noise.matrices.items()
        },
        vectors={name: values.to(dtype) for name, values in noise.vectors.items()},
        rank=noise.rank,
    )


def _population_readout_sums(
    model: VanillaRNN,
    states: Tensor,
    targets: Tensor,
    noise: AntitheticNoise,
    sigma: float,
) -> tuple[Tensor, Tensor]:
    logits = _population_affine(
        states,
        model.output_weight,
        "output_weight",
        noise,
        sigma,
        bias=model.output_bias,
        bias_name="output_bias",
    ).float()
    target_logits = logits.gather(
        -1,
        targets[None, :, None].expand(noise.population_size, -1, 1),
    ).squeeze(-1)
    losses = (torch.logsumexp(logits, dim=-1) - target_logits).sum(dim=-1)
    correct = logits.argmax(dim=-1).eq(targets).sum(dim=-1)
    return losses, correct


@torch.no_grad()
def _grouped_population_loss_and_accuracy(
    model: VanillaRNN,
    inputs: Tensor,
    targets: Tensor,
    noise: AntitheticNoise,
    sigma: float,
) -> tuple[Tensor, Tensor]:
    """Evaluate one candidate-specific sequence per population member."""

    if inputs.shape != targets.shape or inputs.shape[0] != noise.population_size:
        raise ValueError("grouped inputs and targets must match the population")
    readout_mask = targets.ne(-100)
    counts = readout_mask.sum(dim=1)
    if not counts.numel() or int(counts.min()) < 1 or not counts.eq(counts[0]).all():
        raise ValueError("grouped candidates must have equal nonzero target counts")
    query_count = int(counts[0])
    selected_times = readout_mask.any(dim=0)
    selected_time_flags = selected_times.tolist()
    time_to_slot = selected_times.cumsum(dim=0) - 1
    hidden = model.input_weight.new_zeros(
        noise.population_size, 1, model.hidden_size,
    )
    selected_states = []
    for time, token_ids in enumerate(inputs.unbind(dim=1)):
        input_term = _population_embedding(
            token_ids,
            model.input_weight,
            "input_weight",
            noise,
            sigma,
            candidate_inputs=True,
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
        if selected_time_flags[time]:
            selected_states.append(hidden[:, 0])
    stacked_states = torch.stack(selected_states, dim=1)
    readout_positions = readout_mask.nonzero(as_tuple=False)[:, 1].reshape(
        noise.population_size, query_count,
    )
    readout_slots = time_to_slot[readout_positions]
    candidate_rows = torch.arange(noise.population_size, device=inputs.device)[:, None]
    readout_states = stacked_states[candidate_rows, readout_slots]
    readout_targets = targets[readout_mask].reshape(noise.population_size, query_count)
    loss_sums = model.input_weight.new_zeros(noise.population_size)
    correct_counts = model.input_weight.new_zeros(noise.population_size)
    for start in range(0, query_count, 8):
        end = min(start + 8, query_count)
        logits = _population_affine(
            readout_states[:, start:end],
            model.output_weight,
            "output_weight",
            noise,
            sigma,
            bias=model.output_bias,
            bias_name="output_bias",
        ).float()
        selected_targets = readout_targets[:, start:end]
        target_logits = logits.gather(-1, selected_targets[..., None]).squeeze(-1)
        loss_sums += (torch.logsumexp(logits, dim=-1) - target_logits).sum(dim=-1)
        correct_counts += logits.argmax(dim=-1).eq(selected_targets).sum(dim=-1)
    return loss_sums / query_count, correct_counts / query_count


@torch.no_grad()
def _population_loss_and_accuracy(
    model: VanillaRNN,
    inputs: Tensor,
    targets: Tensor,
    noise: AntitheticNoise,
    sigma: float,
) -> tuple[Tensor, Tensor]:
    """Accumulate exact query metrics without retaining every readout logit."""

    hidden = model.input_weight.new_zeros(
        noise.population_size, inputs.shape[0], model.hidden_size,
    )
    loss_sums = model.input_weight.new_zeros(noise.population_size)
    correct_counts = model.input_weight.new_zeros(noise.population_size)
    supervised_count = 0
    buffered_count = 0
    buffered_states: list[Tensor] = []
    buffered_targets: list[Tensor] = []

    def flush_readouts() -> None:
        nonlocal buffered_count, loss_sums, correct_counts
        states = torch.cat(buffered_states, dim=1)
        selected_targets = torch.cat(buffered_targets)
        losses, correct = _population_readout_sums(
            model, states, selected_targets, noise, sigma,
        )
        loss_sums += losses
        correct_counts += correct
        buffered_states.clear()
        buffered_targets.clear()
        buffered_count = 0

    for time, token_ids in enumerate(inputs.unbind(dim=1)):
        input_term = _population_embedding(
            token_ids, model.input_weight, "input_weight", noise, sigma,
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

        selected = targets[:, time].ne(-100)
        if not selected.any():
            continue
        selected_targets = targets[:, time][selected]
        buffered_states.append(hidden[:, selected])
        buffered_targets.append(selected_targets)
        buffered_count += selected_targets.numel()
        supervised_count += selected_targets.numel()
        if buffered_count >= 128:
            flush_readouts()

    if supervised_count == 0:
        raise ValueError("targets must select at least one supervised position")
    if buffered_count:
        flush_readouts()
    return loss_sums / supervised_count, correct_counts / supervised_count


@torch.no_grad()
def evaluate_population(
    model: VanillaRNN,
    inputs: Tensor,
    targets: Tensor,
    noise: AntitheticNoise,
    sigma: float,
    *,
    candidate_chunk_size: int,
    data_mode: str = "cartesian",
    precision: str = "float32",
) -> tuple[Tensor, Tensor]:
    """Return per-candidate loss and accuracy, chunking complete +/- pairs."""

    if candidate_chunk_size < 2 or candidate_chunk_size % 2:
        raise ValueError("candidate_chunk_size must be positive and even")
    if data_mode not in {"cartesian", "grouped"}:
        raise ValueError("data_mode must be cartesian or grouped")
    if precision not in {"float32", "bfloat16"}:
        raise ValueError("precision must be float32 or bfloat16")
    if data_mode == "grouped" and noise.pair_count % inputs.shape[0]:
        raise ValueError("antithetic pairs must divide evenly across grouped examples")
    device_type = model.input_weight.device.type
    if (
        precision == "bfloat16"
        and device_type == "cuda"
        and not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("CUDA device does not support bfloat16")
    pair_chunk_size = candidate_chunk_size // 2
    positive_losses = []
    negative_losses = []
    positive_accuracies = []
    negative_accuracies = []
    for start in range(0, noise.pair_count, pair_chunk_size):
        end = min(start + pair_chunk_size, noise.pair_count)
        chunk = slice_pair_noise(
            noise, start, end,
        )
        if precision == "bfloat16":
            chunk = _cast_noise(chunk, torch.bfloat16)
        if data_mode == "grouped":
            pair_examples = torch.arange(start, end, device=inputs.device)
            pair_examples.remainder_(inputs.shape[0])
            signed_examples = torch.cat((pair_examples, pair_examples))
            chunk_inputs = inputs[signed_examples]
            chunk_targets = targets[signed_examples]
        else:
            chunk_inputs = inputs
            chunk_targets = targets
        with torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=precision == "bfloat16",
        ):
            if data_mode == "grouped":
                losses, accuracies = _grouped_population_loss_and_accuracy(
                    model, chunk_inputs, chunk_targets, chunk, sigma,
                )
            else:
                losses, accuracies = _population_loss_and_accuracy(
                    model, chunk_inputs, chunk_targets, chunk, sigma,
                )
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
    if mode == "antithetic-sign":
        if losses.numel() % 2:
            raise ValueError("antithetic-sign requires an even population")
        pair_count = losses.numel() // 2
        # Candidates are ordered as [+E_1, ..., +E_n, -E_1, ..., -E_n].
        # Lower loss is better, so a positive vote means moving toward +E.
        votes = torch.sign(losses[pair_count:] - losses[:pair_count]).float()
        return torch.cat((votes, -votes))
    raise ValueError(
        "mode must be 'zscore', 'centered-rank', 'centered', or "
        "'antithetic-sign'"
    )


def estimate_reward_gradients(
    noise: AntitheticNoise, fitness: Tensor,
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


def estimate_elite_centroid_directions(
    noise: AntitheticNoise,
    losses: Tensor,
    *,
    elite_count: int,
) -> tuple[dict[str, Tensor], Tensor]:
    """Average the winning signs from the best unique antithetic directions."""

    if losses.shape != (noise.population_size,):
        raise ValueError("losses must have one value per population member")
    if not 1 <= elite_count <= noise.pair_count:
        raise ValueError("elite_count must fit the antithetic directions")
    positive_losses = losses[: noise.pair_count]
    negative_losses = losses[noise.pair_count :]
    prefer_positive = positive_losses <= negative_losses
    preferred_losses = torch.minimum(positive_losses, negative_losses)
    elite_pairs = preferred_losses.topk(elite_count, largest=False).indices
    elite_signs = torch.where(
        prefer_positive[elite_pairs],
        torch.ones_like(preferred_losses[elite_pairs]),
        -torch.ones_like(preferred_losses[elite_pairs]),
    )
    directions: dict[str, Tensor] = {}
    for name, factors in noise.matrices.items():
        directions[name] = torch.einsum(
            "p,por,pir->oi",
            elite_signs.to(factors.left.dtype),
            factors.left[elite_pairs],
            factors.right[elite_pairs],
        ) / (elite_count * math.sqrt(noise.rank))
    for name, values in noise.vectors.items():
        broadcast_shape = (elite_count,) + (1,) * (values.ndim - 1)
        directions[name] = (
            elite_signs.to(values.dtype).reshape(broadcast_shape)
            * values[elite_pairs]
        ).mean(dim=0)
    selected_candidates = torch.where(
        prefer_positive[elite_pairs],
        elite_pairs,
        elite_pairs + noise.pair_count,
    )
    return directions, selected_candidates


def assign_maximization_gradients(
    model: nn.Module, reward_gradients: dict[str, Tensor],
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
    model: nn.Module, noise: AntitheticNoise, candidate: int, sigma: float,
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
