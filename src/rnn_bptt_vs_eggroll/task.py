"""Synthetic dynamic key-value associative recall."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class AssociativeRecallConfig:
    """Vocabulary and distractor settings shared by every task split."""

    num_keys: int = 8
    num_values: int = 8
    distractor_std: float = 0.0

    def __post_init__(self) -> None:
        if self.num_keys < 2 or self.num_values < 2:
            raise ValueError("num_keys and num_values must both be at least two")
        if self.distractor_std < 0:
            raise ValueError("distractor_std must be non-negative")

    @property
    def input_size(self) -> int:
        # Concatenated key/value slots plus STORE, DISTRACTOR, and QUERY flags.
        return self.num_keys + self.num_values + 3

    @property
    def store_flag(self) -> int:
        return self.num_keys + self.num_values

    @property
    def distractor_flag(self) -> int:
        return self.store_flag + 1

    @property
    def query_flag(self) -> int:
        return self.store_flag + 2


@dataclass(frozen=True)
class AssociativeRecallBatch:
    """A batch plus metadata that makes shortcut checks straightforward."""

    inputs: Tensor
    targets: Tensor
    stored_keys: Tensor
    stored_values: Tensor
    query_indices: Tensor
    query_keys: Tensor
    delay: int

    def to(self, device: torch.device | str) -> "AssociativeRecallBatch":
        return AssociativeRecallBatch(
            inputs=self.inputs.to(device),
            targets=self.targets.to(device),
            stored_keys=self.stored_keys.to(device),
            stored_values=self.stored_values.to(device),
            query_indices=self.query_indices.to(device),
            query_keys=self.query_keys.to(device),
            delay=self.delay,
        )


def _sample_without_replacement(
    batch_size: int,
    vocabulary_size: int,
    count: int,
    *,
    generator: torch.Generator,
) -> Tensor:
    scores = torch.rand(batch_size, vocabulary_size, generator=generator)
    return scores.argsort(dim=-1)[:, :count]


def sample_batch(
    batch_size: int,
    num_pairs: int,
    delay: int,
    config: AssociativeRecallConfig,
    *,
    generator: torch.Generator,
) -> AssociativeRecallBatch:
    """Sample random one-to-one associations and query one stored key.

    Sequence layout is ``STORE(key, value) * num_pairs``, followed by ``delay``
    distractor steps and one ``QUERY(key)`` step. Only the final-step value is
    supervised. Keys and values are unique within an example, preventing
    frequency and duplicate-value shortcuts.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_pairs < 1:
        raise ValueError("num_pairs must be positive")
    if num_pairs > min(config.num_keys, config.num_values):
        raise ValueError("num_pairs cannot exceed either vocabulary size")
    if delay < 0:
        raise ValueError("delay must be non-negative")

    stored_keys = _sample_without_replacement(
        batch_size,
        config.num_keys,
        num_pairs,
        generator=generator,
    )
    stored_values = _sample_without_replacement(
        batch_size,
        config.num_values,
        num_pairs,
        generator=generator,
    )
    query_indices = torch.randint(
        num_pairs,
        (batch_size,),
        generator=generator,
    )
    rows = torch.arange(batch_size)
    query_keys = stored_keys[rows, query_indices]
    targets = stored_values[rows, query_indices]

    sequence_length = num_pairs + delay + 1
    inputs = torch.zeros(batch_size, sequence_length, config.input_size)
    pair_positions = torch.arange(num_pairs)
    pair_rows = rows[:, None].expand(-1, num_pairs)
    pair_steps = pair_positions[None, :].expand(batch_size, -1)
    inputs[pair_rows, pair_steps, stored_keys] = 1.0
    inputs[
        pair_rows,
        pair_steps,
        config.num_keys + stored_values,
    ] = 1.0
    inputs[:, :num_pairs, config.store_flag] = 1.0

    if delay:
        delay_slice = slice(num_pairs, num_pairs + delay)
        inputs[:, delay_slice, config.distractor_flag] = 1.0
        if config.distractor_std:
            content_noise = torch.randn(
                batch_size,
                delay,
                config.num_keys + config.num_values,
                generator=generator,
            )
            inputs[:, delay_slice, : config.num_keys + config.num_values] = (
                config.distractor_std * content_noise
            )

    inputs[rows, -1, query_keys] = 1.0
    inputs[:, -1, config.query_flag] = 1.0
    return AssociativeRecallBatch(
        inputs=inputs,
        targets=targets,
        stored_keys=stored_keys,
        stored_values=stored_values,
        query_indices=query_indices,
        query_keys=query_keys,
        delay=delay,
    )
