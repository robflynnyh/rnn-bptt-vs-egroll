"""Zoology-style multi-query associative recall."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


IGNORE_INDEX = -100


@dataclass(frozen=True)
class MQARConfig:
    """Vocabulary and query-gap settings from the Zoology MQAR benchmark."""

    vocab_size: int = 8_192
    power_a: float = 0.01
    random_non_queries: bool = False

    def __post_init__(self) -> None:
        if self.vocab_size < 8 or self.vocab_size % 2:
            raise ValueError("vocab_size must be an even integer of at least 8")
        if self.power_a <= 0:
            raise ValueError("power_a must be positive")

    @property
    def key_vocab_size(self) -> int:
        return self.vocab_size // 2


@dataclass(frozen=True)
class MQARBatch:
    """A token batch with labels only at MQAR query positions."""

    inputs: Tensor
    targets: Tensor
    keys: Tensor
    values: Tensor
    query_positions: Tensor
    sequence_length: int
    num_kv_pairs: int

    def to(self, device: torch.device | str) -> "MQARBatch":
        return MQARBatch(
            inputs=self.inputs.to(device),
            targets=self.targets.to(device),
            keys=self.keys.to(device),
            values=self.values.to(device),
            query_positions=self.query_positions.to(device),
            sequence_length=self.sequence_length,
            num_kv_pairs=self.num_kv_pairs,
        )


def _sample_without_replacement(
    batch_size: int, choices: int, count: int, *, generator: torch.Generator,
) -> Tensor:
    scores = torch.rand(batch_size, choices, generator=generator)
    return scores.topk(count, dim=-1, largest=False).indices


def sample_batch(
    batch_size: int,
    sequence_length: int,
    num_kv_pairs: int,
    config: MQARConfig,
    *,
    generator: torch.Generator,
) -> MQARBatch:
    """Generate MQAR examples following Zoology's released construction.

    Each sequence starts with alternating unique key/value tokens. Every key is
    then placed once as a query at a power-law sampled even position in the
    remaining sequence. Labels are ignored everywhere except query positions.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if sequence_length < 4 or sequence_length % 2:
        raise ValueError("sequence_length must be an even integer of at least 4")
    if num_kv_pairs < 1:
        raise ValueError("num_kv_pairs must be positive")
    if 4 * num_kv_pairs > sequence_length:
        raise ValueError("sequence_length must provide context and query slots")
    if num_kv_pairs >= config.key_vocab_size:
        raise ValueError("num_kv_pairs exceeds the available key vocabulary")
    if config.vocab_size <= sequence_length:
        raise ValueError("Zoology MQAR requires vocab_size > sequence_length")

    key_choice_count = config.key_vocab_size - 1
    keys = 1 + _sample_without_replacement(
        batch_size, key_choice_count, num_kv_pairs, generator=generator,
    )
    values = config.key_vocab_size + _sample_without_replacement(
        batch_size, config.key_vocab_size, num_kv_pairs, generator=generator,
    )

    context_size = 2 * num_kv_pairs
    inputs = torch.zeros(batch_size, sequence_length, dtype=torch.long)
    inputs[:, :context_size:2] = keys
    inputs[:, 1:context_size:2] = values

    query_space = (sequence_length - context_size) // 2
    distances = torch.arange(1, query_space + 1, dtype=torch.float64)
    probabilities = config.power_a * distances.pow(config.power_a - 1)
    probabilities /= probabilities.sum()
    query_slots = torch.multinomial(
        probabilities.expand(batch_size, -1),
        num_kv_pairs,
        replacement=False,
        generator=generator,
    )
    query_positions = context_size + 2 * query_slots
    inputs.scatter_(1, query_positions, keys)

    targets = torch.full_like(inputs, IGNORE_INDEX)
    targets.scatter_(1, query_positions, values)

    if config.random_non_queries:
        filler_mask = inputs.eq(0)
        random_tokens = torch.randint(
            config.vocab_size, inputs.shape, generator=generator,
        )
        inputs[filler_mask] = random_tokens[filler_mask]

    return MQARBatch(
        inputs=inputs,
        targets=targets,
        keys=keys,
        values=values,
        query_positions=query_positions,
        sequence_length=sequence_length,
        num_kv_pairs=num_kv_pairs,
    )


def sample_dense_recall_batch(
    batch_size: int,
    num_kv_pairs: int,
    config: MQARConfig,
    *,
    generator: torch.Generator,
) -> MQARBatch:
    """Generate context and a query; the target is the omitted final answer token."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_kv_pairs < 1:
        raise ValueError("num_kv_pairs must be positive")
    if num_kv_pairs >= config.key_vocab_size:
        raise ValueError("num_kv_pairs exceeds the available key vocabulary")

    key_choice_count = config.key_vocab_size - 1
    keys = 1 + _sample_without_replacement(
        batch_size, key_choice_count, num_kv_pairs, generator=generator,
    )
    values = config.key_vocab_size + _sample_without_replacement(
        batch_size, config.key_vocab_size, num_kv_pairs, generator=generator,
    )
    query_indices = torch.randint(
        num_kv_pairs, (batch_size,), generator=generator,
    )
    batch_indices = torch.arange(batch_size)
    query_keys = keys[batch_indices, query_indices]
    query_values = values[batch_indices, query_indices]

    sequence_length = 2 * num_kv_pairs + 1
    inputs = torch.empty(batch_size, sequence_length, dtype=torch.long)
    inputs[:, : 2 * num_kv_pairs : 2] = keys
    inputs[:, 1 : 2 * num_kv_pairs : 2] = values
    inputs[:, -1] = query_keys

    targets = torch.full_like(inputs, IGNORE_INDEX)
    targets[:, -1] = query_values
    query_positions = torch.full(
        (batch_size, 1), sequence_length - 1, dtype=torch.long,
    )
    return MQARBatch(
        inputs=inputs,
        targets=targets,
        keys=keys,
        values=values,
        query_positions=query_positions,
        sequence_length=sequence_length,
        num_kv_pairs=num_kv_pairs,
    )
