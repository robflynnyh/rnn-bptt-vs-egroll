import torch

from rnn_bptt_vs_eggroll.task import IGNORE_INDEX, MQARConfig, sample_batch


def test_batch_matches_zoology_mqar_layout() -> None:
    config = MQARConfig(vocab_size=32, random_non_queries=False)
    batch = sample_batch(
        5,
        sequence_length=16,
        num_kv_pairs=2,
        config=config,
        generator=torch.Generator().manual_seed(12),
    )

    assert batch.inputs.shape == (5, 16)
    assert batch.targets.shape == (5, 16)
    assert torch.equal(batch.inputs[:, :4:2], batch.keys)
    assert torch.equal(batch.inputs[:, 1:4:2], batch.values)
    assert torch.all(batch.keys > 0)
    assert torch.all(batch.keys < config.key_vocab_size)
    assert torch.all(batch.values >= config.key_vocab_size)
    assert torch.all(batch.targets.ne(IGNORE_INDEX).sum(dim=1) == 2)

    for row in range(5):
        assert len(batch.keys[row].unique()) == 2
        assert len(batch.values[row].unique()) == 2
        for pair in range(2):
            position = batch.query_positions[row, pair]
            assert batch.inputs[row, position] == batch.keys[row, pair]
            assert batch.targets[row, position] == batch.values[row, pair]


def test_sampling_is_reproducible() -> None:
    config = MQARConfig(vocab_size=64)
    first = sample_batch(3, 16, 2, config, generator=torch.Generator().manual_seed(99),)
    second = sample_batch(
        3, 16, 2, config, generator=torch.Generator().manual_seed(99),
    )
    assert torch.equal(first.inputs, second.inputs)
    assert torch.equal(first.targets, second.targets)


def test_random_non_queries_only_changes_filler_tokens() -> None:
    fixed = sample_batch(
        4,
        16,
        2,
        MQARConfig(vocab_size=64, random_non_queries=False),
        generator=torch.Generator().manual_seed(4),
    )
    random = sample_batch(
        4,
        16,
        2,
        MQARConfig(vocab_size=64, random_non_queries=True),
        generator=torch.Generator().manual_seed(4),
    )
    content_mask = fixed.inputs.ne(0)
    assert torch.equal(random.inputs[content_mask], fixed.inputs[content_mask])
    assert torch.equal(random.targets, fixed.targets)
    assert torch.any(random.inputs[~content_mask].ne(0))
