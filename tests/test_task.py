import torch

from rnn_bptt_vs_eggroll.task import AssociativeRecallConfig, sample_batch


def test_batch_encodes_and_queries_the_sampled_association() -> None:
    config = AssociativeRecallConfig(num_keys=6, num_values=7)
    batch = sample_batch(
        5,
        num_pairs=4,
        delay=9,
        config=config,
        generator=torch.Generator().manual_seed(12),
    )

    assert batch.inputs.shape == (5, 14, config.input_size)
    rows = torch.arange(5)
    assert torch.equal(
        batch.targets,
        batch.stored_values[rows, batch.query_indices],
    )
    assert torch.equal(
        batch.query_keys,
        batch.stored_keys[rows, batch.query_indices],
    )
    assert torch.all(batch.inputs[:, :4, config.store_flag] == 1)
    assert torch.all(batch.inputs[:, 4:-1, config.distractor_flag] == 1)
    assert torch.all(batch.inputs[:, -1, config.query_flag] == 1)

    for keys, values in zip(batch.stored_keys, batch.stored_values):
        assert len(keys.unique()) == 4
        assert len(values.unique()) == 4


def test_sampling_is_reproducible() -> None:
    config = AssociativeRecallConfig()
    first = sample_batch(
        3,
        2,
        5,
        config,
        generator=torch.Generator().manual_seed(99),
    )
    second = sample_batch(
        3,
        2,
        5,
        config,
        generator=torch.Generator().manual_seed(99),
    )
    assert torch.equal(first.inputs, second.inputs)
    assert torch.equal(first.targets, second.targets)


def test_distractor_noise_does_not_modify_control_flags() -> None:
    config = AssociativeRecallConfig(distractor_std=0.2)
    batch = sample_batch(
        2,
        2,
        3,
        config,
        generator=torch.Generator().manual_seed(4),
    )
    assert torch.any(batch.inputs[:, 2:5, : config.num_keys + config.num_values] != 0)
    assert torch.all(batch.inputs[:, 2:5, config.distractor_flag] == 1)
