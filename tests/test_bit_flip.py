import torch

from rnn_bptt_vs_eggroll.bit_flip import (
    FLIP,
    HOLD,
    QUERY,
    BitFlipRNN,
    build_curriculum,
    sample_bit_flip_batch,
)


def test_bit_flip_targets_follow_all_operations() -> None:
    batch = sample_bit_flip_batch(
        64,
        7,
        generator=torch.Generator().manual_seed(11),
    )
    start_bits = batch.inputs[:, 0]
    flips = batch.inputs[:, 1:-1].eq(FLIP).sum(dim=1).remainder(2)
    assert torch.equal(batch.targets, start_bits.bitwise_xor(flips))
    assert batch.inputs.shape == (64, 9)
    assert batch.inputs[:, 1:-1].ge(HOLD).all()
    assert batch.inputs[:, 1:-1].le(FLIP).all()
    assert batch.inputs[:, -1].eq(QUERY).all()


def test_bit_flip_sampling_is_deterministic() -> None:
    first = sample_bit_flip_batch(
        8,
        4,
        generator=torch.Generator().manual_seed(3),
    )
    second = sample_bit_flip_batch(
        8,
        4,
        generator=torch.Generator().manual_seed(3),
    )
    assert torch.equal(first.inputs, second.inputs)
    assert torch.equal(first.targets, second.targets)


def test_curriculum_is_dense_then_geometric_and_bounded() -> None:
    curriculum = build_curriculum(dense_until=4, maximum=10, growth=1.5)
    assert curriculum == (1, 2, 3, 4, 6, 9, 10)


def test_bit_flip_rnn_has_binary_terminal_readout() -> None:
    model = BitFlipRNN(hidden_size=8)
    batch = sample_bit_flip_batch(
        5,
        3,
        generator=torch.Generator().manual_seed(9),
    )
    logits = model(batch.inputs)
    assert logits.shape == (5, 2)
    assert model.parameter_count == sum(p.numel() for p in model.parameters())
