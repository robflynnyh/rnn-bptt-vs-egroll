import torch
import torch.nn.functional as F

from rnn_bptt_vs_eggroll.eggroll import (
    evaluate_population,
    materialize_candidate_parameters,
    population_forward,
    sample_antithetic_noise,
    shape_fitness,
)
from rnn_bptt_vs_eggroll.model import VanillaRNN, functional_rnn_forward


def test_population_forward_matches_materialized_candidates() -> None:
    torch.manual_seed(10)
    model = VanillaRNN(11, 4)
    inputs = torch.randint(11, (2, 6))
    noise = sample_antithetic_noise(
        model, population_size=6, rank=2, generator=torch.Generator().manual_seed(11),
    )
    sigma = 0.03

    population_logits = population_forward(model, inputs, noise, sigma)
    explicit_logits = torch.stack(
        [
            functional_rnn_forward(
                inputs, materialize_candidate_parameters(model, noise, index, sigma),
            )
            for index in range(noise.population_size)
        ]
    )
    assert torch.allclose(population_logits, explicit_logits, atol=2e-6, rtol=2e-5)


def test_chunked_population_preserves_antithetic_order() -> None:
    torch.manual_seed(5)
    model = VanillaRNN(7, 3)
    inputs = torch.randint(7, (3, 4))
    targets = torch.full((3, 4), -100)
    targets[0, 1] = 0
    targets[1, 2] = 1
    targets[2, 1] = 0
    noise = sample_antithetic_noise(
        model, population_size=8, rank=1, generator=torch.Generator().manual_seed(6),
    )
    losses, accuracies = evaluate_population(
        model, inputs, targets, noise, 0.02, candidate_chunk_size=4,
    )
    readout_mask = targets.ne(-100)
    logits = population_forward(model, inputs, noise, 0.02, readout_mask=readout_mask,)
    supervised_targets = torch.cat(
        [targets[:, time][readout_mask[:, time]] for time in range(targets.shape[1])]
    )
    expected_losses = (
        -logits.log_softmax(dim=-1)
        .gather(-1, supervised_targets[None, :, None].expand(8, -1, 1),)
        .squeeze(-1)
        .mean(dim=-1)
    )
    expected_accuracies = (
        logits.argmax(dim=-1).eq(supervised_targets).float().mean(dim=-1)
    )
    assert torch.allclose(losses, expected_losses)
    assert torch.equal(accuracies, expected_accuracies)


def test_grouped_population_matches_materialized_candidates() -> None:
    torch.manual_seed(21)
    model = VanillaRNN(7, 3)
    inputs = torch.randint(7, (2, 6))
    targets = torch.full((2, 6), -100)
    targets[0, 2] = 3
    targets[1, 4] = 5
    noise = sample_antithetic_noise(
        model, population_size=8, rank=1, generator=torch.Generator().manual_seed(22),
    )
    sigma = 0.02

    losses, accuracies = evaluate_population(
        model,
        inputs,
        targets,
        noise,
        sigma,
        candidate_chunk_size=4,
        data_mode="grouped",
    )
    expected_losses = []
    expected_accuracies = []
    for candidate in range(noise.population_size):
        pair = candidate % noise.pair_count
        example = pair % inputs.shape[0]
        mask = targets[example : example + 1].ne(-100)
        selected_targets = targets[example : example + 1][mask]
        logits = functional_rnn_forward(
            inputs[example : example + 1],
            materialize_candidate_parameters(model, noise, candidate, sigma),
            readout_mask=mask,
        )
        expected_losses.append(F.cross_entropy(logits, selected_targets))
        expected_accuracies.append(
            logits.argmax(dim=-1).eq(selected_targets).float().mean()
        )

    assert torch.allclose(losses, torch.stack(expected_losses), atol=2e-6, rtol=2e-5)
    assert torch.equal(accuracies, torch.stack(expected_accuracies))


def test_antithetic_candidates_are_symmetric() -> None:
    model = VanillaRNN(9, 4)
    noise = sample_antithetic_noise(
        model, population_size=4, rank=1, generator=torch.Generator().manual_seed(8),
    )
    plus = materialize_candidate_parameters(model, noise, 0, 0.1)
    minus = materialize_candidate_parameters(model, noise, 2, 0.1)
    for name, parameter in model.named_parameters():
        assert torch.allclose(plus[name] + minus[name], 2 * parameter)


def test_zscore_fitness_is_zero_mean_and_unit_variance() -> None:
    fitness = shape_fitness(torch.tensor([3.0, 1.0, 2.0, 9.0]), "zscore")
    assert abs(float(fitness.mean())) < 1e-6
    assert torch.allclose(fitness.var(unbiased=False), torch.tensor(1.0))


def test_antithetic_sign_fitness_keeps_only_pairwise_winner() -> None:
    # First half is +E, second half is -E. Lower loss wins.
    losses = torch.tensor([1.0, 4.0, 2.0, 3.0, 2.0, 2.0])
    fitness = shape_fitness(losses, "antithetic-sign")
    assert torch.equal(fitness, torch.tensor([1.0, -1.0, 0.0, -1.0, 1.0, -0.0]))
