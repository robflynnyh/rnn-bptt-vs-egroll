import torch

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
    model = VanillaRNN(5, 4, 3)
    inputs = torch.randn(2, 6, 5)
    noise = sample_antithetic_noise(
        model,
        population_size=6,
        rank=2,
        generator=torch.Generator().manual_seed(11),
    )
    sigma = 0.03

    population_logits = population_forward(model, inputs, noise, sigma)
    explicit_logits = torch.stack(
        [
            functional_rnn_forward(
                inputs,
                materialize_candidate_parameters(model, noise, index, sigma),
            )
            for index in range(noise.population_size)
        ]
    )
    assert torch.allclose(population_logits, explicit_logits, atol=2e-6, rtol=2e-5)


def test_chunked_population_preserves_antithetic_order() -> None:
    torch.manual_seed(5)
    model = VanillaRNN(4, 3, 2)
    inputs = torch.randn(3, 4, 4)
    targets = torch.tensor([0, 1, 0])
    noise = sample_antithetic_noise(
        model,
        population_size=8,
        rank=1,
        generator=torch.Generator().manual_seed(6),
    )
    losses, accuracies = evaluate_population(
        model,
        inputs,
        targets,
        noise,
        0.02,
        candidate_chunk_size=4,
    )
    logits = population_forward(model, inputs, noise, 0.02)
    expected_losses = -logits.log_softmax(dim=-1).gather(
        -1,
        targets[None, :, None].expand(8, -1, 1),
    ).squeeze(-1).mean(dim=-1)
    expected_accuracies = logits.argmax(dim=-1).eq(targets).float().mean(dim=-1)
    assert torch.allclose(losses, expected_losses)
    assert torch.equal(accuracies, expected_accuracies)


def test_antithetic_candidates_are_symmetric() -> None:
    model = VanillaRNN(3, 4, 2)
    noise = sample_antithetic_noise(
        model,
        population_size=4,
        rank=1,
        generator=torch.Generator().manual_seed(8),
    )
    plus = materialize_candidate_parameters(model, noise, 0, 0.1)
    minus = materialize_candidate_parameters(model, noise, 2, 0.1)
    for name, parameter in model.named_parameters():
        assert torch.allclose(plus[name] + minus[name], 2 * parameter)


def test_zscore_fitness_is_zero_mean_and_unit_variance() -> None:
    fitness = shape_fitness(torch.tensor([3.0, 1.0, 2.0, 9.0]), "zscore")
    assert abs(float(fitness.mean())) < 1e-6
    assert torch.allclose(fitness.var(unbiased=False), torch.tensor(1.0))
