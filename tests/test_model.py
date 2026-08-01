import torch
import torch.nn.functional as F

from rnn_bptt_vs_eggroll.model import VanillaRNN


def test_model_returns_final_logits_and_states() -> None:
    torch.manual_seed(3)
    model = VanillaRNN(9, 7, 5, recurrent_radius=0.8)
    inputs = torch.randn(4, 11, 9)
    logits, states = model(inputs, return_states=True)
    assert logits.shape == (4, 5)
    assert states.shape == (4, 11, 7)

    loss = F.cross_entropy(logits, torch.tensor([0, 1, 2, 3]))
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_recurrent_initialization_has_requested_singular_values() -> None:
    model = VanillaRNN(5, 6, 3, recurrent_radius=0.73)
    singular_values = torch.linalg.svdvals(model.recurrent_weight.detach())
    assert torch.allclose(singular_values, torch.full_like(singular_values, 0.73))
