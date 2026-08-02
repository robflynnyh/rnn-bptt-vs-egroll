import torch
import torch.nn.functional as F

from rnn_bptt_vs_eggroll.model import VanillaRNN


def test_model_returns_selected_logits_and_states() -> None:
    torch.manual_seed(3)
    model = VanillaRNN(17, 7, recurrent_radius=0.8)
    inputs = torch.randint(17, (4, 11))
    readout_mask = torch.zeros_like(inputs, dtype=torch.bool)
    readout_mask[:, (3, 9)] = True
    logits, states = model(inputs, readout_mask=readout_mask, return_states=True,)
    assert logits.shape == (8, 17)
    assert states.shape == (4, 11, 7)

    loss = F.cross_entropy(logits, torch.arange(8) % 17)
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_recurrent_initialization_has_requested_singular_values() -> None:
    model = VanillaRNN(11, 6, recurrent_radius=0.73)
    singular_values = torch.linalg.svdvals(model.recurrent_weight.detach())
    assert torch.allclose(singular_values, torch.full_like(singular_values, 0.73))


def test_tied_model_reuses_input_weight_for_output_and_gradients() -> None:
    torch.manual_seed(4)
    model = VanillaRNN(17, 7, tie_input_output=True)
    inputs = torch.randint(17, (4, 8))
    readout_mask = torch.zeros_like(inputs, dtype=torch.bool)
    readout_mask[:, 5] = True

    logits, states = model(inputs, readout_mask=readout_mask, return_states=True)
    expected = F.linear(
        states[:, 5], model.input_weight.transpose(0, 1), model.output_bias,
    )

    assert "output_weight" not in dict(model.named_parameters())
    assert torch.equal(logits, expected)
    F.cross_entropy(logits, torch.arange(4) % 17).backward()
    assert model.input_weight.grad is not None
    assert torch.isfinite(model.input_weight.grad).all()
