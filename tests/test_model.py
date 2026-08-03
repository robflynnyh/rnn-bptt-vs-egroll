import torch
import torch.nn.functional as F

from rnn_bptt_vs_eggroll.model import ProductTokenLSTM, TokenLSTM, VanillaRNN


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


def test_lstm_matches_selected_logit_interface_and_tied_readout() -> None:
    torch.manual_seed(5)
    model = TokenLSTM(17, 7, tie_input_output=True)
    inputs = torch.randint(17, (4, 8))
    readout_mask = torch.zeros_like(inputs, dtype=torch.bool)
    readout_mask[:, 6] = True

    logits, states = model(inputs, readout_mask=readout_mask, return_states=True)
    expected = F.linear(
        states[:, 6], model.input_weight.transpose(0, 1), model.output_bias,
    )

    assert logits.shape == (4, 17)
    assert states.shape == (4, 8, 7)
    assert torch.equal(logits, expected)
    F.cross_entropy(logits, torch.arange(4) % 17).backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_lstm_recurrence_matches_torch_lstm() -> None:
    torch.manual_seed(19)
    vocab_size = 31
    hidden_size = 11
    model = TokenLSTM(vocab_size, hidden_size, tie_input_output=True)
    reference = torch.nn.LSTM(hidden_size, hidden_size, batch_first=True)
    with torch.no_grad():
        reference.weight_ih_l0.copy_(model.lstm_input_weight)
        reference.weight_hh_l0.copy_(model.lstm_recurrent_weight)
        reference.bias_ih_l0.copy_(model.lstm_bias)
        reference.bias_hh_l0.zero_()

    inputs = torch.randint(vocab_size, (7, 5))
    logits, states = model(inputs, return_states=True)
    embeddings = F.embedding(inputs, model.input_weight.transpose(0, 1))
    reference_states, _ = reference(embeddings)
    reference_logits = F.linear(
        reference_states[:, -1],
        model.input_weight.transpose(0, 1),
        model.output_bias,
    )

    assert torch.allclose(states, reference_states, atol=1e-6, rtol=1e-6)
    assert torch.allclose(logits, reference_logits, atol=1e-6, rtol=1e-6)


def test_product_lstm_factorizes_ids_and_ties_component_readout() -> None:
    torch.manual_seed(23)
    model = ProductTokenLSTM(
        8_192, 16, num_codebooks=4, codebook_size=8,
    )
    inputs = torch.tensor([[1, 4_097], [4_095, 8_191]])
    components = model.token_components(inputs)

    assert components.tolist() == [
        [[1, 0, 0, 0], [1, 0, 0, 0]],
        [[7, 7, 7, 7], [7, 7, 7, 7]],
    ]
    embeddings = model.embed_tokens(inputs)
    expected_role_delta = model.role_weight[1] - model.role_weight[0]
    assert torch.allclose(embeddings[0, 1] - embeddings[0, 0], expected_role_delta)

    readout_mask = torch.ones_like(inputs, dtype=torch.bool)
    logits, states = model(inputs, readout_mask=readout_mask, return_states=True)
    assert logits.shape == (4, 4, 8)
    assert states.shape == (2, 2, 16)
    targets = model.token_components(torch.tensor([4_097, 8_191, 4_097, 8_191]))
    F.cross_entropy(logits.flatten(0, 1), targets.flatten()).backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
