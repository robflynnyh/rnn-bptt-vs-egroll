"""A deliberately plain token-level tanh RNN used by both methods."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def functional_rnn_forward(
    inputs: Tensor,
    parameters: Mapping[str, Tensor],
    *,
    readout_mask: Tensor | None = None,
    return_states: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Run the benchmark RNN from an explicit parameter mapping.

    Without a mask, the function retains the final-state readout used by small
    parity tests. MQAR training supplies a mask and receives logits only for
    supervised query positions, avoiding a dense ``[batch, time, vocab]``
    allocation.
    """

    if inputs.ndim != 2 or inputs.dtype != torch.long:
        raise ValueError("inputs must be integer token IDs with shape [batch, time]")
    if readout_mask is not None and readout_mask.shape != inputs.shape:
        raise ValueError("readout_mask must have the same shape as inputs")
    hidden_size = parameters["recurrent_weight"].shape[0]
    hidden = parameters["input_weight"].new_zeros(inputs.shape[0], hidden_size)
    states = []
    for token_ids in inputs.unbind(dim=1):
        input_term = F.embedding(token_ids, parameters["input_weight"].transpose(0, 1))
        hidden = torch.tanh(
            input_term
            + F.linear(
                hidden, parameters["recurrent_weight"], parameters["hidden_bias"],
            )
        )
        if return_states or readout_mask is not None:
            states.append(hidden)

    stacked_states = torch.stack(states, dim=1) if states else None
    readout_states = hidden if readout_mask is None else stacked_states[readout_mask]
    output_weight = (
        parameters["input_weight"].transpose(0, 1)
        if "output_weight" not in parameters
        else parameters["output_weight"]
    )
    logits = F.linear(readout_states, output_weight, parameters["output_bias"])
    if not return_states:
        return logits
    assert stacked_states is not None
    return logits, stacked_states


def _signed_pair(values: Tensor) -> Tensor:
    return torch.cat((values, -values), dim=0)


def _repeated_pair(values: Tensor) -> Tensor:
    return torch.cat((values, values), dim=0)


def _corrected_affine(
    inputs: Tensor,
    weight: Tensor,
    weight_name: str,
    *,
    corrections: Any | None,
    sigma: float,
    bias: Tensor | None = None,
    bias_name: str | None = None,
    transpose_correction: bool = False,
) -> Tensor:
    if corrections is None:
        return F.linear(inputs, weight, bias)
    factors = corrections.matrices[weight_name]
    if transpose_correction:
        left = _signed_pair(factors.right)
        right = _repeated_pair(factors.left)
    else:
        left = _signed_pair(factors.left)
        right = _repeated_pair(factors.right)
    base = F.linear(inputs, weight, bias)
    if inputs.ndim == 2:
        outputs = base.unsqueeze(0).expand(
            corrections.population_size, -1, -1,
        ).clone()
        projected = torch.einsum("bi,pir->pbr", inputs, right)
    elif inputs.ndim == 3:
        outputs = base
        projected = torch.einsum("pbi,pir->pbr", inputs, right)
    else:
        raise ValueError("corrected affine input must be [B, D] or [P, B, D]")
    scale = sigma * corrections.scale(weight_name) / math.sqrt(corrections.rank)
    for rank_index in range(corrections.rank):
        outputs.addcmul_(
            projected[:, :, rank_index, None],
            left[:, None, :, rank_index],
            value=scale,
        )
    if bias is not None:
        if bias_name is None:
            raise ValueError("bias_name is required when a bias is provided")
        outputs.add_(
            _signed_pair(corrections.vectors[bias_name])[:, None],
            alpha=sigma * corrections.scale(bias_name),
        )
    return outputs


def _corrected_embedding(
    token_ids: Tensor,
    weight: Tensor,
    weight_name: str,
    *,
    corrections: Any | None,
    sigma: float,
    candidate_inputs: bool,
) -> Tensor:
    if corrections is None:
        if candidate_inputs:
            raise ValueError("candidate_inputs requires low-rank corrections")
        return F.embedding(token_ids, weight.transpose(0, 1))
    factors = corrections.matrices[weight_name]
    left = _signed_pair(factors.left)
    right = _repeated_pair(factors.right)
    base = F.embedding(token_ids, weight.transpose(0, 1))
    if candidate_inputs:
        if token_ids.shape[0] != corrections.population_size:
            raise ValueError("candidate token inputs must match the population")
        rows = torch.arange(corrections.population_size, device=token_ids.device)
        selected_right = right[rows, token_ids]
        outputs = base[:, None].clone()
    else:
        selected_right = right[:, token_ids]
        outputs = base.unsqueeze(0).expand(
            corrections.population_size, -1, -1,
        ).clone()
    scale = sigma * corrections.scale(weight_name) / math.sqrt(corrections.rank)
    for rank_index in range(corrections.rank):
        if candidate_inputs:
            outputs.addcmul_(
                selected_right[:, rank_index, None, None],
                left[:, None, :, rank_index],
                value=scale,
            )
        else:
            outputs.addcmul_(
                selected_right[:, :, rank_index, None],
                left[:, None, :, rank_index],
                value=scale,
            )
    return outputs


def functional_lstm_forward(
    inputs: Tensor,
    parameters: Mapping[str, Tensor],
    *,
    readout_mask: Tensor | None = None,
    return_states: bool = False,
    corrections: Any | None = None,
    sigma: float = 0.0,
    candidate_inputs: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Run one LSTM recurrence, optionally for low-rank-corrected candidates."""

    if inputs.ndim != 2 or inputs.dtype != torch.long:
        raise ValueError("inputs must be integer token IDs with shape [batch, time]")
    if readout_mask is not None and readout_mask.shape != inputs.shape:
        raise ValueError("readout_mask must have the same shape as inputs")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    hidden_size = parameters["lstm_recurrent_weight"].shape[1]
    if corrections is None:
        hidden = parameters["input_weight"].new_zeros(inputs.shape[0], hidden_size)
    else:
        candidate_batch = 1 if candidate_inputs else inputs.shape[0]
        hidden = parameters["input_weight"].new_zeros(
            corrections.population_size, candidate_batch, hidden_size,
        )
    cell = torch.zeros_like(hidden)
    states = []
    selected_states = []
    for time, token_ids in enumerate(inputs.unbind(dim=1)):
        embedded = _corrected_embedding(
            token_ids,
            parameters["input_weight"],
            "input_weight",
            corrections=corrections,
            sigma=sigma,
            candidate_inputs=candidate_inputs,
        )
        gates = _corrected_affine(
            embedded,
            parameters["lstm_input_weight"],
            "lstm_input_weight",
            corrections=corrections,
            sigma=sigma,
        )
        gates = gates + _corrected_affine(
            hidden,
            parameters["lstm_recurrent_weight"],
            "lstm_recurrent_weight",
            corrections=corrections,
            sigma=sigma,
            bias=parameters["lstm_bias"],
            bias_name="lstm_bias",
        )
        input_gate, forget_gate, candidate, output_gate = gates.chunk(4, dim=-1)
        cell = torch.sigmoid(forget_gate) * cell + (
            torch.sigmoid(input_gate) * torch.tanh(candidate)
        )
        hidden = torch.sigmoid(output_gate) * torch.tanh(cell)
        states.append(hidden)
        if readout_mask is not None and not candidate_inputs:
            selected = readout_mask[:, time]
            if selected.any():
                selected_states.append(
                    hidden[selected] if corrections is None else hidden[:, selected]
                )

    if corrections is None:
        stacked_states = torch.stack(states, dim=1)
        readout_states = hidden if readout_mask is None else torch.cat(selected_states)
    elif candidate_inputs:
        stacked_states = torch.stack([state[:, 0] for state in states], dim=1)
        if readout_mask is None:
            readout_states = hidden
        else:
            counts = readout_mask.sum(dim=1)
            if int(counts.min()) < 1 or not counts.eq(counts[0]).all():
                raise ValueError("candidate readout counts must be equal and nonzero")
            readout_states = stacked_states[readout_mask].reshape(
                corrections.population_size, int(counts[0]), hidden_size,
            )
    else:
        stacked_states = torch.stack(states, dim=2)
        readout_states = (
            hidden if readout_mask is None else torch.cat(selected_states, dim=1)
        )

    tied_output = "output_weight" not in parameters
    output_weight = (
        parameters["input_weight"].transpose(0, 1)
        if tied_output
        else parameters["output_weight"]
    )
    logits = _corrected_affine(
        readout_states,
        output_weight,
        "input_weight" if tied_output else "output_weight",
        corrections=corrections,
        sigma=sigma,
        bias=parameters["output_bias"],
        bias_name="output_bias",
        transpose_correction=tied_output,
    )
    if not return_states:
        return logits
    return logits, stacked_states


class VanillaRNN(nn.Module):
    """Single-layer Elman RNN with one-hot-equivalent token inputs."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        *,
        recurrent_radius: float = 0.9,
        tie_input_output: bool = False,
    ) -> None:
        super().__init__()
        if min(vocab_size, hidden_size) < 1:
            raise ValueError("all model dimensions must be positive")
        if recurrent_radius <= 0:
            raise ValueError("recurrent_radius must be positive")
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.recurrent_radius = recurrent_radius
        self.tie_input_output = tie_input_output
        self.input_weight = nn.Parameter(torch.empty(hidden_size, vocab_size))
        self.recurrent_weight = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.hidden_bias = nn.Parameter(torch.zeros(hidden_size))
        if not tie_input_output:
            self.output_weight = nn.Parameter(torch.empty(vocab_size, hidden_size))
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.input_weight, std=1 / math.sqrt(self.hidden_size))
        nn.init.orthogonal_(self.recurrent_weight)
        with torch.no_grad():
            self.recurrent_weight.mul_(self.recurrent_radius)
        if not self.tie_input_output:
            nn.init.uniform_(
                self.output_weight,
                -1 / math.sqrt(self.hidden_size),
                1 / math.sqrt(self.hidden_size),
            )
        nn.init.zeros_(self.hidden_bias)
        nn.init.zeros_(self.output_bias)

    def forward(
        self,
        inputs: Tensor,
        *,
        readout_mask: Tensor | None = None,
        return_states: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        return functional_rnn_forward(
            inputs,
            dict(self.named_parameters()),
            readout_mask=readout_mask,
            return_states=return_states,
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class TokenLSTM(nn.Module):
    """Single-layer LSTM with the same token embedding and readout interface."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        *,
        tie_input_output: bool = False,
    ) -> None:
        super().__init__()
        if min(vocab_size, hidden_size) < 1:
            raise ValueError("all model dimensions must be positive")
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.tie_input_output = tie_input_output
        self.input_weight = nn.Parameter(torch.empty(hidden_size, vocab_size))
        self.lstm_input_weight = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.lstm_recurrent_weight = nn.Parameter(
            torch.empty(4 * hidden_size, hidden_size)
        )
        self.lstm_bias = nn.Parameter(torch.empty(4 * hidden_size))
        if not tie_input_output:
            self.output_weight = nn.Parameter(torch.empty(vocab_size, hidden_size))
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.input_weight, std=1 / math.sqrt(self.hidden_size))
        bound = 1 / math.sqrt(self.hidden_size)
        nn.init.uniform_(self.lstm_input_weight, -bound, bound)
        nn.init.uniform_(self.lstm_recurrent_weight, -bound, bound)
        nn.init.uniform_(self.lstm_bias, -bound, bound)
        if not self.tie_input_output:
            nn.init.uniform_(
                self.output_weight,
                -1 / math.sqrt(self.hidden_size),
                1 / math.sqrt(self.hidden_size),
            )
        nn.init.zeros_(self.output_bias)

    def forward(
        self,
        inputs: Tensor,
        *,
        readout_mask: Tensor | None = None,
        return_states: bool = False,
        corrections: Any | None = None,
        sigma: float = 0.0,
        candidate_inputs: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        return functional_lstm_forward(
            inputs,
            dict(self.named_parameters()),
            readout_mask=readout_mask,
            return_states=return_states,
            corrections=corrections,
            sigma=sigma,
            candidate_inputs=candidate_inputs,
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
