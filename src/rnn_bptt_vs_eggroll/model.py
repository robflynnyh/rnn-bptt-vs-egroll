"""A deliberately plain token-level tanh RNN used by both methods."""

from __future__ import annotations

import math
from collections.abc import Mapping

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
    logits = F.linear(
        readout_states, parameters["output_weight"], parameters["output_bias"],
    )
    if not return_states:
        return logits
    assert stacked_states is not None
    return logits, stacked_states


class VanillaRNN(nn.Module):
    """Single-layer Elman RNN with one-hot-equivalent token inputs."""

    def __init__(
        self, vocab_size: int, hidden_size: int, *, recurrent_radius: float = 0.9,
    ) -> None:
        super().__init__()
        if min(vocab_size, hidden_size) < 1:
            raise ValueError("all model dimensions must be positive")
        if recurrent_radius <= 0:
            raise ValueError("recurrent_radius must be positive")
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.recurrent_radius = recurrent_radius
        self.input_weight = nn.Parameter(torch.empty(hidden_size, vocab_size))
        self.recurrent_weight = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.hidden_bias = nn.Parameter(torch.zeros(hidden_size))
        self.output_weight = nn.Parameter(torch.empty(vocab_size, hidden_size))
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.input_weight, std=1 / math.sqrt(self.hidden_size))
        nn.init.orthogonal_(self.recurrent_weight)
        with torch.no_grad():
            self.recurrent_weight.mul_(self.recurrent_radius)
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
