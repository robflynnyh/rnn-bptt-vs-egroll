"""A deliberately plain tanh RNN used by both training methods."""

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
    return_states: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Run the benchmark RNN from an explicit parameter mapping."""

    if inputs.ndim != 3:
        raise ValueError("inputs must have shape [batch, time, input_size]")
    hidden_size = parameters["recurrent_weight"].shape[0]
    hidden = inputs.new_zeros(inputs.shape[0], hidden_size)
    states = []
    for step in inputs.unbind(dim=1):
        hidden = torch.tanh(
            F.linear(step, parameters["input_weight"])
            + F.linear(
                hidden,
                parameters["recurrent_weight"],
                parameters["hidden_bias"],
            )
        )
        if return_states:
            states.append(hidden)
    logits = F.linear(
        hidden,
        parameters["output_weight"],
        parameters["output_bias"],
    )
    if not return_states:
        return logits
    return logits, torch.stack(states, dim=1)


class VanillaRNN(nn.Module):
    """Single-layer Elman RNN with tanh dynamics and a final-step readout."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        *,
        recurrent_radius: float = 0.9,
    ) -> None:
        super().__init__()
        if min(input_size, hidden_size, output_size) < 1:
            raise ValueError("all model dimensions must be positive")
        if recurrent_radius <= 0:
            raise ValueError("recurrent_radius must be positive")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.recurrent_radius = recurrent_radius
        self.input_weight = nn.Parameter(torch.empty(hidden_size, input_size))
        self.recurrent_weight = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.hidden_bias = nn.Parameter(torch.zeros(hidden_size))
        self.output_weight = nn.Parameter(torch.empty(output_size, hidden_size))
        self.output_bias = nn.Parameter(torch.zeros(output_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_weight)
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
        return_states: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        return functional_rnn_forward(
            inputs,
            dict(self.named_parameters()),
            return_states=return_states,
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
