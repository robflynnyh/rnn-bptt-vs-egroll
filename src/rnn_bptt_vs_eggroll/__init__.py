"""BPTT versus EGGROLL on associative recall."""

from .model import VanillaRNN
from .task import AssociativeRecallBatch, AssociativeRecallConfig, sample_batch

__all__ = [
    "AssociativeRecallBatch",
    "AssociativeRecallConfig",
    "VanillaRNN",
    "sample_batch",
]
