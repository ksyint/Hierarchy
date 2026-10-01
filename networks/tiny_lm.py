from types import SimpleNamespace

import torch
from torch import nn


class TinyLanguageModel(nn.Module):
    """A small autoregressive fixture, not a substitute for the paper's SLMs."""
    def __init__(self, dim=48):
        super().__init__()
        self.embed = nn.Embedding(259, dim)
        self.rnn = nn.GRU(dim, dim, batch_first=True)
        self.lm_head = nn.Linear(dim, 259)

    def forward(self, input_ids, attention_mask=None):
        # A copied frozen reference needs its CUDA recurrent weights repacked.
        self.rnn.flatten_parameters()
        hidden, _ = self.rnn(self.embed(input_ids))
        return SimpleNamespace(logits=self.lm_head(hidden))
