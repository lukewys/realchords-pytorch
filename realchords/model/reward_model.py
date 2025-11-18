"""Reward Models for ReaLchords."""

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn import Module

import numpy as np

from x_transformers import TransformerWrapper
from realchords.nn.transformers import Encoder


class ContrastiveReward(Module):
    """Contrastive reward model."""

    def __init__(
        self,
        dim: int = 512,
        depth: int = 6,
        heads: int = 8,
        num_tokens: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super(ContrastiveReward, self).__init__()
        self.chord_encoder = TransformerWrapper(
            attn_layers=Encoder(
                dim=dim,
                depth=depth,
                heads=heads,
                dropout=dropout,
            ),
            num_tokens=num_tokens,
            max_seq_len=max_seq_len,
            return_only_embed=True,
        )
        self.melody_encoder = TransformerWrapper(
            attn_layers=Encoder(
                dim=dim,
                depth=depth,
                heads=heads,
                dropout=dropout,
            ),
            num_tokens=num_tokens,
            max_seq_len=max_seq_len,
            return_only_embed=True,
        )
        self.chord_emb_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.melody_emb_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def get_chord_embed(
        self, chord: Tensor, chord_mask: Tensor = None
    ) -> Tensor:
        """Get chord embeddings."""
        chord_embed = self.chord_encoder(chord, mask=chord_mask)
        chord_embed = self.chord_emb_proj(chord_embed)
        chord_embed = F.normalize(chord_embed, dim=-1)
        # take bos as output
        return chord_embed[:, 0]

    def get_melody_embed(
        self, melody: Tensor, melody_mask: Tensor = None
    ) -> Tensor:
        """Get melody embeddings."""
        melody_embed = self.melody_encoder(melody, mask=melody_mask)
        melody_embed = self.melody_emb_proj(melody_embed)
        melody_embed = F.normalize(melody_embed, dim=-1)
        # take bos as output
        return melody_embed[:, 0]

    def forward(
        self,
        chord: Tensor,
        melody: Tensor,
        chord_mask: Tensor = None,
        melody_mask: Tensor = None,
    ) -> Tensor:
        """Forward pass."""
        chord_embed = self.get_chord_embed(chord, chord_mask)
        melody_embed = self.get_melody_embed(melody, melody_mask)
        return chord_embed, melody_embed, self.logit_scale.exp()


class DiscriminativeReward(Module):
    """Discriminative reward model."""

    def __init__(
        self,
        dim: int = 512,
        depth: int = 6,
        heads: int = 8,
        num_tokens: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super(DiscriminativeReward, self).__init__()
        self.encoder = TransformerWrapper(
            attn_layers=Encoder(
                dim=dim,
                depth=depth,
                heads=heads,
                dropout=dropout,
            ),
            num_tokens=num_tokens,
            max_seq_len=max_seq_len,
            return_only_embed=True,
        )
        self.out_proj = nn.Linear(dim, 2)  # binary classification

    def forward(
        self,
        input: Tensor,
        input_mask: Tensor = None,
    ) -> Tensor:
        """Forward pass."""
        logits = self.out_proj(self.encoder(input, mask=input_mask))
        # use the first token as output
        logits = logits[:, 0]
        return logits
