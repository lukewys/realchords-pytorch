"""Critic or value model for RL training."""

from typing import Optional, Union

import torch
import torch.nn as nn


class Critic(nn.Module):
    """Critic model base class.

    Internally we call x_transformers API for model generation,
    and externally we follow the OpenRLHF API.

    We removed the following features:
    1. packing samples
    """

    def __init__(self, model: nn.Module):
        """The model here should be a loaded pretrain model."""
        super().__init__()
        self.model = model

        # Change value head
        self.model.decoder.to_logits = nn.Linear(
            self.model.decoder.emb_dim, 1, bias=False
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        num_actions: Optional[Union[int, list[int]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Returns action values.

        Compared to the original forward, remove the creation of position_ids.
        """

        values = self.model(input_ids, mask=attention_mask).squeeze(-1)[:, :-1]
        outputs = None

        if num_actions is None:
            assert return_output
            return outputs

        action_values = values[:, -num_actions:]

        if return_output:
            return (action_values, outputs)
        else:
            return action_values
