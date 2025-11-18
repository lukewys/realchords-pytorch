"""Base class for reward functions.

API for reward functions:
Input:
- samples: Samples object containing the following fields:
    - batch: dict[str, torch.Tensor]
    - sequences: torch.Tensor
    - attention_mask: torch.Tensor
    - action_mask: torch.Tensor
    - num_actions: Union[int, torch.Tensor]
    - packed_seq_lens: Optional[torch.Tensor]
    - response_length: torch.Tensor
    - total_length: torch.Tensor
    - **kwargs: additional arguments from the subclass

Output: dict[str, torch.Tensor], with the following keys:
- reward: torch.Tensor, shape: [B, S] the reward for each token
- key_metrics: value (shape: [B]), key_metrics for logging for each batch
key_metrics can be any named metrics.
"""

from typing import Dict

import torch
import torch.nn as nn

from realchords.rl.experience_maker import Samples


class BaseRewardFn:
    """Base class for CPU-based reward functions."""

    def __init__(self):
        pass

    def forward(
        self,
        samples: Samples,
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def __call__(
        self,
        samples: Samples,
    ) -> Dict[str, torch.Tensor]:
        return self.forward(samples)


class BaseRewardModel(nn.Module):
    """Base class for GPU-based reward functions that require neural network components."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        samples: Samples,
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def __call__(
        self,
        samples: Samples,
    ) -> Dict[str, torch.Tensor]:
        return self.forward(samples)
