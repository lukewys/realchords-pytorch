"""Linear warmup + cosine decay learning rate scheduler."""

from typing import List

import torch
import math
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer


class LinearWarmupCosineDecay(_LRScheduler):
    """
    Learning rate scheduler that combines a linear warmup + cosine decay.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_iters (int): Number of iterations for the linear warmup phase. Can be 0 for no warmup.
        total_iters (int): Total number of iterations for the entire schedule.
        eta_min (float, optional): Minimum learning rate after cosine decay. Default is 0.
        last_epoch (int, optional): The index of the last iteration/step (note: despite the name, this tracks steps not epochs). Default is -1.

    Methods:
        get_lr() -> List[float]:
            Compute the learning rate for the current iteration.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_iters: int = 0,
        total_iters: int = 0,
        eta_min: float = 0,
        last_epoch: int = -1,
    ):
        if warmup_iters < 0:
            raise ValueError("Warmup iterations must be non-negative.")
        if total_iters <= 0:
            raise ValueError("Total iterations must be positive.")
        if warmup_iters > total_iters:
            raise ValueError("Warmup iterations cannot be greater than total iterations.")
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        self.eta_min = eta_min
        super(LinearWarmupCosineDecay, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_iters:
            # Linear warmup phase
            if self.warmup_iters == 0:
                warmup_factor = 1.0
            else:
                warmup_factor = (self.last_epoch + 1) / self.warmup_iters
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine decay phase
            cosine_iter = self.last_epoch - self.warmup_iters
            curr_progress = cosine_iter / max(
                1, self.total_iters - self.warmup_iters
            )
            cosine_decay_factor = 0.5 * (1 + math.cos(math.pi * curr_progress))
            return [
                self.eta_min + (base_lr - self.eta_min) * cosine_decay_factor
                for base_lr in self.base_lrs
            ]
