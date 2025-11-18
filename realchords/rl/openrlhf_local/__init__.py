"""
This file is part of OpenRLHF v0.6.1.post1, copied locally for stability.
Source: https://github.com/OpenRLHF/OpenRLHF/blob/v0.6.1.post1/openrlhf

Simplified local copy of OpenRLHF v0.6.1.post1 containing only the modules used by RealChords.
This eliminates dependency on an older version of OpenRLHF.

Structure:
- All code is in flat files (no subdirectories)
- Only includes the necessary classes and functions actually used in the codebase

Original imports mapping:
- openrlhf.trainer.ppo_utils.AdaptiveKLController -> openrlhf_local.AdaptiveKLController
- openrlhf.trainer.ppo_utils.FixedKLController -> openrlhf_local.FixedKLController
- openrlhf.models.utils.* -> openrlhf_local.*
- openrlhf.models.ring_attn_utils.* -> openrlhf_local.*
- openrlhf.utils.distributed_sampler.DistributedSampler -> openrlhf_local.DistributedSampler
- openrlhf.utils.deepspeed.deepspeed_utils.* -> openrlhf_local.*
- openrlhf.models.Actor -> openrlhf_local.Actor
- openrlhf.models.(GPTLMLoss|PolicyLoss|ValueLoss) -> openrlhf_local.*
- openrlhf.trainer.PPOTrainer -> openrlhf_local.PPOTrainer

Note: Experience maker and replay buffer classes are now included for reference.
RealChords uses customized versions that inherit from these base classes.
"""

# KL Controllers
from .kl_controller import AdaptiveKLController, FixedKLController

# Experience maker classes
from .experience_maker import Experience, Samples, NaiveExperienceMaker

# Replay buffer classes
from .replay_buffer import BufferItem, NaiveReplayBuffer

# Model utilities
from .utils import (
    compute_approx_kl,
    compute_reward,
    log_probs_from_logits,
    masked_mean,
    masked_normalize,
    reset_position_ids,
    unpacking_samples,
)

# Ring attention utilities
from .ring_attn_utils import (
    get_ring_attn_group,
    set_ring_attn_group,
    pad_sequences,
    unpad_sequences,
)

# Distributed sampler
from .distributed_sampler import DistributedSampler

# DeepSpeed utilities
from .deepspeed_utils import (
    _z3_params_to_fetch,
    get_eval_ds_config,
    get_optimizer_grouped_parameters,
    get_train_ds_config,
)

# Models
from .actor import Actor

# Loss functions
from .loss import GPTLMLoss, PolicyLoss, ValueLoss

# Trainers
from .ppo_trainer import PPOTrainer


__all__ = [
    # KL Controllers
    "AdaptiveKLController",
    "FixedKLController",
    # Experience maker classes
    "Experience",
    "Samples",
    "NaiveExperienceMaker",
    # Replay buffer classes
    "BufferItem",
    "NaiveReplayBuffer",
    # Model utilities
    "compute_approx_kl",
    "compute_reward",
    "log_probs_from_logits",
    "masked_mean",
    "masked_normalize",
    "reset_position_ids",
    "unpacking_samples",
    # Ring attention utilities
    "get_ring_attn_group",
    "set_ring_attn_group",
    "pad_sequences",
    "unpad_sequences",
    # Distributed sampler
    "DistributedSampler",
    # DeepSpeed utilities
    "_z3_params_to_fetch",
    "get_eval_ds_config",
    "get_optimizer_grouped_parameters",
    "get_train_ds_config",
    # Models
    "Actor",
    # Loss functions
    "GPTLMLoss",
    "PolicyLoss",
    "ValueLoss",
    # Trainers
    "PPOTrainer",
]
