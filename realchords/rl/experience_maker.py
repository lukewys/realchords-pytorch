"""Experience maker modified from openrlhf."""

from typing import Dict, Any, Union, Tuple, List
import random

import torch
import torch.nn as nn
from copy import deepcopy
from tqdm import tqdm
import numpy as np

from typing import List, Callable, Optional
from dataclasses import dataclass
from realchords.rl.openrlhf_local import (
    Actor,
    compute_approx_kl,
    masked_mean,
    NaiveExperienceMaker,
)
from realchords.rl.utils import compute_full_kl

compute_full_kl = torch.compile(compute_full_kl)


def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    return tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor


def pin_memory(tensor: Union[torch.Tensor, list[torch.Tensor]]):
    if isinstance(tensor, list):
        return [pin_memory(t) for t in tensor]
    return tensor.pin_memory() if isinstance(tensor, torch.Tensor) else tensor


def postprocess_reward(
    r: Union[torch.Tensor, float],
    kl_coef: float,
    kl: Union[torch.Tensor, list[torch.Tensor]],
    action_mask: Optional[torch.Tensor] = None,
    num_actions: Optional[Union[int, list[int]]] = None,
    reward_clip_range: Tuple[float, float] = None,
) -> Union[torch.Tensor, list[torch.Tensor]]:
    """The modified compute_reward function for per-token rewards.
    Add KL to the reward.

    Compared to original one in openrlhf.model.utils, this function
    assumes r to be a tensor of shape (B, S), where B is the batch size
    and S is the sequence length.
    Also, this function requires action_mask to be provided, removing
    the support of packed samples.

    Args:
        r: (B, S) tensor of rewards.
        kl: (B, S) tensor of KL divergences.
        action_mask: (B, S) tensor of action masks.
        num_actions: (B,) tensor of number of actions.
        reward_clip_range: (min, max) tuple of reward clipping range.
    """
    if kl_coef <= 0.0:
        kl_coef = 0.0

    if reward_clip_range:
        r = r.clamp(min=reward_clip_range[0], max=reward_clip_range[1])

    kl_reward = -kl_coef * kl
    reward = r + kl_reward

    return reward


# create a new Experience class that contains targets
@dataclass
class Experience:
    """Experience is a batch of data.
    These data should have the the sequence length and number of actions.
    Left padding for sequences is applied.

    Shapes of each tensor:
    targets: (B, S), the ground truth targets of the sequences from the batch.
    sequences: (B, S)
    action_log_probs: (B, A)
    base_action_log_probs: (B, A)
    values: (B, A)
    returns: (B, A)
    advantages: (B, A)
    attention_mask: (B, S)
    action_mask: (B, A)
    kl: (B, A)

    "A" is the number of actions.
    """

    targets: torch.Tensor
    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    base_action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    info: Optional[dict]
    kl: Optional[torch.Tensor] = None

    @torch.no_grad()
    def to_device(self, device: torch.device):
        self.sequences = to(self.sequences, device)
        self.action_log_probs = to(self.action_log_probs, device)
        self.base_action_log_probs = to(self.base_action_log_probs, device)
        self.returns = to(self.returns, device)
        self.advantages = to(self.advantages, device)
        self.values = to(self.values, device)
        self.attention_mask = to(self.attention_mask, device)
        self.action_mask = to(self.action_mask, device)
        self.kl = to(self.kl, device)
        self.info = {key: to(value, device) for key, value in self.info.items()}
        return self

    def pin_memory(self):
        self.sequences = pin_memory(self.sequences)
        self.action_log_probs = pin_memory(self.action_log_probs)
        self.base_action_log_probs = pin_memory(self.base_action_log_probs)
        self.returns = pin_memory(self.returns)
        self.advantages = pin_memory(self.advantages)
        self.values = pin_memory(self.values)
        self.attention_mask = pin_memory(self.attention_mask)
        self.action_mask = pin_memory(self.action_mask)
        self.kl = pin_memory(self.kl)
        self.info = {key: pin_memory(value) for key, value in self.info.items()}
        return self


# Create a different Samples class that contains batch
@dataclass
class Samples:
    """Samples is a batch of data.
    There can be 2 formats to store the samples, batched or packed.
    The batched format means padding is applied to the sequences, while the packed format
    will concatenate the prompt and response without padding.

    Shapes of each tensor, when 2 shapes are shown, the first one is for batched format
        and the second one is for packed format:
    batch: Dict[str, Any], the batch data as input.
    sequences: (B, S) or (1, total_length), the tokens of both prompt and response.
    attention_mask: (B, S) or (1, total_length), the attention mask for sequences.
    action_mask: (B, A) or None, the action (response) mask to show which part of the
        sequence is the response. When the samples are packed, this is None.
    num_actions: int or (B,), the number of actions (tokens) in the response.
        When the samples are not packed, we will use action_mask, so this is an int to
        show the size of action_mask. Otherwise, this is a tensor to show the number of
        actions for each sample.
    packed_seq_lens: None or (B,), the length of each sample in the packed samples.
    response_length: (B,), the number of tokens in the response.
    total_length: (B,), the total number of tokens in the sequences.
    """

    batch: Dict[str, Any]
    sequences: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    packed_seq_lens: Optional[torch.Tensor]
    response_length: torch.Tensor
    total_length: torch.Tensor


class ExperienceMaker(NaiveExperienceMaker):
    """
    Experience maker modified from openrlhf.

    The format of the reward_configs is as follows:
    reward_configs = [
        {
            "reward_fn": reward_fn,
            "weight": weight,
            "name": name,
        },
        ...
    ]

    It removes the following features:
    1. Support of remote_rm_url and the original reward_fn
        which is used in remote reward.
    2. Remove micro_rollout_batch_size vs. rollout_batch_size.
        Now they are the same, and we only do 1 rollout step.

    In addition, it supports the following features:
    - Return and log metrics from the reward function.
    - In generate_samples function, allow generate from a batch of items
        rather than a list of prompts. Now actor takes in a batch.
    - Compute full KL rather than approximate KL.
    - Support ensemble of reward functions and reward models.
    - Support configurable combination of the rewards from each sources.
        (now only sum reward with coefficients).
    - Support VRAM swapping for the model-based rewards.
    - Support token-wise rewards.
    """

    def __init__(
        self,
        actor: Actor,
        critic: nn.Module,
        reward_configs: List[Dict[str, Any]],
        initial_model: Actor,
        tokenizer,
        kl_controller,
        strategy=None,
        reward_vram_swap: bool = False,
        logits_vram_swap: bool = False,
        **kwargs,
    ):

        self.actor = actor
        self.critic = critic
        self.initial_model = initial_model
        self.tokenizer = tokenizer
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.perf_stats = None
        self.advantage_estimator = strategy.args.advantage_estimator
        # In here, we assume the reward model is already moved to the
        #   dedicated device for inference when initializing.
        self.reward_vram_swap = reward_vram_swap
        self.logits_vram_swap = logits_vram_swap
        self.init_rewards(reward_configs)

    def init_rewards(self, reward_configs: List[Dict[str, Any]]):
        """
        Initialize the reward functions.
        """
        reward_fns = [k["reward_fn"] for k in reward_configs]
        reward_weights = [k.get("weight", None) for k in reward_configs]
        reward_names = [k.get("name", None) for k in reward_configs]
        reward_clip_range = [k.get("clip_range", None) for k in reward_configs]
        for i, reward_fn in enumerate(reward_fns):
            if not isinstance(reward_fn, Callable):
                raise ValueError(
                    f"reward_fn at index {i} is not a callable function."
                )
            if reward_names[i] is None:
                print(
                    f"Warning: reward function {reward_fn.__name__} has no name. "
                    "Using the function name as the reward name."
                )
                reward_names[i] = reward_fn.__name__
            if reward_weights[i] is None:
                print(
                    f"Warning: reward function {reward_names[i]} has no weight. "
                    "Setting the weight to 1.0."
                )
                reward_weights[i] = 1.0
            if reward_clip_range[i] == (None, None) or reward_clip_range[i] == [
                None,
                None,
            ]:
                # To compatible with a tuple of None, None or list of None, None
                #   or None
                reward_clip_range[i] = None

        self.reward_fns = reward_fns
        self.reward_weights = reward_weights
        self.reward_names = reward_names
        self.reward_clip_range = reward_clip_range
        self.reward_device_dict = {}
        for reward_fn, name in zip(self.reward_fns, self.reward_names):
            if isinstance(reward_fn, nn.Module):
                reward_fn.eval()
                if self.reward_vram_swap:
                    self.reward_device_dict[name] = next(
                        reward_fn.parameters()
                    ).device
                    reward_fn.cpu()

    @torch.no_grad()
    def compute_rewards(
        self,
        samples: Samples,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute rewards from the reward functions.
        """
        rewards = []
        reward_metrics_all = {}
        for i, reward_fn in enumerate(self.reward_fns):
            name = self.reward_names[i]

            if self.reward_vram_swap and isinstance(reward_fn, nn.Module):
                reward_fn.to(self.reward_device_dict[name])
            with torch.no_grad():
                reward_metrics = reward_fn(samples)
            if self.reward_vram_swap and isinstance(reward_fn, nn.Module):
                reward_fn.cpu()

            r = reward_metrics.pop("reward")

            # Clip the reward, and sum across time to get scalar reward for logging.
            reward_clip_range = self.reward_clip_range[i]
            if reward_clip_range is not None:
                # Log the raw reward
                reward_metrics[f"reward_{name}_raw"] = r.sum(dim=-1)
                r = r.clamp(
                    min=reward_clip_range[0],
                    max=reward_clip_range[1],
                )
            # Log the clipped reward
            reward_metrics[f"reward_{name}"] = r.sum(dim=-1)

            rewards.append(r)
            reward_metrics_all.update(reward_metrics)

        # combine rewards
        rewards = torch.stack(rewards, dim=0)  # [num_rewards, B, S]
        weights = (
            torch.tensor(self.reward_weights, device=rewards.device)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )  # [num_rewards, 1, 1]

        rewards = torch.sum(rewards * weights, dim=0)  # [B, S]
        return rewards, reward_metrics_all

    @torch.no_grad()
    def generate_samples(
        self, batch: Dict[str, Any], is_eval: bool = False, **generate_kwargs
    ) -> List[Samples]:
        """
        Generate samples and return in batches.

        We removed the micro rollout in here.
        Now we only do 1 rollout step.
        """
        assert not getattr(self, "packing_samples", False)
        self.actor.eval()
        samples_list = []
        sequences, attention_mask, action_mask = self.actor.generate(
            batch, enable_filter_fn=is_eval, **generate_kwargs
        )
        samples = Samples(
            batch=batch,
            sequences=sequences,
            attention_mask=attention_mask,
            action_mask=action_mask,
            num_actions=action_mask.size(1),
            packed_seq_lens=None,
            response_length=action_mask.float().sum(dim=-1),
            total_length=attention_mask.float().sum(dim=-1),
        )
        samples_list.append(samples)
        return samples_list

    @torch.no_grad()
    def get_advantages_and_returns_interleave(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Modified version for interleaved realchords trajectory.

        In realchords, the trajectory is [bos, chord1, melody1, chord2, melody2, ...],
        and the action mask is [0, 1, 0, 1, 0, ...].

        We formulate the state in our case as st = melody<t, chord<t.
        So, the value for the first action chord1,
        which should be V(s1), should be V(bos),
        and the value for the second action chord2,
        which should be V(s2), should be V(bos, chord1, melody1).

        This function calculates the advantage following the above definition.

        Assumes that valid (action) timesteps occur at even indices (i.e.,
        action_mask[:, 0] == 1, action_mask[:, 1] == 0, action_mask[:, 2] == 1, etc.),
        though each batch sample may have a different number of valid tokens due to padding.

        The advantage/return calculations are performed on the compressed valid tokens,
        then the resulting advantages and returns are scattered back into tensors of shape
        (batch_size, response_size) so that valid tokens are at the corresponding even indices,
        and the other positions remain 0.
        """
        # First, mask out invalid positions.
        values = values * action_mask
        rewards = rewards * action_mask

        # Compress the sequence by selecting only even-indexed timesteps.
        # (This assumes that valid tokens are exactly those positions.)
        valid_values = values[:, ::2]
        valid_rewards = rewards[:, ::2]
        # 1 for valid tokens, 0 for padded positions.
        valid_mask = action_mask[:, ::2]

        batch_size, valid_length = valid_values.shape

        # Compute advantages on the compressed (valid) sequence.
        advantages_valid = torch.zeros_like(valid_values)
        lastgaelam = torch.zeros(
            batch_size, device=values.device, dtype=values.dtype
        )

        # Backward recurrence over valid timesteps
        for t in reversed(range(valid_length)):
            # Get the next valid value if available; else use 0.
            next_value = (
                valid_values[:, t + 1]
                if t < valid_length - 1
                else torch.zeros_like(lastgaelam)
            )
            # Compute the temporal difference error.
            delta = (
                valid_rewards[:, t] + gamma * next_value - valid_values[:, t]
            )
            lastgaelam = delta + gamma * lambd * lastgaelam
            # If this timestep is actually padded (mask==0), force advantage to 0.
            lastgaelam = lastgaelam * valid_mask[:, t]
            advantages_valid[:, t] = lastgaelam

        # Compute returns on valid timesteps.
        returns_valid = advantages_valid + valid_values

        # Create full-sized tensors for advantages and returns (same shape as original inputs).
        response_length = values.size(1)
        advantages = torch.zeros_like(values)
        returns = torch.zeros_like(values)

        # Scatter the computed valid advantages/returns back into the even indices.
        advantages[:, ::2] = advantages_valid
        returns[:, ::2] = returns_valid

        return advantages.detach(), returns

    @torch.no_grad()
    def make_experience_list(
        self, batch: Dict[str, Any], is_eval: bool = False, **generate_kwargs
    ) -> List[Experience]:
        """
        Make a list of experience with the rollout_batch_size (only 1 rollout step).

        This method will first calculate the response sequences and rewards for the given prompts.
        Then, if we need certain processing for the rewards or do certain filtering, we can process the rollout as a whole.
        After that, we will calculate the advantages and returns for each experience.
        """
        args = self.strategy.args
        experiences = []
        for samples in tqdm(
            self.generate_samples(batch, is_eval=is_eval, **generate_kwargs),
            desc="make_experience",
            disable=not self.strategy.is_rank_0(),
        ):
            experiences.append(self.make_experience(samples))

        # We will not use process_experiences in here
        # (we are not using rloo or grpo)
        experiences, _ = self.process_experiences(experiences)

        # calculate return and advantages
        for experience in experiences:
            num_actions = experience.info["num_actions"]
            reward = postprocess_reward(
                experience.info["reward"],
                self.kl_ctl.value,
                experience.kl,
                action_mask=experience.action_mask,
                num_actions=num_actions,
                reward_clip_range=args.reward_clip_range,
            )

            # Sum across time to get scalar reward for logging.
            # here experience.info["reward"] is a tensor of shape [B, S]
            # and it comes from the reward functions
            experience.info["reward"] = experience.info["reward"].sum(dim=-1)

            if self.advantage_estimator == "gae":
                experience.advantages, experience.returns = (
                    self.get_advantages_and_returns(
                        experience.values,
                        reward,
                        experience.action_mask,
                        generate_kwargs["gamma"],
                        generate_kwargs["lambd"],
                    )
                )
            elif self.advantage_estimator == "reinforce":
                experience.returns = self.get_cumulative_returns(
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                )
                experience.advantages = deepcopy(experience.returns)
            elif self.advantage_estimator == "gae_interleave":
                experience.advantages, experience.returns = (
                    self.get_advantages_and_returns_interleave(
                        experience.values,
                        reward,
                        experience.action_mask,
                        generate_kwargs["gamma"],
                        generate_kwargs["lambd"],
                    )
                )
            else:
                raise Exception(
                    f"Unkown advantage_estimator {self.advantage_estimator}"
                )

            # calculate the return info.
            if not getattr(self, "packing_samples", False):
                return_sums = reward.sum(dim=-1)
            else:
                return_sums = torch.tensor(
                    [each_reward.sum() for each_reward in reward],
                    device=torch.cuda.current_device(),
                )
            experience.info["return"] = return_sums
            # remove unnecessary info
            experience.kl = None
            del experience.info["num_actions"]
        return experiences

    def postprocess_info(
        self, info: Dict[str, Any], samples: Samples
    ) -> Dict[str, Any]:
        """
        Postprocess the info dictionary.
        """
        return info

    @torch.no_grad()
    def make_experience(self, samples: Samples) -> Experience:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        self.actor.eval()
        self.initial_model.eval()
        if self.critic is not None:
            self.critic.eval()

        # extract values from samples
        sequences = samples.sequences
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions
        batch = samples.batch

        if self.logits_vram_swap:
            device = sequences.device

        # log probs
        action_log_probs, logits = self.actor(
            sequences, num_actions, attention_mask, return_output=True
        )

        if self.logits_vram_swap:
            action_log_probs = action_log_probs.cpu()
            logits = logits.cpu()
            torch.cuda.empty_cache()

        # init log probs
        base_action_log_probs, base_logits = self.initial_model(
            sequences, num_actions, attention_mask, return_output=True
        )

        if self.logits_vram_swap:
            base_action_log_probs = base_action_log_probs.cpu()
            base_logits = base_logits.cpu()
            torch.cuda.empty_cache()

        # values
        if self.critic is not None:
            value = self.critic(sequences, num_actions, attention_mask)
        else:
            value = None

        # rewards
        r, reward_metrics = self.compute_rewards(samples)

        if self.logits_vram_swap:
            # transfer back to device
            action_log_probs = to(action_log_probs, device)
            logits = to(logits, device)
            base_action_log_probs = to(base_action_log_probs, device)
            base_logits = to(base_logits, device)

        if self.strategy.args.use_full_kl or self.strategy.args.use_reverse_kl:
            if (
                self.strategy.args.use_full_kl
                and self.strategy.args.use_reverse_kl
            ):
                raise ValueError(
                    "Cannot use both full and reverse KL divergence at the same time."
                )
            kl = compute_full_kl(
                logits,
                base_logits,
                action_mask=action_mask,
                use_reverse_kl=self.strategy.args.use_reverse_kl,
            )
        else:
            kl = compute_approx_kl(
                action_log_probs,
                base_action_log_probs,
                action_mask=action_mask,
                use_kl_estimator_k3=self.strategy.args.kl_estimator,
            )

        info = {
            "kl": masked_mean(kl, action_mask, dim=-1),
            "reward": r,
            "response_length": samples.response_length,
            "total_length": samples.total_length,
            "num_actions": num_actions,
            **reward_metrics,
        }

        info = self.postprocess_info(info, samples)

        # reset model state
        self.actor.train()
        if self.critic is not None:
            self.critic.train()

        # HACK: this is a hack to make sure targets has same samples as sequences.
        # However, the samples after this removal won't always align.
        targets = batch["targets"]
        batch_size = sequences.size(0)
        targets = targets[:batch_size]
        return Experience(
            targets,
            sequences,
            action_log_probs,
            base_action_log_probs,
            value,
            None,  # returns, will be calculated later
            None,  # advantages, will be calculated later
            attention_mask,
            action_mask,
            info,
            kl,
        )