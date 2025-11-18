"""Model-based reward functions/wrappers for ReaLchords.

API for reward functions:
Input:
- batch: dict[str, torch.Tensor]
- sequence: torch.Tensor
- output_mask: torch.Tensor
- action_mask: torch.Tensor

Output: dict[str, torch.Tensor], with the following keys:
- reward: torch.Tensor, shape: [B, S] the reward for each token
- key_metrics: value (shape: [B]), key_metrics for logging for each batch
key_metrics can be any named metrics.
"""

from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from realchords.rl.openrlhf_local import log_probs_from_logits, masked_mean

from realchords.rl.reward.base import BaseRewardModel
from realchords.model.reward_model import (
    ContrastiveReward,
    DiscriminativeReward,
)
from realchords.rl.experience_maker import Samples
from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer


from realchords.utils.sequence_utils import (
    add_bos_to_sequence,
    add_eos_to_sequence,
    sequences_order_to_counterpart,
    log_probs_from_online_model,
    get_seperated_parts_from_sequence,
)
from realchords.utils.sequence_utils import (
    create_table_from_mapping,
    remap_from_table,
)
from realchords.rl.utils import assign_reward_to_last_token, compute_full_kl


class ContrastiveRewardFn(BaseRewardModel):
    def __init__(
        self,
        model: ContrastiveReward,
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
        model_part: str,
    ):
        super().__init__()
        self.model = model
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.model_part = model_part
        if model_part not in ["chord", "melody"]:
            raise ValueError(
                f"model_part must be either 'chord' or 'melody', got {model_part}"
            )

    def get_inputs_from_sequence(
        self, sequence: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # remove bos
        sequence = sequence[:, 1:].clone()
        model_tokens = sequence[:, ::2]
        context_tokens = sequence[:, 1::2]

        # add eos to the first padding token
        model_tokens = add_eos_to_sequence(
            model_tokens, self.pad_token_id, self.eos_token_id
        )
        context_tokens = add_eos_to_sequence(
            context_tokens, self.pad_token_id, self.eos_token_id
        )

        # add bos to the beginning of the sequence
        model_tokens = add_bos_to_sequence(model_tokens, self.bos_token_id)
        context_tokens = add_bos_to_sequence(context_tokens, self.bos_token_id)
        model_mask = model_tokens != self.pad_token_id
        context_mask = context_tokens != self.pad_token_id
        return model_tokens, context_tokens, model_mask, context_mask

    @torch.no_grad()
    def forward(
        self,
        samples: Samples,
    ) -> Dict[str, torch.Tensor]:
        # Extract variables from samples
        sequence = samples.sequences
        action_mask = samples.action_mask

        # sequence: [B, T]
        # output_mask: [B, T]

        # Now we get device from model. The model is initialized with DeepSpeed strategy.
        device = self.model.device

        model_tokens, context_tokens, model_mask, context_mask = (
            self.get_inputs_from_sequence(sequence)
        )
        if self.model_part == "chord":
            chord_tokens = model_tokens
            chord_mask = model_mask
            melody_tokens = context_tokens
            melody_mask = context_mask
        else:
            melody_tokens = model_tokens
            melody_mask = model_mask
            chord_tokens = context_tokens
            chord_mask = context_mask

        chord_embed, melody_embed, _ = self.model(
            chord=chord_tokens.to(device),
            melody=melody_tokens.to(device),
            chord_mask=chord_mask.to(device),
            melody_mask=melody_mask.to(device),
        )

        reward = (chord_embed * melody_embed).sum(-1).to(sequence.device)

        return {"reward": assign_reward_to_last_token(reward, action_mask)}


class DiscriminativeRewardFn(BaseRewardModel):
    def __init__(
        self,
        model: DiscriminativeReward,
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
        model_part: str,
    ):
        super().__init__()
        self.model = model
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.model_part = model_part
        if model_part not in ["chord", "melody"]:
            raise ValueError(
                f"model_part must be either 'chord' or 'melody', got {model_part}"
            )

    def get_inputs_from_sequence(
        self,
        sequence: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # remove bos
        sequence = sequence[:, 1:].clone()
        model_tokens = sequence[:, ::2]
        context_tokens = sequence[:, 1::2]

        # add eos to the first padding token
        model_tokens = add_eos_to_sequence(
            model_tokens, self.pad_token_id, self.eos_token_id
        )
        context_tokens = add_eos_to_sequence(
            context_tokens, self.pad_token_id, self.eos_token_id
        )

        # add bos to the beginning of the sequence
        model_tokens = add_bos_to_sequence(model_tokens, self.bos_token_id)
        context_tokens = add_bos_to_sequence(context_tokens, self.bos_token_id)

        if self.model_part == "chord":
            chord_tokens = model_tokens
            melody_tokens = context_tokens
        else:
            chord_tokens = context_tokens
            melody_tokens = model_tokens

        bos_token = torch.full_like(
            model_tokens[:, :1],
            self.bos_token_id,
            device=model_tokens.device,
            dtype=model_tokens.dtype,
        )

        # Combine tokens: [melody, bos (sep), chord]
        input_tokens = torch.cat(
            [
                melody_tokens,
                bos_token,
                chord_tokens,
            ],
            dim=1,
        )
        input_mask = input_tokens != self.pad_token_id
        return input_tokens, input_mask

    @torch.no_grad()
    def forward(
        self,
        samples: Samples,
    ) -> Dict[str, torch.Tensor]:
        # Extract variables from samples
        sequence = samples.sequences
        action_mask = samples.action_mask

        # sequence: [B, T]
        # output_mask: [B, T]

        # Get generated sequence inputs and logits
        input_tokens_gen, input_mask_gen = self.get_inputs_from_sequence(
            sequence
        )
        logits = self.model(input_tokens_gen, input_mask_gen)
        # probability of being positive
        reward = F.softmax(logits, dim=-1)[:, 1]

        return {
            "reward": assign_reward_to_last_token(
                reward.to(sequence.device), action_mask
            )
        }


class ContrastiveRewardRhythmFn(ContrastiveRewardFn):
    def __init__(
        self,
        model: ContrastiveReward,
        tokenizer: HooktheoryTokenizer,
        model_part: str,
    ):
        super().__init__(
            model,
            tokenizer.pad_token,
            tokenizer.bos_token,
            tokenizer.eos_token,
            model_part,
        )
        self.tokenizer = tokenizer
        self.create_rhythm_token_map_table(tokenizer)

    def create_rhythm_token_map_table(self, tokenizer):
        # Create dictionary that contains only rhythm
        num_special_tokens = tokenizer.special_token_range[1] + 1  # inclusive
        # +1 for silence, +2 for on/off
        self.num_tokens = num_special_tokens + 1 + 2
        silence_token = tokenizer.silence_token
        onset_tokens = tokenizer.onset_tokens
        hold_tokens = tokenizer.hold_tokens
        # Create mapping table for rhythm
        rhythm_mapping = {}
        for i in range(num_special_tokens):
            rhythm_mapping[i] = i
        rhythm_mapping[silence_token] = num_special_tokens
        rhythm_mapping[tuple(onset_tokens)] = num_special_tokens + 1
        rhythm_mapping[tuple(hold_tokens)] = num_special_tokens + 2
        self.rhythm_mapping = rhythm_mapping
        self.rhythm_token_map_table = create_table_from_mapping(
            rhythm_mapping, default=-1, dtype=torch.int32
        )

    @torch.no_grad()
    def forward(
        self,
        samples: Samples,
    ) -> Dict[str, torch.Tensor]:
        # Extract variables from samples
        sequence = samples.sequences
        action_mask = samples.action_mask

        # sequence: [B, T]
        # output_mask: [B, T]

        # Now we get device from model. The model is initialized with DeepSpeed strategy.
        device = self.model.device

        model_tokens, context_tokens, model_mask, context_mask = (
            self.get_inputs_from_sequence(sequence)
        )
        if self.model_part == "chord":
            chord_tokens = model_tokens
            chord_mask = model_mask
            melody_tokens = context_tokens
            melody_mask = context_mask
        else:
            melody_tokens = model_tokens
            melody_mask = model_mask
            chord_tokens = context_tokens
            chord_mask = context_mask

        # remap rhythm tokens
        melody_tokens = remap_from_table(
            melody_tokens, self.rhythm_token_map_table
        )
        chord_tokens = remap_from_table(
            chord_tokens, self.rhythm_token_map_table
        )

        chord_embed, melody_embed, _ = self.model(
            chord=chord_tokens.to(device),
            melody=melody_tokens.to(device),
            chord_mask=chord_mask.to(device),
            melody_mask=melody_mask.to(device),
        )

        reward = (chord_embed * melody_embed).sum(-1).to(sequence.device)

        return {"reward": assign_reward_to_last_token(reward, action_mask)}


class DiscriminativeRewardRhythmFn(DiscriminativeRewardFn):
    def __init__(
        self,
        model: DiscriminativeReward,
        tokenizer: HooktheoryTokenizer,
        model_part: str,
    ):
        super().__init__(
            model,
            tokenizer.pad_token,
            tokenizer.bos_token,
            tokenizer.eos_token,
            model_part,
        )
        self.tokenizer = tokenizer
        self.create_rhythm_token_map_table(tokenizer)

    def create_rhythm_token_map_table(self, tokenizer):
        # Create dictionary that contains only rhythm
        num_special_tokens = tokenizer.special_token_range[1] + 1  # inclusive
        # +1 for silence, +2 for on/off
        self.num_tokens = num_special_tokens + 1 + 2
        silence_token = tokenizer.silence_token
        onset_tokens = tokenizer.onset_tokens
        hold_tokens = tokenizer.hold_tokens
        # Create mapping table for rhythm
        rhythm_mapping = {}
        for i in range(num_special_tokens):
            rhythm_mapping[i] = i
        rhythm_mapping[silence_token] = num_special_tokens
        rhythm_mapping[tuple(onset_tokens)] = num_special_tokens + 1
        rhythm_mapping[tuple(hold_tokens)] = num_special_tokens + 2
        self.rhythm_mapping = rhythm_mapping
        self.rhythm_token_map_table = create_table_from_mapping(
            rhythm_mapping, default=-1, dtype=torch.int32
        )

    @torch.no_grad()
    def forward(
        self,
        samples: Samples,
    ) -> Dict[str, torch.Tensor]:
        # Extract variables from samples
        sequence = samples.sequences
        action_mask = samples.action_mask

        # sequence: [B, T]
        # output_mask: [B, T]

        # Get generated sequence inputs and logits
        input_tokens_gen, input_mask_gen = self.get_inputs_from_sequence(
            sequence
        )

        # remap rhythm tokens
        input_tokens_gen = remap_from_table(
            input_tokens_gen, self.rhythm_token_map_table
        )

        logits = self.model(input_tokens_gen, input_mask_gen)
        # probability of being positive
        reward = F.softmax(logits, dim=-1)[:, 1]

        return {
            "reward": assign_reward_to_last_token(
                reward.to(sequence.device), action_mask
            )
        }


class UncondOfflineKLRewardFn(BaseRewardModel):
    """Compute KL between an unconditional model and an offline model.

    The KL is computed on generated sequence for the model part.
    """

    def __init__(
        self,
        offline_model: nn.Module,
        unconditional_model: nn.Module,
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
    ):
        super().__init__()
        self.offline_model = offline_model
        self.unconditional_model = unconditional_model
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.offline_model.eval()
        self.unconditional_model.eval()

    @torch.no_grad()
    def log_probs_from_offline_model(
        self,
        sequences: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        context_tokens, model_tokens, context_mask, model_mask = (
            get_seperated_parts_from_sequence(
                sequences,
                self.pad_token_id,
                self.bos_token_id,
                self.eos_token_id,
            )
        )

        # In x-transformers, mask is True for unmasked tokens
        model_part_logits = self.offline_model(
            context_tokens,
            model_tokens,
            enc_mask=context_mask,
            dec_mask=model_mask,
        )

        # remove the EOS token logits
        model_part_logits = model_part_logits[:, :-1, :]

        logits = torch.zeros(
            (
                sequences.size(0),
                sequences.size(1),
                self.offline_model.dec_num_tokens,
            ),
            device=sequences.device,
        )

        logits[:, ::2, :] = model_part_logits

        log_probs = log_probs_from_logits(logits[:, :-1, :], sequences[:, 1:])

        logits = logits[:, :-1, :]  # remove the last token logits

        return log_probs, logits

    @torch.no_grad()
    def log_probs_from_unconditional_model(
        self,
        sequences: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sequences_uncond = torch.cat(
            [sequences[:, 0][:, None], sequences[:, 1::2]], dim=1
        )
        logits = self.unconditional_model(sequences_uncond, mask=None)
        log_probs = log_probs_from_logits(
            logits[:, :-1, :], sequences_uncond[:, 1:]
        )
        logits = logits[:, :-1, :]  # remove the last token logits
        return log_probs, logits

    @torch.no_grad()
    def forward(
        self,
        samples: Samples,
    ) -> Dict[str, torch.Tensor]:
        # Extract variables from samples
        sequence = samples.sequences
        action_mask = samples.action_mask

        # sequence: [B, T]
        # output_mask: [B, T]

        log_probs_offline, logits_offline = self.log_probs_from_offline_model(
            sequence
        )
        log_probs_uncond, logits_uncond = (
            self.log_probs_from_unconditional_model(sequence)
        )
        kl_uncond_offline = compute_full_kl(
            logits_uncond,
            logits_offline[:, ::2],
        )

        model_part_mask = sequence[:, ::2] != self.pad_token_id
        model_part_mask = model_part_mask[:, 1:]  # remove the first token
        kl_uncond_offline = (kl_uncond_offline * model_part_mask).mean(dim=-1)

        # reward: [B]
        reward = kl_uncond_offline

        # TODO: later make it as per-token reward, reference to EncoderDecoderOfflineAnchor.

        return {
            "reward": assign_reward_to_last_token(
                reward.to(sequence.device), action_mask
            )
        }


class CounterpartUncondOfflineKLRewardFn(UncondOfflineKLRewardFn):
    """Compute KL between an unconditional model and an offline model.

    The KL is computed on generated sequence for the counterpart.
    """

    @torch.no_grad()
    def forward(
        self,
        samples: Samples,
    ) -> Dict[str, torch.Tensor]:
        # Extract variables from samples
        sequence = samples.sequences
        action_mask = samples.action_mask

        sequence_counterpart = sequences_order_to_counterpart(sequence)
        log_probs_offline, logits_offline = self.log_probs_from_offline_model(
            sequence_counterpart
        )
        log_probs_uncond, logits_uncond = (
            self.log_probs_from_unconditional_model(sequence_counterpart)
        )
        kl_uncond_offline = compute_full_kl(
            logits_uncond,
            logits_offline[:, ::2],
        )

        counterpart_mask = sequence[:, ::2] != self.pad_token_id
        counterpart_mask = counterpart_mask[:, 1:]  # remove the first token
        kl_uncond_offline = (kl_uncond_offline * counterpart_mask).mean(dim=-1)

        # reward: [B]
        reward = kl_uncond_offline

        # TODO: later make it as per-token reward, reference to EncoderDecoderOfflineAnchor.

        return {
            "reward": assign_reward_to_last_token(
                reward.to(sequence.device), action_mask
            )
        }


class CounterpartUncondOfflineNLLDiffRewardFn(
    CounterpartUncondOfflineKLRewardFn
):
    """Compute NLL difference between an unconditional model and an offline model.

    The NLL is computed on generated sequence for counterpart.
    """

    @torch.no_grad()
    def forward(
        self,
        samples: Samples,
    ) -> Dict[str, torch.Tensor]:
        # Extract variables from samples
        sequence = samples.sequences
        action_mask = samples.action_mask

        sequence_counterpart = sequences_order_to_counterpart(sequence)
        log_probs_offline, logits_offline = self.log_probs_from_offline_model(
            sequence_counterpart
        )
        log_probs_uncond, logits_uncond = (
            self.log_probs_from_unconditional_model(sequence_counterpart)
        )

        # Create masks for the model part (every other token)
        uncond_mask = action_mask[:, ::2]
        nll_uncond = masked_mean(log_probs_uncond, uncond_mask, dim=-1)
        nll_offline = masked_mean(log_probs_offline, action_mask, dim=-1)

        reward = nll_uncond - nll_offline

        return {
            "reward": assign_reward_to_last_token(
                reward.to(sequence.device), action_mask
            )
        }


class UncondOfflineNLLDiffRewardFn(UncondOfflineKLRewardFn):
    """Compute NLL difference between an unconditional model and an offline model.

    The NLL is computed on generated sequence for the model part.
    """

    @torch.no_grad()
    def forward(
        self,
        samples: Samples,
    ) -> Dict[str, torch.Tensor]:
        # Extract variables from samples
        sequence = samples.sequences
        action_mask = samples.action_mask

        # sequence: [B, T]
        # output_mask: [B, T]

        log_probs_offline, logits_offline = self.log_probs_from_offline_model(
            sequence
        )
        log_probs_uncond, logits_uncond = (
            self.log_probs_from_unconditional_model(sequence)
        )

        # Create masks for the model part (every other token)
        uncond_mask = action_mask[:, ::2]
        nll_uncond = masked_mean(log_probs_uncond, uncond_mask, dim=-1)
        nll_offline = masked_mean(log_probs_offline, action_mask, dim=-1)

        reward = nll_uncond - nll_offline

        return {
            "reward": assign_reward_to_last_token(
                reward.to(sequence.device), action_mask
            )
        }


class UncondOfflineKLAdaptationRewardFn(BaseRewardModel):
    """Compute KL adaptation reward.

    If model part is chord:
    (KL(C) - KL(M)) / (KL(C) + KL(M))

    KL(C) = KL(p(C) || p(C|M)): Uncond offline KL
    KL(M) = KL(p(M) || p(M|C)): Counterpart Uncond offline KL

    <0: melody adapts more
    >0: chord adapts more

    The KL is computed on generated sequence.
    """

    def __init__(
        self,
        unconditional_model: nn.Module,
        offline_model: nn.Module,
        counterpart_unconditional_model: nn.Module,
        counterpart_offline_model: nn.Module,
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
    ):
        super().__init__()
        self.uncond_offline_kl_reward_fn = UncondOfflineKLRewardFn(
            offline_model,
            unconditional_model,
            pad_token_id,
            bos_token_id,
            eos_token_id,
        )
        self.counterpart_uncond_offline_kl_reward_fn = (
            CounterpartUncondOfflineKLRewardFn(
                counterpart_offline_model,
                counterpart_unconditional_model,
                pad_token_id,
                bos_token_id,
                eos_token_id,
            )
        )

    @torch.no_grad()
    def forward(
        self,
        samples: Samples,
    ) -> Dict[str, torch.Tensor]:
        # Extract variables from samples
        sequence = samples.sequences
        action_mask = samples.action_mask

        # sum across time
        # TODO: later make it as per-token reward, reference to EncoderDecoderOfflineAnchor.
        uncond_offline_kl = self.uncond_offline_kl_reward_fn(samples)[
            "reward"
        ].sum(-1)
        counterpart_uncond_offline_kl = (
            self.counterpart_uncond_offline_kl_reward_fn(samples)["reward"].sum(
                -1
            )
        )

        reward = (uncond_offline_kl - counterpart_uncond_offline_kl) / (
            uncond_offline_kl + counterpart_uncond_offline_kl
        )

        return {
            "reward": assign_reward_to_last_token(
                reward.to(sequence.device), action_mask
            ),
            "uncond_offline_kl": uncond_offline_kl,
            "counterpart_uncond_offline_kl": counterpart_uncond_offline_kl,
        }


class GAILDiscriminativeRewardFn(BaseRewardModel):
    def __init__(
        self,
        model: DiscriminativeReward,
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
        model_part: str,
        reward_formulation: str = "prob",
    ):
        super().__init__()
        self.model = model
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.model_part = model_part
        if model_part not in ["chord", "melody"]:
            raise ValueError(
                f"model_part must be either 'chord' or 'melody', got {model_part}"
            )
        self.reward_formulation = reward_formulation

    def get_inputs_from_sequence(
        self,
        sequence: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _sequence = sequence.clone()
        # remove eos
        _sequence[_sequence == self.eos_token_id] = self.pad_token_id
        # remove bos
        _sequence = _sequence[:, 1:]
        model_tokens = _sequence[:, ::2]

        tokens = add_bos_to_sequence(model_tokens, self.bos_token_id)
        mask = tokens != self.pad_token_id
        return tokens, mask

    @torch.no_grad()
    def forward(
        self,
        samples: Samples,
    ) -> Dict[str, torch.Tensor]:
        # Extract variables from samples
        sequence = samples.sequences
        action_mask = samples.action_mask

        # sequence: [B, T]
        # action_mask: [B, T]

        tokens_gen, mask_gen = self.get_inputs_from_sequence(sequence)
        logits = self.model(tokens_gen, mask_gen)
        # probability of being positive
        if self.reward_formulation == "prob":
            reward = F.softmax(logits, dim=-1)[:, 1]
        elif self.reward_formulation == "logits":
            reward = logits[:, 1]
        elif self.reward_formulation == "logits_prob_log":
            prob = F.softmax(logits, dim=-1)[:, 1]
            reward = -torch.log(1 - prob + 1e-12)
        else:
            raise ValueError(
                f"Invalid reward formulation: {self.reward_formulation}"
            )
        return {
            "reward": assign_reward_to_last_token(
                reward.to(sequence.device), action_mask
            )
        }
