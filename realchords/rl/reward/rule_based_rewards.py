"""Rule-based reward functions for ReaLchords.

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

from typing import Dict
from itertools import groupby

import torch

from realchords.rl.reward.base import BaseRewardFn
from realchords.dataset.hooktheory_tokenizer import (
    HooktheoryTokenizer,
    to_midi_pitch,
)
from realchords.rl.utils import assign_reward_to_last_token
from realchords.rl.experience_maker import Samples


class ConstantReward(BaseRewardFn):
    def __init__(self, reward: float = 0.0):
        super().__init__()
        self.reward = reward

    def forward(
        self,
        samples: Samples,
    ) -> Dict[str, torch.Tensor]:
        # Extract variables from samples
        sequence = samples.sequences
        action_mask = samples.action_mask

        # sequence: [B, T]
        # output_mask: [B, T]
        device = sequence.device
        batch_size = sequence.shape[0]
        reward = torch.full((batch_size,), self.reward, device=device)
        return {
            "reward": assign_reward_to_last_token(reward, action_mask),
        }


class EarlyStopPenalty(BaseRewardFn):
    def __init__(self, pad_token_id: int, bos_token_id: int, eos_token_id: int):
        """Penalize the ReaLchords model if it stops early."""
        super().__init__()
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    def forward(
        self,
        samples: Samples,
    ) -> Dict[str, torch.Tensor]:
        # Extract variables from samples
        sequence = samples.sequences
        action_mask = samples.action_mask

        # sequence: [B, T]
        # output_mask: [B, T]
        device = sequence.device
        sequence = sequence.cpu()
        batch_size = sequence.shape[0]
        per_batch_rewards = []
        for i in range(batch_size):
            sequence_single = sequence[i][1:]  # Skip the first bos token
            model_part = sequence_single[::2]  # Get the model part
            context_part = sequence_single[1::2]  # Get the context part
            context_length = (context_part != self.pad_token_id).sum()

            # check the position of first eos token or pad token of the model part
            first_eos = (model_part == self.eos_token_id).nonzero()
            if len(first_eos) > 0:
                first_pos = first_eos[0]
            else:
                first_pos = None
            first_pad = (model_part == self.pad_token_id).nonzero()
            if len(first_pad) > 0:
                first_pad_pos = first_pad[0]
                if first_pos is None or first_pad_pos < first_pos:
                    first_pos = first_pad_pos
            if first_pos is not None and first_pos < context_length:
                # negative reward if the model stops early
                reward_single = first_pos - context_length
            else:
                reward_single = 0

            per_batch_rewards.append(reward_single)

        # [B]
        per_batch_rewards = torch.tensor(per_batch_rewards).to(device).float()

        # scale by 10 to make it comparable with other rewards
        reward = per_batch_rewards / 10

        return {
            "reward": assign_reward_to_last_token(reward, action_mask),
            "num_early_stop_tokens": -1 * per_batch_rewards,
        }


class MajChordReward(BaseRewardFn):
    """Sanity check reward for promoting major chords."""

    def __init__(
        self,
        tokenizer: HooktheoryTokenizer,
    ):
        super().__init__()
        self.tokenizer = tokenizer

    def forward(
        self,
        samples: Samples,
    ) -> Dict[str, torch.Tensor]:
        # Extract variables from samples
        sequence = samples.sequences
        action_mask = samples.action_mask

        # sequence: [B, T]
        # output_mask: [B, T]
        device = sequence.device
        sequence = sequence.cpu()
        batch_size = sequence.shape[0]
        per_batch_rewards = []
        for i in range(batch_size):
            single_reward = 0
            sequence_single = sequence[i][1:]  # Skip the first bos token
            model_part = sequence_single[::2]  # Get the model part
            for j in range(len(model_part)):
                model_token = int(model_part[j])
                model_token_name = self.tokenizer.id_to_name[model_token]
                # This won't get all the major chords, but it's a sanity check
                if "maj" in model_token_name:
                    single_reward += 1
                elif model_token_name != "PAD":
                    single_reward -= 1

            num_tokens = (model_part != self.tokenizer.pad_token).sum()
            single_reward = single_reward / num_tokens
            per_batch_rewards.append(single_reward)

        # [B]
        per_batch_rewards = torch.tensor(per_batch_rewards).to(device).float()

        return {
            "reward": assign_reward_to_last_token(
                per_batch_rewards, action_mask
            ),
        }


class SilencePenalty(BaseRewardFn):
    """Penalize the ReaLchords model if it generates silence for non-silence input.

    To avoid behavior where no chords are generated, we provide a penalty of -1 for each silence token if more than
    4% of frames are silent accompaniment to a non-silent input. This is the rate at which silent accompaniment occurs in the
    training set. We omit this penalty for the first 8 frames (first half-note) to allow early adaptation.
    """

    def __init__(
        self,
        tokenizer: HooktheoryTokenizer,
        min_portion: float = 0.04,
        num_omit_frames: int = 8,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.min_portion = min_portion
        self.silence_token_id = tokenizer.name_to_id["SILENCE"]
        self.num_omit_frames = num_omit_frames

    def forward(
        self,
        samples: Samples,
    ) -> Dict[str, torch.Tensor]:
        # Extract variables from samples
        sequence = samples.sequences
        action_mask = samples.action_mask

        # sequence: [B, T]
        # output_mask: [B, T]
        device = sequence.device
        sequence = sequence.cpu()
        batch_size = sequence.shape[0]
        per_token_rewards = []
        per_batch_silence_counts = []

        for i in range(batch_size):
            sequence_single = sequence[i][1:]  # Skip the first bos token
            model_part = sequence_single[::2]  # Get the model part
            context_part = sequence_single[1::2]  # Get the context part

            # Omit the first n frames to allow early adaptation
            model_part_omit = model_part[self.num_omit_frames :]
            context_part_omit = context_part[self.num_omit_frames :]
            context_silence = context_part_omit == self.silence_token_id
            model_silence = model_part_omit == self.silence_token_id
            model_silence_for_non_silence = model_silence & ~context_silence
            num_frames = (context_part_omit != self.tokenizer.pad_token).sum()
            silence_ratio = model_silence_for_non_silence.sum() / num_frames

            # Calculate per-token rewards
            token_rewards = []
            for j, (model_token, context_token) in enumerate(
                zip(model_part, context_part)
            ):
                if j < self.num_omit_frames:
                    # No penalty for first n frames
                    token_rewards.append(0.0)
                else:
                    # Check if this is a silence token in non-silence context
                    is_silence_in_non_silence = (
                        model_token == self.silence_token_id
                        and context_token != self.silence_token_id
                    )

                    if (
                        is_silence_in_non_silence
                        and silence_ratio > self.min_portion
                    ):
                        token_rewards.append(-1.0)
                    else:
                        token_rewards.append(0.0)

            per_token_rewards.append(token_rewards)
            per_batch_silence_counts.append(
                model_silence_for_non_silence.sum().item()
            )

        # [B, T]
        per_token_rewards = torch.tensor(per_token_rewards).to(device).float()
        per_batch_silence_counts = (
            torch.tensor(per_batch_silence_counts).to(device).float()
        )

        # scale by 10 to make it comparable with other rewards
        sequence_rewards = torch.zeros_like(action_mask, dtype=float)
        sequence_rewards[:, ::2] = per_token_rewards / 10

        return {
            "reward": sequence_rewards,
            "num_silence_tokens": per_batch_silence_counts,
        }


class LongNotePenalty(BaseRewardFn):
    """Penalize the ReaLchords model if it generates long notes.

    The model has a tendency to repeat itself, generating long-held chords. Inspired by repetition penalties used for
    training language models (as in Saleh et al. (2020); Jaques et al. (2019)), we provide a penalty of -1 for each token in a
    chord that is held for longer than 32 frames (8 beats), because chords longer than 8 beats rarely occur in the training set.
    """

    def __init__(self, tokenizer: HooktheoryTokenizer, threshold: int = 32):
        super().__init__()
        self.tokenizer = tokenizer
        self.threshold = threshold

    def forward(
        self,
        samples: Samples,
    ) -> Dict[str, torch.Tensor]:
        # Extract variables from samples
        sequence = samples.sequences
        action_mask = samples.action_mask

        # sequence: [B, T]
        # output_mask: [B, T]
        device = sequence.device
        sequence = sequence.cpu()
        batch_size = sequence.shape[0]
        per_batch_rewards = []
        for i in range(batch_size):
            sequence_single = sequence[i][1:]  # Skip the first bos token
            model_part = sequence_single[::2]  # Get the model part

            # Convert tensor to list of integers for groupby
            model_part = model_part.cpu().numpy().tolist()

            # Calculate penalties using groupby for consecutive tokens
            reward_single = 0
            for token, group in groupby(model_part):
                if token == self.tokenizer.pad_token:
                    break

                group_length = len(list(group))
                # -1 because there will be onset token
                if group_length > self.threshold - 1:
                    reward_single -= group_length

            per_batch_rewards.append(reward_single)

        # [B]
        per_batch_rewards = torch.tensor(per_batch_rewards).to(device).float()

        # scale by 10 to make it comparable with other rewards
        reward = per_batch_rewards / 10

        return {
            "reward": assign_reward_to_last_token(reward, action_mask),
            "num_long_note_tokens": -1
            * per_batch_rewards,  # Convert penalties to positive counts
        }


class InvalidOutputPenalty(BaseRewardFn):
    """Penalize the ReaLchords model if it generates invalid output."""

    def __init__(self, tokenizer: HooktheoryTokenizer, model_part: str):
        super().__init__()
        self.tokenizer = tokenizer
        self.model_part = model_part

    def output_to_midi(self, decoder_pred):
        if self.model_part == "melody":
            melody_frames = decoder_pred[::2]
            chord_frames = decoder_pred[1::2]
            midi = self.tokenizer.decode_to_midi(
                melody_frames=melody_frames,
                chord_frames=chord_frames,
            )
        elif self.model_part == "chord":
            chord_frames = decoder_pred[::2]
            melody_frames = decoder_pred[1::2]
            midi = self.tokenizer.decode_to_midi(
                chord_frames=chord_frames,
                melody_frames=melody_frames,
            )
        return midi

    def forward(
        self,
        samples: Samples,
    ) -> Dict[str, torch.Tensor]:
        # Extract variables from samples
        sequence = samples.sequences
        action_mask = samples.action_mask

        # sequence: [B, T]
        # output_mask: [B, T]
        device = sequence.device
        sequence = sequence.cpu()
        batch_size = sequence.shape[0]

        special_token_range = self.tokenizer.special_token_range
        per_batch_rewards = []
        for i in range(batch_size):
            sequence_single = sequence[i][1:]  # Skip the first bos token
            action_mask_single = action_mask[i]
            # Data mask: the mask for both data and model part.
            data_mask = torch.zeros_like(
                action_mask_single, dtype=bool, device="cpu"
            )
            data_mask[::2] = action_mask_single[::2]
            data_mask[1::2] = action_mask_single[::2]
            sequence_single = sequence_single[data_mask]

            # First check if generation contains special tokens.
            has_special_tokens = sequence_single <= special_token_range[1]
            if has_special_tokens.any():
                reward_single = -10
            else:
                try:
                    midi_gen = self.output_to_midi(sequence_single)
                    reward_single = 0
                except Exception as e:
                    reward_single = -10
            per_batch_rewards.append(reward_single)

        # [B]
        per_batch_rewards = torch.tensor(per_batch_rewards).to(device).float()

        reward = per_batch_rewards
        # Calculate invalid_ratio for logging purpose.
        # Has shape [B] because it will be splitted by OpenRLHF for replay buffer.
        # Later it will be averaged for logging.
        invalid_ratio = (per_batch_rewards < 0).float()

        return {
            "reward": assign_reward_to_last_token(reward, action_mask),
            "invalid_ratio": invalid_ratio,
        }


class RepetitionPenalty(BaseRewardFn):
    """Penalize the ReaLChords model if it generates repetitive output."""

    def __init__(
        self,
        tokenizer: HooktheoryTokenizer,
        model_part: str,
        threshold: int = 4,
    ):
        # In dataset, the longest consecutive chords is 144.
        # top 1 percentile (from largest to smallest) is 5, and top 2 percentile is 4.
        # So we set the threshold to 4.
        super().__init__()
        self.tokenizer = tokenizer
        self.model_part = model_part
        self.threshold = threshold

    def output_to_list(self, decoder_pred):
        if self.model_part == "melody":
            melody_frames = decoder_pred[::2]
            annotations = self.tokenizer.decode_melody_frames(melody_frames)
            gen_list = [
                to_midi_pitch(annotation["octave"], annotation["pitch_class"])
                for annotation in annotations
            ]
        elif self.model_part == "chord":
            chord_frames = decoder_pred[::2]
            annotations = self.tokenizer.decode_chord_frames(chord_frames)
            gen_list = [annotation["chord_name"] for annotation in annotations]
        return gen_list

    def forward(
        self,
        samples: Samples,
    ) -> Dict[str, torch.Tensor]:
        # Extract variables from samples
        sequence = samples.sequences
        action_mask = samples.action_mask

        # sequence: [B, T]
        # output_mask: [B, T]
        device = sequence.device
        sequence = sequence.cpu()
        batch_size = sequence.shape[0]
        per_batch_rewards = []
        for i in range(batch_size):
            sequence_single = sequence[i][1:]  # Skip the first bos token
            try:
                gen_list = self.output_to_list(sequence_single)
            except Exception as e:
                reward_single = -10
                per_batch_rewards.append(reward_single)
                continue
            # Find the longest consecutive sequence of the same value
            reward_single = 0
            for value, group in groupby(gen_list):
                if value in [
                    self.tokenizer.pad_token,
                    self.tokenizer.bos_token,
                    self.tokenizer.eos_token,
                    self.tokenizer.silence_token,
                ]:
                    continue
                group_length = len(list(group))
                if group_length > self.threshold:
                    reward_single -= group_length
            per_batch_rewards.append(reward_single)

        # [B]
        per_batch_rewards = torch.tensor(per_batch_rewards).to(device).float()

        # scale by 10 to make it comparable with other rewards
        per_batch_rewards = per_batch_rewards / 10

        return {
            "reward": assign_reward_to_last_token(
                per_batch_rewards, action_mask
            ),
            "num_repetitive_generations": -1 * per_batch_rewards,
        }
