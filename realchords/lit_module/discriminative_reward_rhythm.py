"""Lightning module for training the encoder-decoder generative model."""

import argbind
import torch
import numpy as np
import torch.nn.functional as F

from realchords.base_trainer import BaseLightningModel
from realchords.model.reward_model import DiscriminativeReward
from realchords.dataset.weighted_joint_dataset import (
    create_weighted_joint_dataset,
    get_dataloader,
)
from realchords.utils.sequence_utils import (
    add_bos_to_sequence,
    add_eos_to_sequence,
)
from realchords.utils.sequence_utils import (
    create_table_from_mapping,
    remap_from_table,
)

from functools import partial

GROUP = __file__
# Function for binding things to parser only when this file is loaded
bind = partial(argbind.bind, group=GROUP)

DiscriminativeReward = bind(DiscriminativeReward)
AdamW = bind(torch.optim.AdamW)
get_dataloader = bind(get_dataloader, without_prefix=True)
create_weighted_joint_dataset = bind(
    create_weighted_joint_dataset, without_prefix=True
)


def compute_metrics(pred, label):
    """
    Computes precision, recall, and F1 score for binary predictions.

    Args:
    pred (torch.Tensor): Tensor of shape [N, num_class], binary predictions (0 or 1).
    label (torch.Tensor): Tensor of shape [N, num_class], binary labels (0 or 1).

    Returns:
    precision (torch.Tensor): Precision score for each class.
    recall (torch.Tensor): Recall score for each class.
    f1 (torch.Tensor): F1 score for each class.
    """
    # Ensure inputs are binary
    pred = pred.int()
    label = label.int()

    # True positives (TP), False positives (FP), and False negatives (FN)
    TP = (pred * label).sum(dim=0)
    FP = (pred * (1 - label)).sum(dim=0)
    FN = ((1 - pred) * label).sum(dim=0)

    # Precision = TP / (TP + FP), avoid division by zero
    precision = TP / (TP + FP + 1e-8)

    # Recall = TP / (TP + FN), avoid division by zero
    recall = TP / (TP + FN + 1e-8)

    # F1 = 2 * (precision * recall) / (precision + recall), avoid division by zero
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    return precision, recall, f1


@bind(without_prefix=True)
class LitDiscriminativeRewardRhythm(BaseLightningModel):
    def __init__(
        self,
        compile: bool = True,
        sample_interval: int = 1000,
        max_log_examples: int = 8,
    ):
        super(LitDiscriminativeRewardRhythm, self).__init__()

        # Create dataloaders
        train_dataset = create_weighted_joint_dataset(split="train")
        val_dataset = create_weighted_joint_dataset(split="valid")
        self.train_dataloader = get_dataloader(train_dataset)
        self.val_dataloader = get_dataloader(val_dataset, shuffle=False)

        self.sample_interval = sample_interval
        self.max_log_examples = max_log_examples
        tokenizer = train_dataset.tokenizer
        self.pad_token = tokenizer.pad_token
        self.eos_token = tokenizer.eos_token
        self.bos_token = tokenizer.bos_token
        self.tokenizer = tokenizer
        self.model_part = train_dataset.model_part

        self.create_rhythm_token_map_table(tokenizer)

        self.model = DiscriminativeReward(
            num_tokens=self.num_tokens,
        )

        self.ce_loss = torch.nn.CrossEntropyLoss()

        if compile:
            self.model = torch.compile(self.model)

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
            rhythm_mapping, default=-1, dtype=torch.int32, device=self.device
        )
        self.onset_token = num_special_tokens + 1
        self.hold_token = num_special_tokens + 2
        self.silence_token = num_special_tokens

    # def random_permute(self, melody_tokens, chord_tokens):
    #     permute_indices = torch.randperm(melody_tokens.size(0))
    #     permuted_melody_tokens = melody_tokens[permute_indices]

    #     # positive samples & negative samples
    #     chord_tokens = torch.cat([chord_tokens, chord_tokens], dim=0)
    #     melody_tokens = torch.cat(
    #         [melody_tokens, permuted_melody_tokens], dim=0
    #     )

    #     # We need to make sure the permuted samples have equal length.
    #     # Otherwise the model can easily learn to cheat.

    #     # remove bos and eos
    #     chord_tokens = chord_tokens[:, 1:-1]
    #     chord_tokens[chord_tokens == self.eos_token] = self.pad_token
    #     melody_tokens = melody_tokens[:, 1:-1]
    #     melody_tokens[melody_tokens == self.eos_token] = self.pad_token

    #     chord_token_mask = chord_tokens != self.pad_token
    #     melody_token_mask = melody_tokens != self.pad_token
    #     joint_mask = chord_token_mask & melody_token_mask

    #     chord_tokens = chord_tokens * joint_mask
    #     melody_tokens = melody_tokens * joint_mask

    #     # add bos and eos back
    #     chord_tokens = add_bos_to_sequence(chord_tokens, self.bos_token)
    #     chord_tokens = add_eos_to_sequence(
    #         chord_tokens, self.pad_token, self.eos_token
    #     )
    #     melody_tokens = add_bos_to_sequence(melody_tokens, self.bos_token)
    #     melody_tokens = add_eos_to_sequence(
    #         melody_tokens, self.pad_token, self.eos_token
    #     )
    #     chord_mask = chord_tokens != self.pad_token
    #     melody_mask = melody_tokens != self.pad_token
    #     return chord_tokens, melody_tokens, chord_mask, melody_mask

    def random_shift_single(self, tokens, shift_amount, total_length):

        # 1. shift tokens
        first_token = tokens[0]
        tokens = torch.cat(
            [
                torch.full(
                    (shift_amount,),
                    self.pad_token,
                    device=tokens.device,
                    dtype=tokens.dtype,
                ),
                tokens,
            ]
        )

        # 2. extend note length in the beginning
        if first_token == self.silence_token:
            # If initial token is silence, we extend the silence
            tokens[:shift_amount] = self.silence_token
        elif first_token == self.onset_token:
            # If initial token is onset, we extend the note length
            tokens[0] = self.onset_token
            tokens[1:shift_amount] = self.hold_token
            tokens[shift_amount] = self.hold_token
        elif first_token == self.hold_token:
            # If initial token is hold, we extend the hold
            tokens[0] = self.onset_token
            tokens[1:shift_amount] = self.hold_token
        else:
            raise ValueError(f"Invalid initial token: {first_token}")

        # 3. truncate overall length or pad with pad_token
        if tokens.size(0) > total_length:
            tokens = tokens[:total_length]
        else:
            tokens = torch.cat(
                [
                    tokens,
                    torch.full(
                        (total_length - tokens.size(0),),
                        self.pad_token,
                        device=tokens.device,
                        dtype=tokens.dtype,
                    ),
                ]
            )

        tokens = tokens[None]
        return tokens

    def create_random_shift_negative(
        self, melody_tokens_single, chord_tokens_single, length
    ):
        # random shift [1, 2, 3] frames
        shift_amount = torch.randint(1, 4, (1,)).item()
        shift_part = np.random.choice(["chord", "melody"])
        if shift_part == "melody":
            melody_tokens_negative = self.random_shift_single(
                melody_tokens_single, shift_amount, length
            )
            chord_tokens_negative = chord_tokens_single[None]
        else:
            melody_tokens_negative = melody_tokens_single[None]
            chord_tokens_negative = self.random_shift_single(
                chord_tokens_single, shift_amount, length
            )
        return melody_tokens_negative, chord_tokens_negative

    def create_random_permute_negative(self, melody_tokens, chord_tokens):
        batch_size = melody_tokens.shape[0]
        random_pair_index = np.random.choice(batch_size, 2, replace=False)

        chord_tokens_negative = chord_tokens[random_pair_index[0]][None]
        melody_tokens_negative = melody_tokens[random_pair_index[1]][None]

        # Make sure the negative samples have equal length
        chord_token_mask = chord_tokens_negative != self.pad_token
        melody_token_mask = melody_tokens_negative != self.pad_token
        joint_mask = chord_token_mask & melody_token_mask

        chord_tokens_negative = chord_tokens_negative * joint_mask
        melody_tokens_negative = melody_tokens_negative * joint_mask

        return melody_tokens_negative, chord_tokens_negative

    def create_examples(self, melody_tokens_gt, chord_tokens_gt):
        """
        Create positive and negative examples.
        """
        # remove bos and eos
        melody_tokens_gt = melody_tokens_gt[:, 1:-1]
        chord_tokens_gt = chord_tokens_gt[:, 1:-1]
        melody_tokens_gt[melody_tokens_gt == self.eos_token] = self.pad_token
        chord_tokens_gt[chord_tokens_gt == self.eos_token] = self.pad_token

        batch_size, length = chord_tokens_gt.shape

        melody_tokens_negative_all = []
        chord_tokens_negative_all = []
        for i in range(batch_size):
            shift_method = np.random.choice(["shift", "permute"], p=[0.5, 0.5])
            if shift_method == "shift":
                melody_tokens_negative, chord_tokens_negative = (
                    self.create_random_shift_negative(
                        melody_tokens_gt[i], chord_tokens_gt[i], length
                    )
                )
            elif shift_method == "permute":
                melody_tokens_negative, chord_tokens_negative = (
                    self.create_random_permute_negative(
                        melody_tokens_gt, chord_tokens_gt
                    )
                )
            else:
                raise ValueError(f"Invalid shift method: {shift_method}")
            melody_tokens_negative_all.append(melody_tokens_negative)
            chord_tokens_negative_all.append(chord_tokens_negative)

        melody_tokens = torch.cat(
            [melody_tokens_gt, torch.cat(melody_tokens_negative_all)]
        )
        chord_tokens = torch.cat(
            [chord_tokens_gt, torch.cat(chord_tokens_negative_all)]
        )

        # add bos to the beginning
        melody_tokens = add_bos_to_sequence(melody_tokens, self.bos_token)
        chord_tokens = add_bos_to_sequence(chord_tokens, self.bos_token)

        # add eos to the end
        melody_tokens = add_eos_to_sequence(
            melody_tokens, self.pad_token, self.eos_token
        )
        chord_tokens = add_eos_to_sequence(
            chord_tokens, self.pad_token, self.eos_token
        )

        return (
            melody_tokens,
            chord_tokens,
        )

    def get_inputs(self, batch):
        # assume inputs are melody and targets are chords
        melody_tokens_gt = batch["inputs"]
        chord_tokens_gt = batch["targets"]

        # remap tokens to rhythm tokens
        melody_tokens_gt = remap_from_table(
            melody_tokens_gt, self.rhythm_token_map_table
        )
        chord_tokens_gt = remap_from_table(
            chord_tokens_gt, self.rhythm_token_map_table
        )

        # create batch of positive and negative samples
        melody_tokens, chord_tokens = self.create_examples(
            melody_tokens_gt, chord_tokens_gt
        )

        # labels
        labels = torch.cat(
            [
                torch.ones(
                    melody_tokens.size(0) // 2,
                    dtype=torch.long,
                    device=melody_tokens.device,
                ),
                torch.zeros(
                    melody_tokens.size(0) // 2,
                    dtype=torch.long,
                    device=melody_tokens.device,
                ),
            ],
            dim=0,
        )

        # input tokens = [melody, bos (sep), chord]
        # input mask = [melody_mask, 1, chord_mask]
        input_tokens = torch.cat(
            [
                melody_tokens,
                torch.full_like(
                    melody_tokens[:, :1],
                    self.bos_token,
                    device=melody_tokens.device,
                    dtype=melody_tokens.dtype,
                ),
                chord_tokens,
            ],
            dim=1,
        )
        input_mask = input_tokens != self.pad_token

        assert input_tokens.shape[0] == labels.shape[0]

        return input_tokens, input_mask, labels

    def training_step(self, batch, batch_idx):
        """
        A single step during training. Calculates the loss for the batch and logs it.
        """
        inputs, input_mask, labels = self.get_inputs(batch)
        logits = self.model(inputs, input_mask)
        loss = self.ce_loss(logits, labels)
        # Log training loss (W&B logging included through self.log)
        self._log_dict({"train/loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        """
        A single step during validation. Calculates the loss for the batch and logs it.
        """
        inputs, input_mask, labels = self.get_inputs(batch)
        logits = self.model(inputs, input_mask)
        loss = self.ce_loss(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        precision, recall, f1 = compute_metrics(logits.argmax(dim=1), labels)
        # Log validation loss (step-based logging)
        self._log_dict(
            {
                "val/loss": loss,
                "val/acc": acc,
                "val/precision": precision,
                "val/recall": recall,
                "val/f1": f1,
            }
        )
        return loss

    def configure_optimizers(self):
        """
        Configures and returns the optimizer(s).
        """
        optimizer = AdamW(filter(lambda p: p.requires_grad, self.parameters()))
        return optimizer
