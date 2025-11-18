"""Lightning module for training the encoder-decoder generative model."""

import argbind
import torch

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
class LitDiscriminativeReward(BaseLightningModel):
    def __init__(
        self,
        compile: bool = True,
        sample_interval: int = 1000,
        max_log_examples: int = 8,
    ):
        super(LitDiscriminativeReward, self).__init__()

        # Create dataloaders
        train_dataset = create_weighted_joint_dataset(split="train")
        val_dataset = create_weighted_joint_dataset(split="valid")
        self.train_dataloader = get_dataloader(train_dataset)
        self.val_dataloader = get_dataloader(val_dataset, shuffle=False)

        self.sample_interval = sample_interval
        self.max_log_examples = max_log_examples
        tokenizer = train_dataset.tokenizer
        self.num_tokens = tokenizer.num_tokens
        self.pad_token = tokenizer.pad_token
        self.eos_token = tokenizer.eos_token
        self.bos_token = tokenizer.bos_token
        self.tokenizer = tokenizer
        self.model_part = train_dataset.model_part

        self.model = DiscriminativeReward(
            num_tokens=self.num_tokens,
        )

        self.ce_loss = torch.nn.CrossEntropyLoss()

        if compile:
            self.model = torch.compile(self.model)

    def get_inputs(self, batch):
        # assume inputs are melody and targets are chords
        melody_tokens = batch["inputs"]
        chord_tokens = batch["targets"]

        permute_indices = torch.randperm(melody_tokens.size(0))
        permuted_melody_tokens = melody_tokens[permute_indices]

        # positive samples & negative samples
        chord_tokens = torch.cat([chord_tokens, chord_tokens], dim=0)
        melody_tokens = torch.cat(
            [melody_tokens, permuted_melody_tokens], dim=0
        )

        # We need to make sure the permuted samples have equal length.
        # Otherwise the model can easily learn to cheat.

        # remove bos and eos
        chord_tokens = chord_tokens[:, 1:-1]
        chord_tokens[chord_tokens == self.eos_token] = self.pad_token
        melody_tokens = melody_tokens[:, 1:-1]
        melody_tokens[melody_tokens == self.eos_token] = self.pad_token

        chord_token_mask = chord_tokens != self.pad_token
        melody_token_mask = melody_tokens != self.pad_token
        joint_mask = chord_token_mask & melody_token_mask

        chord_tokens = chord_tokens * joint_mask
        melody_tokens = melody_tokens * joint_mask

        # add bos and eos back
        chord_tokens = add_bos_to_sequence(chord_tokens, self.bos_token)
        chord_tokens = add_eos_to_sequence(
            chord_tokens, self.pad_token, self.eos_token
        )
        melody_tokens = add_bos_to_sequence(melody_tokens, self.bos_token)
        melody_tokens = add_eos_to_sequence(
            melody_tokens, self.pad_token, self.eos_token
        )
        chord_mask = chord_tokens != self.pad_token
        melody_mask = melody_tokens != self.pad_token

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
        input_mask = torch.cat(
            [
                melody_mask,
                torch.ones_like(
                    melody_mask[:, :1],
                    device=melody_mask.device,
                    dtype=melody_mask.dtype,
                ),
                chord_mask,
            ],
            dim=1,
        )

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
