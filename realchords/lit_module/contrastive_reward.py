"""Lightning module for training the encoder-decoder generative model."""

import argbind
import time

import wandb
import torch

import torch.nn.functional as F

from realchords.base_trainer import BaseLightningModel
from realchords.model.reward_model import ContrastiveReward
from realchords.dataset.weighted_joint_dataset import (
    create_weighted_joint_dataset,
    get_dataloader,
)

from functools import partial

GROUP = __file__
# Function for binding things to parser only when this file is loaded
bind = partial(argbind.bind, group=GROUP)

ContrastiveReward = bind(ContrastiveReward)
AdamW = bind(torch.optim.AdamW)
get_dataloader = bind(get_dataloader, without_prefix=True)
create_weighted_joint_dataset = bind(
    create_weighted_joint_dataset, without_prefix=True
)


@bind(without_prefix=True)
class LitContrastiveReward(BaseLightningModel):
    def __init__(
        self,
        compile: bool = True,
        sample_interval: int = 1000,
        max_log_examples: int = 8,
    ):
        super(LitContrastiveReward, self).__init__()

        # Create dataloaders
        train_dataset = create_weighted_joint_dataset(split="train")
        val_dataset = create_weighted_joint_dataset(split="valid")
        self.train_dataloader = get_dataloader(train_dataset)
        self.val_dataloader = get_dataloader(val_dataset, shuffle=False)

        self.sample_interval = sample_interval
        self.max_log_examples = max_log_examples
        tokenizer = train_dataset.tokenizer
        self.pad_token = tokenizer.pad_token
        self.num_tokens = tokenizer.num_tokens
        self.bos_token = tokenizer.bos_token
        self.tokenizer = tokenizer
        self.model_part = train_dataset.model_part

        self.model = ContrastiveReward(
            num_tokens=self.num_tokens,
        )

        if compile:
            self.model = torch.compile(self.model)

    def get_inputs(self, batch):
        # assume inputs are melody and targets are chords
        melody_tokens = batch["inputs"]
        chord_tokens = batch["targets"]
        melody_mask = batch["inputs_mask"]
        chord_mask = batch["targets_mask"]
        return melody_tokens, chord_tokens, melody_mask, chord_mask

    def contrastive_loss(self, melody_embed, chord_embed, logit_scale):
        """
        Calculate the contrastive loss between the melody and chord embeddings.
        """
        logits_per_melody = logit_scale * melody_embed @ chord_embed.T
        logits_per_chord = logits_per_melody.T

        # calculated ground-truth
        num_logits = logits_per_melody.shape[0]
        labels = torch.arange(
            num_logits, device=melody_embed.device, dtype=torch.long
        )
        total_loss = (
            F.cross_entropy(logits_per_melody, labels)
            + F.cross_entropy(logits_per_chord, labels)
        ) / 2
        return total_loss

    def training_step(self, batch, batch_idx):
        """
        A single step during training. Calculates the loss for the batch and logs it.
        """
        melody_tokens, chord_tokens, melody_mask, chord_mask = self.get_inputs(
            batch
        )
        # In x-transformers, mask is True for unmasked tokens
        chord_embed, melody_embed, logit_scale = self.model(
            chord=chord_tokens,
            melody=melody_tokens,
            chord_mask=chord_mask,
            melody_mask=melody_mask,
        )

        loss = self.contrastive_loss(melody_embed, chord_embed, logit_scale)
        # Log training loss (W&B logging included through self.log)
        self._log_dict({"train/loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        """
        A single step during validation. Calculates the loss for the batch and logs it.
        """
        melody_tokens, chord_tokens, melody_mask, chord_mask = self.get_inputs(
            batch
        )
        # In x-transformers, mask is True for unmasked tokens
        chord_embed, melody_embed, logit_scale = self.model(
            chord=chord_tokens,
            melody=melody_tokens,
            chord_mask=chord_mask,
            melody_mask=melody_mask,
        )

        loss = self.contrastive_loss(melody_embed, chord_embed, logit_scale)
        # Log validation loss (step-based logging)
        self._log_dict({"val/loss": loss})

        # Sample from model and log them

        return loss

    def configure_optimizers(self):
        """
        Configures and returns the optimizer(s).
        """
        optimizer = AdamW(filter(lambda p: p.requires_grad, self.parameters()))
        return optimizer
