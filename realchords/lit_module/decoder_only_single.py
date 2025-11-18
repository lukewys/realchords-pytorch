"""Lightning module for training the decoder-only generative model (single task)."""

import argbind
import time
import torch
import torch.nn.functional as F
import wandb
from functools import partial

from realchords.base_trainer import BaseLightningModel
from realchords.model.gen_model import DecoderTransformer
from realchords.utils.lr_scheduler import LinearWarmupCosineDecay
from realchords.dataset.hooktheory_dataloader import (
    HooktheoryDataset,
    get_dataloader,
)
from realchords.dataset.weighted_joint_dataset import (
    create_weighted_joint_dataset,
)
from realchords.utils.sequence_utils import add_eos_to_sequence
from realchords.utils.logging_utils import logger

GROUP = __file__
# Function for binding things to parser only when this file is loaded
bind = partial(argbind.bind, group=GROUP)

DecoderTransformer = bind(DecoderTransformer)
AdamW = bind(torch.optim.AdamW)
get_dataloader = bind(get_dataloader, without_prefix=True)


@bind(without_prefix=True)
class LitDecoderSingle(BaseLightningModel):
    def __init__(
        self,
        compile: bool = True,
        sample_interval: int = 1000,
        max_log_examples: int = 8,
    ):
        super(LitDecoderSingle, self).__init__()

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

        self.model = DecoderTransformer(
            num_tokens=self.num_tokens,
            pad_value=self.pad_token,
        )

        self.max_gen_seq_len = (
            self.model.decoder.max_seq_len - 1
        )  # -1 for the bos token

        if compile:
            self.model = torch.compile(self.model)

    def get_inputs(self, batch):
        dec_inputs = batch["targets"][:, :-1]
        targets = batch["targets"][:, 1:]
        dec_inputs_mask = batch["targets_mask"][:, :-1]
        return dec_inputs, targets, dec_inputs_mask

    def training_step(self, batch, batch_idx):
        """
        A single step during training. Calculates the loss for the batch and logs it.
        """
        # print(f"batch_idx: {batch_idx}, global_step: {self.global_step}")
        dec_inputs, targets, dec_inputs_mask = self.get_inputs(batch)
        # In x-transformers, mask is True for unmasked tokens
        output = self.model(
            dec_inputs,
            mask=dec_inputs_mask,
        )

        loss = F.cross_entropy(
            output.reshape(-1, output.shape[-1]),
            targets.reshape(-1),
            ignore_index=self.pad_token,
        )
        # Log training loss (W&B logging included through self.log)
        self._log_dict({"train/loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        """
        A single step during validation. Calculates the loss for the batch and logs it.
        """
        dec_inputs, targets, dec_inputs_mask = self.get_inputs(batch)
        output = self.model(
            dec_inputs,
            mask=dec_inputs_mask,
        )
        loss = F.cross_entropy(
            output.reshape(-1, output.shape[-1]),
            targets.reshape(-1),
            ignore_index=self.pad_token,
        )

        # Calculate accuracy
        acc = (output.argmax(dim=-1) == targets).float().mean()

        # Log validation loss (step-based logging)
        self._log_dict({"val/loss": loss, "val/acc": acc})

        # Sample from model and log them
        # Only sample for the first validation batch
        if batch_idx == 0 and self.global_step % self.sample_interval == 0:
            self.sample_and_log(targets, batch["song_url"])

        return loss

    def sample_and_log(self, targets, song_urls):
        gen_inputs = torch.full(
            (targets.shape[0], 1),
            self.bos_token,
            dtype=torch.long,
            device=targets.device,
        )
        curr_time = time.time()
        decoder_preds = self.model.generate(
            gen_inputs,
            seq_len=self.max_gen_seq_len,
            cache_kv=True,
            eos_token=self.tokenizer.eos_token,
        )

        # We don't log audio and image into table because there will be bugs
        table_text = wandb.Table(columns=["Index", "Array GT", "Array Gen"])
        for i in range(min(self.max_log_examples, len(targets))):
            decoder_pred = decoder_preds[i].cpu().numpy()
            target = targets[i].cpu().numpy()
            # Convert array to text
            array_pred_text = str(decoder_pred)
            array_gt_text = str(target)
            table_text.add_data(i, array_gt_text, array_pred_text)

            midi_gt = self.output_to_midi(target)
            self.log_midi(midi_gt, suffix=f"gt_{i}")

            try:
                midi_gen = self.output_to_midi(decoder_pred)
                self.log_midi(midi_gen, suffix=f"gen_{i}")

            except Exception as e:
                logger.error(f"Error converting to MIDI: {e}")
                continue

        self.logger.experiment.log({"array_text": table_text})

        table_url = wandb.Table(columns=["Index", "Song URL"])
        for i, song_url in enumerate(song_urls):
            table_url.add_data(i, song_url)
        self.logger.experiment.log({"song_url": table_url})

    def output_to_midi(self, decoder_pred):
        if self.model_part == "melody":
            midi = self.tokenizer.decode_to_midi(
                melody_frames=decoder_pred,
            )
        elif self.model_part == "chord":
            midi = self.tokenizer.decode_to_midi(
                chord_frames=decoder_pred,
            )
        return midi

    def configure_optimizers(self):
        """
        Configures and returns the optimizer(s).
        """
        optimizer = AdamW(filter(lambda p: p.requires_grad, self.parameters()))
        return optimizer
