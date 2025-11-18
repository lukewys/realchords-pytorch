"""Lightning module for training the encoder-decoder generative model."""

import argbind
import time

import wandb
import torch

import torch.nn.functional as F

from realchords.base_trainer import BaseLightningModel
from realchords.utils.lr_scheduler import LinearWarmupCosineDecay
from realchords.utils.eval_utils import evaluate_note_in_chord_ratio
from realchords.model.gen_model import EncoderDecoderTransformer
from realchords.dataset.weighted_joint_dataset import (
    create_weighted_joint_dataset,
    get_dataloader,
)
from realchords.utils.sequence_utils import (
    add_eos_to_sequence,
    add_bos_to_sequence,
)
import random
from functools import partial
from realchords.utils.logging_utils import logger

GROUP = __file__
# Function for binding things to parser only when this file is loaded
bind = partial(argbind.bind, group=GROUP)

EncoderDecoderTransformer = bind(EncoderDecoderTransformer)
AdamW = bind(torch.optim.AdamW)
get_dataloader = bind(get_dataloader, without_prefix=True)
create_weighted_joint_dataset = bind(
    create_weighted_joint_dataset, without_prefix=True
)


@bind(without_prefix=True)
class LitEncoderDecoder(BaseLightningModel):
    def __init__(
        self,
        compile: bool = True,
        sample_interval: int = 1000,
        max_log_examples: int = 8,
        random_truncate: bool = False,
    ):
        super(LitEncoderDecoder, self).__init__()

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
        self.eos_token = tokenizer.eos_token
        self.tokenizer = tokenizer
        self.model_part = train_dataset.model_part

        self.model = EncoderDecoderTransformer(
            enc_num_tokens=self.num_tokens,
            dec_num_tokens=self.num_tokens,
            pad_value=self.pad_token,
        )

        self.max_gen_seq_len = (
            self.model.decoder.max_seq_len - 2
        )  # -2 for the bos & eos token

        if compile:
            self.model = torch.compile(self.model)

        # Whether to randomly truncate inputs during training
        # We enable this to train model for realjam, because at the
        # beginning of live generation, the input length is short.
        self.random_truncate = random_truncate

    def truncate(self, batch):
        enc_inputs = batch["inputs"]
        targets = batch["targets"]

        # remove bos and eos
        enc_inputs = enc_inputs[:, 1:-1]
        targets = targets[:, 1:-1]
        enc_inputs[enc_inputs == self.eos_token] = 0
        targets[targets == self.eos_token] = 0

        random_num = random.random()
        # 20% chance to truncate to 28
        if random_num < 0.2:
            enc_inputs[:, 28:] = self.pad_token
            targets[:, 28:] = self.pad_token

        # 20% chance to random truncate
        elif random_num < 0.4:
            truncate_len = random.randint(28, len(enc_inputs) - 1)
            enc_inputs[:, truncate_len:] = self.pad_token
            targets[:, truncate_len:] = self.pad_token

        # 60% chance to do nothing
        else:
            pass

        # add bos and eos
        enc_inputs = add_bos_to_sequence(enc_inputs, self.bos_token)
        enc_inputs = add_eos_to_sequence(
            enc_inputs, self.pad_token, self.eos_token
        )
        targets = add_bos_to_sequence(targets, self.bos_token)
        targets = add_eos_to_sequence(targets, self.pad_token, self.eos_token)

        inputs_mask = enc_inputs != self.pad_token
        targets_mask = targets != self.pad_token

        batch["inputs"] = enc_inputs
        batch["targets"] = targets
        batch["inputs_mask"] = inputs_mask
        batch["targets_mask"] = targets_mask

        return batch

    def get_inputs(self, batch):
        enc_inputs = batch["inputs"]
        dec_inputs = batch["targets"][:, :-1]
        targets = batch["targets"][:, 1:]
        enc_inputs_mask = batch["inputs_mask"]
        dec_inputs_mask = batch["targets_mask"][:, :-1]
        return enc_inputs, dec_inputs, targets, enc_inputs_mask, dec_inputs_mask

    def training_step(self, batch, batch_idx):
        """
        A single step during training. Calculates the loss for the batch and logs it.
        """
        # print(f"batch_idx: {batch_idx}, global_step: {self.global_step}")
        if self.random_truncate:
            batch = self.truncate(batch)
        (
            enc_inputs,
            dec_inputs,
            targets,
            enc_inputs_mask,
            dec_inputs_mask,
        ) = self.get_inputs(batch)
        # In x-transformers, mask is True for unmasked tokens
        output = self.model(
            enc_inputs,
            dec_inputs,
            enc_mask=enc_inputs_mask,
            dec_mask=dec_inputs_mask,
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
        (
            enc_inputs,
            dec_inputs,
            targets,
            enc_inputs_mask,
            dec_inputs_mask,
        ) = self.get_inputs(batch)
        output = self.model(
            enc_inputs,
            dec_inputs,
            enc_mask=enc_inputs_mask,
            dec_mask=dec_inputs_mask,
        )
        loss = F.cross_entropy(
            output.reshape(-1, output.shape[-1]),
            targets.reshape(-1),
            ignore_index=self.pad_token,
        )

        # Calculate accuracy
        acc = (output.argmax(dim=-1) == targets).float().mean()

        # Also log NLL
        output_log_prob = F.log_softmax(output, dim=-1)
        nll = F.nll_loss(
            output_log_prob.reshape(-1, output_log_prob.shape[-1]),
            targets.reshape(-1),
            ignore_index=self.pad_token,
        )

        # Log validation loss (step-based logging)
        self._log_dict({"val/loss": loss, "val/acc": acc, "val/nll": nll})

        # Sample from model and log them
        # Only sample for the first validation batch
        if batch_idx == 0 and self.global_step % self.sample_interval == 0:
            self.sample_and_log(
                enc_inputs, enc_inputs_mask, targets, batch["song_url"]
            )

        return loss

    def sample_and_log(self, enc_inputs, enc_inputs_mask, targets, song_urls):
        gen_inputs = torch.full(
            (enc_inputs.shape[0], 1),
            self.bos_token,
            dtype=torch.long,
            device=enc_inputs.device,
        )
        curr_time = time.time()
        decoder_preds = self.model.generate(
            enc_inputs,
            gen_inputs,
            seq_len=self.max_gen_seq_len,
            mask=enc_inputs_mask,
            cache_kv=True,
            eos_token=self.tokenizer.eos_token,
        )

        # Calculate note in chord ratio
        eval_input_gt_all = []
        eval_input_pred_all = []
        overall_length = decoder_preds.shape[1] * 2

        for i in range(enc_inputs.shape[0]):
            enc_input_single = enc_inputs[i][1:-1].cpu()
            target_single = targets[i][:-1].cpu()
            decoder_pred_single = decoder_preds[i].cpu()

            eval_input_gt = torch.zeros(overall_length)
            eval_input_pred = torch.zeros(overall_length)

            eval_input_gt[::2] = target_single
            eval_input_pred[::2] = decoder_pred_single

            eval_input_gt[1::2] = enc_input_single
            eval_input_pred[1::2] = enc_input_single

            eval_input_gt_all.append(eval_input_gt)
            eval_input_pred_all.append(eval_input_pred)

        eval_input_gt_all = torch.stack(eval_input_gt_all)
        eval_input_pred_all = torch.stack(eval_input_pred_all)

        note_in_chord_ratio_gt = evaluate_note_in_chord_ratio(
            eval_input_gt_all, self.tokenizer, model_part=self.model_part
        )
        note_in_chord_ratio_pred = evaluate_note_in_chord_ratio(
            eval_input_pred_all, self.tokenizer, model_part=self.model_part
        )

        self._log_dict(
            {
                "val/note_in_chord_ratio_gt": note_in_chord_ratio_gt.mean(),
                "val/note_in_chord_ratio_pred": note_in_chord_ratio_pred.mean(),
            }
        )

        # We don't log audio and image into table because there will be bugs
        table_text = wandb.Table(columns=["Index", "Array GT", "Array Gen"])
        for i in range(min(self.max_log_examples, len(enc_inputs))):
            enc_input = enc_inputs[i].cpu().numpy()
            decoder_pred = decoder_preds[i].cpu().numpy()
            target = targets[i].cpu().numpy()
            # Convert array to text
            array_pred_text = str(decoder_pred)
            array_gt_text = str(target)
            table_text.add_data(i, array_gt_text, array_pred_text)

            midi_gt = self.output_to_midi(enc_input, target)
            self.log_midi(midi_gt, suffix=f"gt_{i}")

            try:
                midi_gen = self.output_to_midi(enc_input, decoder_pred)
                self.log_midi(midi_gen, suffix=f"gen_{i}")

            except Exception as e:
                logger.error(f"Error converting to MIDI: {e}")
                continue

        self.logger.experiment.log({"array_text": table_text})

        table_url = wandb.Table(columns=["Index", "Song URL"])
        for i, song_url in enumerate(song_urls):
            table_url.add_data(i, song_url)
        self.logger.experiment.log({"song_url": table_url})

    def output_to_midi(self, enc_input, decoder_pred):
        if self.model_part == "melody":
            midi = self.tokenizer.decode_to_midi(
                chord_frames=enc_input,
                melody_frames=decoder_pred,
            )
        elif self.model_part == "chord":
            midi = self.tokenizer.decode_to_midi(
                chord_frames=decoder_pred,
                melody_frames=enc_input,
            )
        return midi

    def configure_optimizers(self):
        """
        Configures and returns the optimizer(s).
        """
        optimizer = AdamW(filter(lambda p: p.requires_grad, self.parameters()))
        return optimizer
