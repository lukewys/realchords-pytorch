"""Dataloader for Hooktheory dataset."""

import os
import gzip
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
import note_seq
import note_seq.chord_symbols_lib
from copy import deepcopy

from realchords.constants import DATA_PATH, FRAME_PER_BEAT
from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer
from realchords.utils.sequence_utils import pad_and_get_mask
from realchords.utils.io_utils import save_jsonl, JSONLIndexer
from realchords.utils.logging_utils import logger


class HooktheoryDataset(Dataset):
    """Hooktheory dataset."""

    def __init__(
        self,
        data_path: str = DATA_PATH,
        frame_per_beat: int = FRAME_PER_BEAT,
        chord_names_path: str = None,
        model_part: str = "chord",
        split: str = "train",
        max_len: int = 512,
        model_type: str = "decoder_only",
        cache_dir: str = "",
        data_augmentation: bool = True,
        load_augmented_chord_names: bool = False,
        num_workers: int = 8,
        **kwargs,
    ):
        """Initialize the Hooktheory dataset.

        Args:
            data_path (str, optional): DEPRECATED. Path to the Hooktheory dataset. No longer used
                as the dataset is now loaded from cache. Defaults to DATA_PATH.
            frame_per_beat (int, optional): Number of frames per beat. Defaults to FRAME_PER_BEAT.
            chord_names_path (str, optional): Path to the chord names. Defaults to CHORD_NAMES_PATH.
            model_part (str, optional): Model part to use. Defaults to "chord".
            split (str, optional): Split to use. Defaults to "train".
            max_len (int, optional): Maximum length of the sequence. Defaults to 512.
            model_type (str, optional): Model type. Defaults to "decoder_only".
            cache_dir (str, optional): Directory to load the cache from. Defaults to "".
                For Hooktheory dataset, use "data/cache/hooktheory".
            data_augmentation (bool, optional): Whether to use data augmentation. Defaults to True.
            load_augmented_chord_names (bool, optional): Whether to load the augmented chord names
                (even if data_augmentation is False). Defaults to False.
            num_workers (int, optional): Number of workers for parallel data processing. Defaults to 4.

        Raises:
            ValueError: If the split is not in ["train", "valid", "test"].
            ValueError: If the model type is not in ["decoder_only", "encoder_decoder", "decoder_only_single"].
            ValueError: If the model part is not in ["chord", "melody"].
            FileNotFoundError: If cache files are not found at cache_dir.
        """
        assert split in ["train", "valid", "test"]
        # decoder_only: ReaLchords, online representation
        # encoder_decoder: ReaLchords, offline representation
        # decoder_only_single: only one part, unconditional generation
        allowed_model_type = [
            "decoder_only",
            "encoder_decoder",
            "decoder_only_single",
        ]

        self.data_path = data_path
        self.frame_per_beat = frame_per_beat
        self.chord_names_cache_path = (
            Path(chord_names_path) if chord_names_path else None
        )
        self.split = split
        self.max_len = max_len
        self.max_len_per_part = max_len // 2
        if model_part not in ["chord", "melody"]:
            raise ValueError("model_part must be either 'chord' or 'melody'.")
        self.model_part = model_part
        if model_type not in allowed_model_type:
            raise ValueError(
                f"model_type must be in {allowed_model_type}. Got {model_type}."
            )
        self.model_type = model_type
        self.data_augmentation = data_augmentation
        self.load_augmented_chord_names = load_augmented_chord_names
        self.num_workers = num_workers

        self.cache_dir = cache_dir
        self.load_data()
        self.filter_data()

        self.tokenizer = HooktheoryTokenizer(
            frame_per_beat=frame_per_beat, chord_names=self.chord_names
        )

    def cache_file_exists(self):
        """Check if the cache files exist.

        Returns:
            bool: True if both the dataset cache and chord names cache files exist, False otherwise.
        """
        self.cache_postfix = "_augmented" if self.data_augmentation else ""
        self.cache_path = (
            Path(self.cache_dir) / f"{self.split}{self.cache_postfix}.jsonl"
        )
        if self.load_augmented_chord_names or self.data_augmentation:
            self.chord_names_cache_postfix = "_augmented"
        else:
            self.chord_names_cache_postfix = ""
        if self.chord_names_cache_path is None:
            self.chord_names_cache_path = (
                Path(self.cache_dir)
                / f"chord_names{self.chord_names_cache_postfix}.json"
            )
        return self.cache_path.exists() and self.chord_names_cache_path.exists()

    def try_load_data_from_cache(self):
        """Attempt to load the dataset and chord names from cache files.

        Returns:
            bool: True if data was successfully loaded from cache, False otherwise.
        """
        has_cache = False
        if self.cache_file_exists():
            logger.info(f"Data cache dir: {Path(self.cache_dir).absolute()}")
            try:
                import time

                start_time = time.time()
                self.data = JSONLIndexer(self.cache_path)
                end_time = time.time()
                logger.info(
                    f"Time taken to load dataset: {end_time - start_time:.2f} seconds"
                )
                with open(self.chord_names_cache_path, "r") as f:
                    self.chord_names = json.load(f)
                logger.info(
                    f"Loaded {len(self.chord_names)} chord names from {self.chord_names_cache_path}"
                )
                has_cache = True
            except FileNotFoundError:
                logger.warning(
                    f"Dataset cache not found or corrupted at {self.cache_path} "
                    f" and {self.chord_names_cache_path}. Loading data..."
                )
        return has_cache

    def load_data(self):
        """Load the dataset from cache.

        Raises:
            FileNotFoundError: If cache files are not found at cache_dir.
        """
        has_cache = self.try_load_data_from_cache()
        if not has_cache:
            raise FileNotFoundError(
                f"Cache not found at {self.cache_dir}. "
                f"Please run the appropriate conversion script first:\n"
                f"  For Hooktheory: python scripts/convert_hooktheory_to_cache.py\n"
                f"  For POP909: python scripts/convert_pop909_to_cache.py\n"
                f"  For Nottingham: python scripts/convert_nottingham_to_cache.py\n"
                f"  For Wikifonia: python scripts/convert_wikifonia_to_cache.py"
            )
        logger.info(f"Loaded {len(self.data)} items")

    def filter_data(self):
        """Filter the dataset based on metadata criteria.

        Note:
            Currently a placeholder for future implementation.
        """
        # TODO: for later: filter data based on metadata
        pass

    def __len__(self):
        """Get the length of the dataset.

        Returns:
            int: Number of items in the dataset.
        """
        return len(self.data)

    @property
    def num_tokens(self):
        """Get the number of tokens in the vocabulary.

        Returns:
            int: Number of tokens in the tokenizer's vocabulary.
        """
        return self.tokenizer.num_tokens

    def random_crop(
        self, melody: torch.Tensor, chord: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly crop the sequences based on the max_len and frame_per_beat.

        Args:
            melody (torch.Tensor): Melody sequence.
            chord (torch.Tensor): Chord sequence.

        Returns:
            tuple: Cropped melody and chord sequences.
        """

        assert melody.shape[0] == chord.shape[0]

        if melody.shape[0] <= self.max_len_per_part:
            return melody, chord

        # Only allow starting from the multiple of frame_per_beat
        max_len = min(self.max_len_per_part, melody.shape[0])
        start_allowed = list(
            range(0, melody.shape[0] - max_len + 1, self.frame_per_beat)
        )
        start = np.random.choice(start_allowed)
        end = start + max_len
        return melody[start:end], chord[start:end]

    def add_bos_eos(self, sequence: torch.Tensor, eos: bool = True) -> torch.Tensor:
        """Add BOS and EOS tokens to the sequence.

        Args:
            sequence (torch.Tensor): Sequence to add tokens to.
            eos (bool, optional): Whether to add EOS token. Defaults to True.

        Returns:
            torch.Tensor: Sequence with BOS and EOS tokens added.
        """
        bos = torch.tensor([self.tokenizer.bos_token])
        sequence = torch.cat([bos, sequence])
        if eos:
            eos = torch.tensor([self.tokenizer.eos_token])
            sequence = torch.cat([sequence, eos])
        return sequence

    def interleave(self, seq1: torch.Tensor, seq2: torch.Tensor) -> torch.Tensor:
        """Interleave two sequences.

        Args:
            seq1 (torch.Tensor): First sequence.
            seq2 (torch.Tensor): Second sequence.

        Returns:
            torch.Tensor: Interleaved sequence.
        """
        interleaved = torch.zeros(seq1.shape[0] + seq2.shape[0], dtype=seq1.dtype)
        interleaved[0::2] = seq1
        interleaved[1::2] = seq2
        return interleaved

    def serialize(
        self, melody: torch.Tensor, chord: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Serialize the melody and chord sequences based on the model type.

        Args:
            melody (torch.Tensor): Melody sequence.
            chord (torch.Tensor): Chord sequence.

        Returns:
            dict: Dictionary containing serialized sequences based on model type.
        """
        if self.model_type == "encoder_decoder":
            if self.model_part == "melody":
                inputs = self.add_bos_eos(chord)
                targets = self.add_bos_eos(melody)
            elif self.model_part == "chord":
                inputs = self.add_bos_eos(melody)
                targets = self.add_bos_eos(chord)
            return {"inputs": inputs, "targets": targets}

        elif self.model_type == "decoder_only":
            if self.model_part == "melody":
                inputs = None
                targets = self.interleave(melody, chord)
                targets = self.add_bos_eos(targets)
            elif self.model_part == "chord":
                inputs = None
                targets = self.interleave(chord, melody)
                targets = self.add_bos_eos(targets)
            return {"targets": targets}

        elif self.model_type == "decoder_only_single":
            if self.model_part == "melody":
                inputs = None
                targets = melody
                targets = self.add_bos_eos(targets)
            elif self.model_part == "chord":
                inputs = None
                targets = chord
                targets = self.add_bos_eos(targets)
            return {"targets": targets}

    def process_item(self, item):
        output = self.tokenizer.encode(item)

        # Handle Hooktheory, Nottingham, and POP909 datasets
        if "hooktheory" in item:
            output["song_url"] = item["hooktheory"]["urls"]["song"]
        elif "nottingham" in item:
            # Create a compatible URL for Nottingham dataset
            output[
                "song_url"
            ] = f"nottingham://{item['nottingham']['file']}#{item['nottingham']['index']}"
        elif "pop909" in item:
            # Create a compatible URL for POP909 dataset
            output[
                "song_url"
            ] = f"pop909://{item['pop909']['file']}#{item['pop909']['id']}"
        elif "wikifonia" in item:
            # Create a compatible URL for Wikifonia dataset
            output[
                "song_url"
            ] = f"wikifonia://{item['wikifonia']['file']}#{item['wikifonia']['id']}"
        else:
            # Fallback for other datasets
            output["song_url"] = "unknown://unknown"

        # TODO: optionally add metadata
        return output

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        """Get an item from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            dict: Dictionary containing the processed item data including:
                - inputs (torch.Tensor, optional): Input sequence for encoder-decoder models
                - inputs_mask (torch.Tensor, optional): Mask for input sequence
                - targets (torch.Tensor): Target sequence
                - targets_mask (torch.Tensor): Mask for target sequence
                - song_url (str): URL of the song
        """
        item = self.data[idx]
        item = self.process_item(item)
        melody = torch.tensor(item["melody"])
        chord = torch.tensor(item["chord"])

        # Random crop
        melody, chord = self.random_crop(melody, chord)

        # Serialize
        output = self.serialize(melody, chord)

        # Pad and get mask
        if self.model_type == "encoder_decoder":
            inputs_pad, inputs_mask = pad_and_get_mask(
                output["inputs"],
                self.max_len_per_part + 2,  # +2 for BOS and EOS
            )
            targets_pad, targets_mask = pad_and_get_mask(
                output["targets"],
                self.max_len_per_part + 2,  # +2 for BOS and EOS
            )
            output["inputs"] = inputs_pad
            output["inputs_mask"] = inputs_mask
            output["targets"] = targets_pad
            output["targets_mask"] = targets_mask
        elif self.model_type == "decoder_only":
            targets_pad, targets_mask = pad_and_get_mask(
                output["targets"],
                self.max_len_per_part * 2 + 2,  # +2 for BOS and EOS
            )
            output["targets"] = targets_pad
            output["targets_mask"] = targets_mask
        elif self.model_type == "decoder_only_single":
            targets_pad, targets_mask = pad_and_get_mask(
                output["targets"],
                self.max_len_per_part + 2,  # +2 for BOS and EOS
            )
            output["targets"] = targets_pad
            output["targets_mask"] = targets_mask

        output["song_url"] = item["song_url"]
        return output


def get_dataloader(
    dataset: Dataset,
    shuffle: bool = True,
    batch_size: int = 8,
    num_workers: int = 4,
):
    """Get the dataloader for the Hooktheory dataset.

    Args:
        batch_size (int, optional): Batch size. Defaults to 8.
        num_workers (int, optional): Number of workers. Defaults to 4.

    Returns:
        torch.utils.data.DataLoader: Dataloader for the Hooktheory dataset.
    """

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader


if __name__ == "__main__":
    # Test dataloader
    dataset = HooktheoryDataset(
        split="train",
        model_type="decoder_only",
        max_len=512,
        cache_dir="data/cache/hooktheory",
    )
    dataloader = get_dataloader(dataset, batch_size=8)
    batch = next(iter(dataloader))
    print(f"Loaded dataset with {len(dataset)} items")
    print(f"Batch keys: {list(batch.keys())}")
    print(f"Batch shape: {batch['targets'].shape}")
