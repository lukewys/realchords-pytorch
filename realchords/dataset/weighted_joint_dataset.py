"""
Weighted Joint Dataset for combining multiple music datasets with custom sampling weights.

This extends the basic joint dataset functionality to support weighted sampling,
allowing you to control the proportion of samples from each dataset during training.

Usage:
    from scripts.weighted_joint_dataset import WeightedJointDataset

    # Create weighted dataset with custom sampling
    dataset = WeightedJointDataset(
        datasets=['pop909', 'nottingham'],
        weights=[0.7, 0.3],  # 70% POP909, 30% Nottingham
        data_augmentation=True
    )
"""

import sys
import os
from pathlib import Path
import time
import random
from typing import List, Tuple, Dict, Optional
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

# Add ReaLchords to path
sys.path.append(str(Path(__file__).parent.parent))
from realchords.dataset.hooktheory_dataloader import HooktheoryDataset
from realchords.constants import CACHE_DIR, DATA_PATH, FRAME_PER_BEAT


class RepeatableWeightedSampler(Sampler[int]):
    """Chunked weighted sampler that avoids allocating huge index tensors."""

    def __init__(
        self,
        weights: torch.Tensor,
        num_samples: int,
        seed: int = 0,
        chunk_size: int = 4096,
    ):
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = int(num_samples)
        self.seed = seed
        self.chunk_size = max(1, int(chunk_size))
        self.epoch = 0

    def set_epoch(self, epoch: int):
        # Set epoch for different sampling order each epoch
        # But we use a very long num_samples to avoid exhausting the dataset
        # so we don't need to set epoch
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        remaining = self.num_samples
        weights = self.weights.to(torch.double)

        while remaining > 0:
            curr = min(self.chunk_size, remaining)
            indices = torch.multinomial(
                weights,
                curr,
                replacement=True,
                generator=generator,
            )
            yield from indices.tolist()
            remaining -= curr

    def __len__(self):
        return self.num_samples


class WeightedJointDataset(Dataset):
    """Joint dataset with weighted sampling from multiple sources.

    Supports custom weighting to control the sampling proportion from each dataset.
    For example, you can ensure 70% of batches come from POP909 and 30% from Nottingham,
    regardless of their actual dataset sizes.
    """

    def __init__(
        self,
        datasets: List[str] = ["pop909", "nottingham"],
        chord_names_path: str = None,
        weights: Optional[List[float]] = None,
        split: str = "train",
        data_augmentation: bool = True,
        max_len: int = 512,
        model_type: str = "decoder_only",
        model_part: str = "chord",
        seed: int = 42,
        data_path: str = DATA_PATH,
        frame_per_beat: int = FRAME_PER_BEAT,
        load_augmented_chord_names: bool = False,
        num_workers: int = 8,
        train_samples_multiplier: float = 100.0,
        max_train_samples: Optional[int] = None,
        sampler_chunk_size: int = 4096,
    ):
        """Initialize weighted joint dataset.

        Args:
            datasets (List[str]): List of datasets to include
            chord_names_path (str): Path to chord names file
            weights (List[float], optional): Sampling weights for each dataset.
                If None, uses equal weights. Must sum to 1.0.
            split (str): Data split to use
            data_augmentation (bool): Enable data augmentation
            max_len (int): Maximum sequence length
            model_type (str): Model type
            model_part (str): Model part
            seed (int): Random seed for reproducible sampling
            data_path (str): Path to the dataset
            frame_per_beat (int): Number of frames per beat
            load_augmented_chord_names (bool): Load augmented chord names
            num_workers (int): Number of workers for data loading
        """
        self.datasets = datasets
        self.split = split
        self.data_augmentation = data_augmentation
        self.seed = seed
        self.model_part = model_part
        self.model_type = model_type
        self.train_samples_multiplier = train_samples_multiplier
        self.max_train_samples = max_train_samples
        self.sampler_chunk_size = sampler_chunk_size

        # Set default equal weights if not provided
        if weights is None:
            weights = [1.0 / len(datasets)] * len(datasets)

        if len(weights) != len(datasets):
            raise ValueError(
                f"Number of weights ({len(weights)}) must match number of datasets ({len(datasets)})"
            )

        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {sum(weights)}")

        self.weights = weights

        # Cache directory mapping
        self.cache_dirs = {
            "pop909": os.path.join(CACHE_DIR, "pop909"),
            "nottingham": os.path.join(CACHE_DIR, "nottingham"),
            "hooktheory": os.path.join(CACHE_DIR, "hooktheory"),
            "wikifonia": os.path.join(CACHE_DIR, "wikifonia"),
        }

        # Load individual datasets
        self.individual_datasets = []
        self.dataset_info = []
        self._load_datasets(
            max_len=max_len,
            model_type=model_type,
            model_part=model_part,
            chord_names_path=chord_names_path,
            data_path=data_path,
            frame_per_beat=frame_per_beat,
            load_augmented_chord_names=load_augmented_chord_names,
            num_workers=num_workers,
        )

        # Prepare sampling utilities
        # - Always compute per-item sample weights
        # - Use WeightedRandomSampler only for training
        # - For valid/test, precompute a deterministic, weight-respecting interleaved order
        self._setup_weighted_sampling()
        if self.split == "train":
            print("Using RepeatableWeightedSampler for training split.")
        else:
            self._setup_eval_indices()
            print(
                "Using deterministic weighted interleaving for eval (no sampler)."
            )

        print(f"\nWeighted joint dataset created:")
        print(f"   Total items: {len(self):,}")
        print(f"   Sampling weights: {dict(zip(datasets, weights))}")

    def _load_datasets(
        self,
        max_len: int,
        model_type: str,
        model_part: str,
        chord_names_path: str,
        data_path: str,
        frame_per_beat: int,
        load_augmented_chord_names: bool,
        num_workers: int,
    ):
        """Load individual datasets."""
        print(f"=== Loading Weighted Joint Dataset ===")
        print(f"Datasets: {', '.join(self.datasets)}")
        print(f"Weights: {self.weights}")
        print(f"Split: {self.split}")
        print(f"Augmentation: {self.data_augmentation}")
        print()

        for i, dataset_name in enumerate(self.datasets):
            if dataset_name not in self.cache_dirs:
                raise ValueError(f"Unknown dataset: {dataset_name}")

            default_cache_dir = self.cache_dirs[dataset_name]
            print(f"Loading {dataset_name} (weight: {self.weights[i]:.1%})...")
            start_time = time.time()

            try:
                dataset = HooktheoryDataset(
                    cache_dir=default_cache_dir,
                    split=self.split,
                    data_augmentation=self.data_augmentation,
                    max_len=max_len,
                    model_type=model_type,
                    model_part=model_part,
                    chord_names_path=chord_names_path,
                    data_path=data_path,
                    frame_per_beat=frame_per_beat,
                    load_augmented_chord_names=load_augmented_chord_names,
                    num_workers=num_workers,
                )

                self.individual_datasets.append(dataset)
                load_time = time.time() - start_time

                self.dataset_info.append(
                    {
                        "name": dataset_name,
                        "size": len(dataset),
                        "weight": self.weights[i],
                        "load_time": load_time,
                        "start_idx": sum(
                            len(d) for d in self.individual_datasets[:-1]
                        ),
                        "end_idx": sum(
                            len(d) for d in self.individual_datasets
                        ),
                    }
                )

                print(
                    f"  {dataset_name}: {len(dataset):,} items ({load_time:.1f}s)"
                )

            except Exception as e:
                print(f"  {dataset_name}: Failed to load - {e}")
                raise

    def _setup_weighted_sampling(self):
        """Setup per-item weights and (for train) a weighted random sampler."""
        # Create sample weights for each item
        sample_weights = []

        for info in self.dataset_info:
            dataset_size = info["size"]
            target_weight = info["weight"]

            # Weight per sample = target_weight / dataset_size
            # This ensures samples from smaller datasets get higher individual weights
            weight_per_sample = target_weight / max(dataset_size, 1)
            sample_weights.extend([weight_per_sample] * dataset_size)

        self.sample_weights = torch.tensor(sample_weights, dtype=torch.float)

        # Create weighted random sampler for training only
        self.sampler = None
        if self.split == "train":
            if self.max_train_samples is not None:
                num_samples_per_epoch = self.max_train_samples
            else:
                num_samples_per_epoch = int(
                    len(self) * self.train_samples_multiplier
                )
            num_samples_per_epoch = max(num_samples_per_epoch, 1)
            self.sampler = RepeatableWeightedSampler(
                weights=self.sample_weights,
                num_samples=num_samples_per_epoch,
                seed=self.seed,
                chunk_size=self.sampler_chunk_size,
            )

    def _setup_eval_indices(self):
        """Create a deterministic, weighted-interleaved global index order for eval.

        Strategy:
        - Shuffle indices within each dataset deterministically (seeded)
        - Interleave datasets using Smooth Weighted Round Robin (SWRR)
          to approximate the provided target weights across the sequence

        Note: this does not guarantee iterate over all items from each dataset.
        """
        # Build per-dataset index lists and shuffle them deterministically
        per_ds_indices = []  # list[list[int]]
        active = []  # list[bool]
        weights = []  # list[float]

        for i, info in enumerate(self.dataset_info):
            start, end = info["start_idx"], info["end_idx"]
            idxs = list(range(start, end))
            rnd = random.Random(self.seed + i)
            rnd.shuffle(idxs)
            per_ds_indices.append(idxs)
            active.append(len(idxs) > 0)
            weights.append(float(info["weight"]))

        total_items = len(self)
        eval_indices = []

        # Smooth Weighted Round Robin accumulators
        acc = [0.0 for _ in weights]

        # Avoid division by zero if all weights are zero (should not happen)
        if sum(weights) <= 0:
            weights = [1.0 / max(len(weights), 1)] * len(weights)

        # Generate the full order
        while len(eval_indices) < total_items:
            # Update accumulators
            for i, w in enumerate(weights):
                if active[i]:
                    acc[i] += w

            # Choose dataset with max accumulator among active ones
            best_i = None
            best_score = -1.0
            for i, a in enumerate(acc):
                if active[i] and a > best_score:
                    best_score = a
                    best_i = i

            if best_i is None:
                # No active datasets left; break to avoid infinite loop
                break

            # Emit one index from the chosen dataset
            chosen_list = per_ds_indices[best_i]
            if chosen_list:
                eval_indices.append(chosen_list.pop())
                acc[best_i] -= 1.0
                if not chosen_list:
                    active[best_i] = False
            else:
                active[best_i] = False

        # If anything remains (due to edge cases), append in any deterministic order
        if len(eval_indices) < total_items:
            for i, lst in enumerate(per_ds_indices):
                eval_indices.extend(lst)

        # Final safety trim
        self.eval_indices = eval_indices[:total_items]

    def __len__(self):
        """Return total number of items across all datasets."""
        return sum(len(dataset) for dataset in self.individual_datasets)

    def __getitem__(self, idx: int):
        """Get item by index from the appropriate dataset."""
        # For eval splits, map through the precomputed weighted-interleaved order
        if (
            getattr(self, "eval_indices", None) is not None
            and self.split != "train"
        ):
            idx = self.eval_indices[idx]
        # Find which dataset this index belongs to
        for info in self.dataset_info:
            if info["start_idx"] <= idx < info["end_idx"]:
                local_idx = idx - info["start_idx"]
                dataset_idx = self.datasets.index(info["name"])
                return self.individual_datasets[dataset_idx][local_idx]

        raise IndexError(f"Index {idx} out of range")

    @property
    def tokenizer(self):
        return self.individual_datasets[0].tokenizer

    def get_weighted_dataloader(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        shuffle: bool = None,
        **kwargs,
    ):
        """Create DataLoader with weighted sampling for train and mixed order for eval.

        The returned DataLoader supports multiple epochs without exhaustion:
        - For training: Uses WeightedRandomSampler that can iterate indefinitely
        - For eval: Uses deterministic weighted interleaving that can be repeated

        Args:
            batch_size (int): Batch size
            num_workers (int): Number of workers
            shuffle (bool, optional): For eval, controls DataLoader shuffling. Defaults to False.
                For train, ignored when sampler is used.
            **kwargs: Additional DataLoader arguments

        Returns:
            torch.utils.data.DataLoader: DataLoader that supports multiple epochs

        Note:
            The training DataLoader uses WeightedRandomSampler with num_samples=10*dataset_size,
            allowing it to be iterated multiple times across epochs without exhaustion.
            Each epoch will see different samples due to the large sample pool.
        """

        if self.split == "train":
            if shuffle is not None:
                print(
                    "Note: 'shuffle' is ignored for train since a sampler is used."
                )
            return DataLoader(
                self,
                batch_size=batch_size,
                sampler=self.sampler,
                num_workers=num_workers,
                **kwargs,
            )
        else:
            # Default to deterministic order unless explicitly overridden
            effective_shuffle = False if shuffle is None else shuffle
            return DataLoader(
                self,
                batch_size=batch_size,
                shuffle=effective_shuffle,
                num_workers=num_workers,
                **kwargs,
            )

    # Override the get_dataloader method to use the new get_weighted_dataloader method
    def get_dataloader(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        shuffle: bool = None,
        **kwargs,
    ):
        return self.get_weighted_dataloader(
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            **kwargs,
        )

    def get_sampling_statistics(
        self, num_samples: int = 10000
    ) -> Dict[str, float]:
        """Get actual sampling statistics by drawing samples.

        Args:
            num_samples (int): Number of samples to draw for statistics

        Returns:
            Dict[str, float]: Actual sampling proportions per dataset
        """
        dataset_counts = {name: 0 for name in self.datasets}

        # Sample indices using weighted random sampling
        # Create a new sampler for statistics that can generate the requested samples
        stats_sampler = RepeatableWeightedSampler(
            weights=self.sample_weights,
            num_samples=num_samples,
            replacement=True,
            generator=torch.Generator().manual_seed(self.seed),
        )

        sample_indices = list(stats_sampler)

        for idx in sample_indices:
            for info in self.dataset_info:
                if info["start_idx"] <= idx < info["end_idx"]:
                    dataset_counts[info["name"]] += 1
                    break

        # Convert to proportions
        return {
            name: count / num_samples for name, count in dataset_counts.items()
        }

    def print_statistics(self):
        """Print detailed dataset statistics."""
        print("\\n=== Weighted Dataset Statistics ===")

        total_items = len(self)

        for info in self.dataset_info:
            actual_proportion = info["size"] / total_items
            target_weight = info["weight"]

            print(f"{info['name']}:")
            print(
                f"  Items: {info['size']:,} ({actual_proportion:.1%} of total)"
            )
            print(f"  Target weight: {target_weight:.1%}")
            print(f"  Sampling boost: {target_weight/actual_proportion:.2f}x")

        print(f"\\nTotal items: {total_items:,}")

        # Test actual sampling
        print("\\nActual sampling test (10k samples):")
        actual_stats = self.get_sampling_statistics()
        for name, proportion in actual_stats.items():
            target = dict(zip(self.datasets, self.weights))[name]
            print(f"  {name}: {proportion:.1%} (target: {target:.1%})")

    def test_multiple_iterations(
        self, num_epochs: int = 3, batches_per_epoch: int = 10
    ):
        """Test that the dataloader can be iterated multiple times without issues.

        Args:
            num_epochs (int): Number of epochs to test
            batches_per_epoch (int): Number of batches to sample per epoch

        Returns:
            bool: True if test passes, False otherwise
        """
        print(
            f"\\n=== Testing Multiple Iterations ({num_epochs} epochs, {batches_per_epoch} batches each) ==="
        )

        try:
            dataloader = self.get_weighted_dataloader(
                batch_size=8, num_workers=0
            )

            for epoch in range(num_epochs):
                print(f"Epoch {epoch + 1}/{num_epochs}:")
                batch_count = 0

                for i, batch in enumerate(dataloader):
                    if i >= batches_per_epoch:
                        break
                    batch_count += 1

                    # Verify batch structure
                    if not isinstance(batch, dict) or "targets" not in batch:
                        print(f"  Invalid batch structure at batch {i}")
                        return False

                print(f"  Successfully processed {batch_count} batches")

            print("Multiple iteration test passed!")
            return True

        except Exception as e:
            print(f"Multiple iteration test failed: {e}")
            return False


def create_weighted_joint_dataset(
    datasets: List[str] = ["hooktheory"],
    weights: Optional[List[float]] = None,
    chord_names_path: str = None,
    split: str = "train",
    data_augmentation: bool = True,
    max_len: int = 512,
    model_type: str = "decoder_only",
    model_part: str = "chord",
    seed: int = 42,
    data_path: str = DATA_PATH,
    frame_per_beat: int = FRAME_PER_BEAT,
    load_augmented_chord_names: bool = False,
    num_workers: int = 8,
    train_samples_multiplier: float = 100.0,
    max_train_samples: Optional[int] = None,
    sampler_chunk_size: int = 4096,
) -> WeightedJointDataset:
    """Convenience function to create weighted joint dataset.

    Args:
        datasets (List[str]): List of datasets to include
        weights (List[float], optional): Sampling weights for each dataset
        chord_names_path (str): Path to chord names file
        split (str): Data split to use ('train', 'valid', 'test')
        data_augmentation (bool): Enable data augmentation
        max_len (int): Maximum sequence length
        model_type (str): Model type ('decoder_only', etc.)
        model_part (str): Model part ('chord', etc.)
        seed (int): Random seed for reproducible sampling
        data_path (str): Path to the dataset
        frame_per_beat (int): Number of frames per beat
        load_augmented_chord_names (bool): Load augmented chord names
        num_workers (int): Number of workers for data loading

    Returns:
        WeightedJointDataset: Configured weighted dataset

    Example:
        >>> # Equal weights (default)
        >>> dataset = create_weighted_joint_dataset(['pop909', 'nottingham'])

        >>> # Custom weights: 80% POP909, 20% Nottingham
        >>> dataset = create_weighted_joint_dataset(
        ...     ['pop909', 'nottingham'],
        ...     weights=[0.8, 0.2]
        ... )
    """
    return WeightedJointDataset(
        datasets=datasets,
        weights=weights,
        chord_names_path=chord_names_path,
        split=split,
        data_augmentation=data_augmentation,
        max_len=max_len,
        model_type=model_type,
        model_part=model_part,
        seed=seed,
        data_path=data_path,
        frame_per_beat=frame_per_beat,
        load_augmented_chord_names=load_augmented_chord_names,
        num_workers=num_workers,
        train_samples_multiplier=train_samples_multiplier,
        max_train_samples=max_train_samples,
        sampler_chunk_size=sampler_chunk_size,
    )


def get_dataloader(
    dataset: WeightedJointDataset,
    shuffle: bool = True,
    batch_size: int = 8,
    num_workers: int = 4,
):
    return dataset.get_dataloader(batch_size, num_workers, shuffle=shuffle)


if __name__ == "__main__":
    # Example usage
    print("=== Weighted Joint Dataset Examples ===\\n")

    # Example 0: Single Dataset
    print("Example 0: Single Dataset")
    dataset_single = create_weighted_joint_dataset(
        datasets=["hooktheory"],
        weights=None,  # Equal weights
        chord_names_path=os.path.join(CACHE_DIR, "chord_names_augmented.json"),
        data_augmentation=True,
    )
    dataset_single.print_statistics()

    dataset_single = create_weighted_joint_dataset(
        datasets=["pop909"],
        weights=None,  # Equal weights
        chord_names_path=os.path.join(CACHE_DIR, "chord_names_augmented.json"),
        data_augmentation=True,
    )
    dataset_single.print_statistics()

    print("\\n" + "=" * 60 + "\\n")

    # Example 1: Equal weights
    print("Example 1: Equal weights")
    dataset_equal = create_weighted_joint_dataset(
        datasets=["hooktheory", "pop909", "nottingham"],
        weights=None,  # Equal weights
        chord_names_path=os.path.join(CACHE_DIR, "chord_names_augmented.json"),
        data_augmentation=True,
    )
    dataset_equal.print_statistics()

    print("\\n" + "=" * 60 + "\\n")

    # Example 2: Custom weights
    print("Example 2: Custom weights (80% POP909, 20% Nottingham)")
    dataset_weighted = create_weighted_joint_dataset(
        datasets=["hooktheory", "pop909", "nottingham"],
        weights=[0.6, 0.3, 0.1],
        chord_names_path=os.path.join(CACHE_DIR, "chord_names_augmented.json"),
        data_augmentation=True,
    )
    dataset_weighted.print_statistics()

    # Example 3: Create DataLoader
    print("\\n" + "=" * 60 + "\\n")
    print("Example 3: DataLoader with weighted sampling")
    dataloader = dataset_weighted.get_weighted_dataloader(batch_size=8)

    batch = next(iter(dataloader))
    print(f"Batch shape: {batch['targets'].shape}")
    print(f"Batch keys: {list(batch.keys())}")

    print(f"Sample 100 batches from dataset:")
    num_samples = 100
    from tqdm import tqdm

    for i in tqdm(range(num_samples)):
        batch = next(iter(dataloader))

    # Test multiple iterations
    dataset_weighted.test_multiple_iterations(num_epochs=2, batches_per_epoch=5)

    # Example 4: Custom weights validation
    print("Example 4: Custom weights validation (80% POP909, 20% Nottingham)")
    dataset_weighted = create_weighted_joint_dataset(
        datasets=["hooktheory", "pop909", "nottingham"],
        weights=[0.6, 0.3, 0.1],
        chord_names_path=os.path.join(CACHE_DIR, "chord_names_augmented.json"),
        data_augmentation=True,
        split="valid",
    )
    dataset_weighted.print_statistics()

    # Example 3: Create DataLoader
    print("\\n" + "=" * 60 + "\\n")
    print("Example 4: DataLoader with weighted sampling")
    dataloader = dataset_weighted.get_weighted_dataloader(
        batch_size=8, shuffle=False
    )

    batch = next(iter(dataloader))
    print(f"Batch shape: {batch['targets'].shape}")
    print(f"Batch keys: {list(batch.keys())}")

    print(f"Sample 100 batches from dataset:")
    num_samples = 100
    from tqdm import tqdm

    for i in tqdm(range(num_samples)):
        batch = next(iter(dataloader))
