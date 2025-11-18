"""Convert Hooktheory dataset to cache format.

This script processes the Hooktheory.json.gz dataset and converts it to
cache files that can be loaded by the HooktheoryDataset dataloader.
"""

import json
import gzip
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import argparse
from copy import deepcopy

from realchords.utils.io_utils import save_jsonl
from realchords.utils.data_utils import (
    to_chord_name,
    transpose_melody,
    transpose_chord,
    update_global_chord_names,
)


def load_all_splits(
    data_path: str, data_augmentation: bool = False
) -> Dict[str, List[Dict]]:
    """Load and process all dataset splits (train, valid, test).

    Args:
        data_path (str): Path to the Hooktheory.json.gz file
        data_augmentation (bool): Whether to apply data augmentation (transposition)

    Returns:
        dict: Dictionary containing processed data for each split.
    """
    with gzip.open(data_path, "r") as f:
        dataset = json.load(f)

    dataset_split_all = {}
    for split in ["train", "valid", "test"]:
        dataset_split = {
            k: v
            for k, v in dataset.items()
            if v["split"] == split.upper()
            and "MELODY" in v["tags"]
            and "HARMONY" in v["tags"]
            and "TEMPO_CHANGES" not in v["tags"]
        }

        dataset_split = list(dataset_split.values())
        dataset_split_all[split] = dataset_split

    # If data_augmentation is True, we will augment the training data.
    # We will transpose [-6, 6] semitones.
    if data_augmentation:
        train_data_augmented = []
        for data in tqdm(dataset_split_all["train"], desc="Augmenting training data"):
            for semitone in range(-6, 7):
                data_augmented = deepcopy(data)
                data_augmented["annotations"]["melody"] = []
                data_augmented["annotations"]["harmony"] = []
                for note in data["annotations"]["melody"]:
                    note_transposed = transpose_melody(note, semitone)
                    data_augmented["annotations"]["melody"].append(note_transposed)
                for chord in data["annotations"]["harmony"]:
                    chord_transposed = transpose_chord(chord, semitone)
                    data_augmented["annotations"]["harmony"].append(chord_transposed)
                train_data_augmented.append(data_augmented)
        dataset_split_all["train"] = train_data_augmented
        print(
            f"Augmented training data. New number of training samples: {len(dataset_split_all['train'])}"
        )

    return dataset_split_all


def get_chord_names(dataset_split_all: Dict[str, List[Dict]]) -> List[str]:
    """Extract and collect all unique chord names from the dataset.

    Args:
        dataset_split_all (dict): Dictionary with train/valid/test splits

    Returns:
        list: Sorted list of unique chord names found in the dataset.
    """
    all_chord_names = set()
    for split in ["train", "valid", "test"]:
        for data in tqdm(
            dataset_split_all[split],
            desc=f"Getting chord names for {split} data",
        ):
            for chord in data["annotations"]["harmony"]:
                chord_name = to_chord_name(
                    chord["root_pitch_class"],
                    chord["root_position_intervals"],
                    chord["inversion"],
                )
                all_chord_names.add(chord_name)

    chord_names = sorted(list(all_chord_names))
    return chord_names


def main():
    parser = argparse.ArgumentParser(
        description="Convert Hooktheory dataset to cache format"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/hooktheory/Hooktheory.json.gz",
        help="Path to Hooktheory.json.gz file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/cache/hooktheory",
        help="Output directory for cache files",
    )
    parser.add_argument(
        "--augmentation",
        action="store_true",
        help="Create augmented dataset with transposition",
    )

    args = parser.parse_args()

    # Check if input file exists
    hooktheory_path = Path(args.hooktheory_path)
    if not hooktheory_path.exists():
        print(f"Error: Hooktheory dataset not found at {hooktheory_path}")
        print(f"Please download it from:")
        print(f"  https://sheetsage.s3.amazonaws.com/hooktheory/Hooktheory.json.gz")
        print(f"And place it in data/hooktheory/")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Hooktheory dataset from {hooktheory_path}")

    # Load all splits
    dataset_split_all = load_all_splits(
        str(hooktheory_path), data_augmentation=args.augmentation
    )

    print(f"\nDataset splits:")
    print(f"  Train: {len(dataset_split_all['train'])} songs")
    print(f"  Valid: {len(dataset_split_all['valid'])} songs")
    print(f"  Test: {len(dataset_split_all['test'])} songs")

    # Collect chord names
    print(f"\nCollecting chord names...")
    chord_names = get_chord_names(dataset_split_all)
    print(f"Found {len(chord_names)} unique chord names")

    # Save regular (non-augmented) splits
    cache_postfix = "_augmented" if args.augmentation else ""
    for split_name, split_songs in dataset_split_all.items():
        cache_path = output_dir / f"{split_name}{cache_postfix}.jsonl"
        save_jsonl(split_songs, cache_path)
        print(f"Saved {split_name} split to {cache_path}")

    # Save chord names to dataset-specific location
    chord_names_path = output_dir / f"chord_names{cache_postfix}.json"
    with open(chord_names_path, "w") as f:
        json.dump(chord_names, f, indent=2)
    print(f"Saved chord names to {chord_names_path}")

    # Update global chord names
    cache_dir = str(output_dir.parent)  # Go up to data/cache/
    print(f"\nUpdating global chord_names in {cache_dir}...")
    update_global_chord_names(chord_names, cache_dir, augmented=args.augmentation)

    print(f"\n{'='*60}")
    print(f"Hooktheory dataset conversion completed!")
    print(f"{'='*60}")
    print(f"\nCache files saved to: {output_dir}")
    print(f"\nYou can now load the dataset with:")
    print(f'  dataset = HooktheoryDataset(cache_dir="{output_dir}", split="train")')


if __name__ == "__main__":
    main()
