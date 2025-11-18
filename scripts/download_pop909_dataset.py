"""Download and setup POP909 dataset.

This script downloads the POP909 dataset from GitHub and sets up the directory structure
for processing with the ReaLchords codebase.
"""

import os
import subprocess
import argparse
from pathlib import Path
import zipfile
import urllib.request
from tqdm import tqdm


def download_file_with_progress(url: str, filepath: Path):
    """Download a file with progress bar.

    Args:
        url (str): URL to download from
        filepath (Path): Path to save the file
    """

    def progress_hook(block_num, block_size, total_size):
        progress_bar.update(block_size)

    # Create progress bar
    response = urllib.request.urlopen(url)
    total_size = int(response.headers.get("Content-Length", 0))
    progress_bar = tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        desc=f"Downloading {filepath.name}",
    )

    try:
        urllib.request.urlretrieve(url, filepath, progress_hook)
    finally:
        progress_bar.close()


def download_pop909_dataset(output_dir: str = "data/pop909"):
    """Download the POP909 dataset from GitHub.

    Args:
        output_dir (str): Directory to save the dataset
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading POP909 dataset to {output_path}")

    # GitHub repository URL
    repo_url = "https://github.com/music-x-lab/POP909-Dataset"
    zip_url = f"{repo_url}/archive/master.zip"

    # Download the zip file
    zip_path = output_path / "POP909-Dataset-master.zip"

    if zip_path.exists():
        print(f"Archive already exists at {zip_path}")
    else:
        print(f"Downloading from {zip_url}")
        download_file_with_progress(zip_url, zip_path)

    # Extract the zip file
    extracted_path = output_path / "POP909-Dataset-master"

    if extracted_path.exists():
        print(f"Dataset already extracted at {extracted_path}")
    else:
        print(f"Extracting {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(output_path)

    # Move contents to a cleaner directory structure
    final_path = output_path / "POP909"
    if not final_path.exists():
        print(f"Moving dataset to {final_path}")
        extracted_path.rename(final_path)

    # Clean up zip file
    if zip_path.exists():
        print(f"Cleaning up {zip_path}")
        zip_path.unlink()

    print(f"POP909 dataset successfully downloaded to {final_path}")

    # Print dataset structure info
    print("\nDataset structure:")
    midi_dir = final_path / "POP909"
    if midi_dir.exists():
        midi_files = list(midi_dir.glob("*/*.mid"))
        chord_files = list(midi_dir.glob("*/*.txt"))
        print(f"  - Found {len(midi_files)} MIDI files")
        print(f"  - Found {len(chord_files)} chord annotation files")
        print(f"  - Dataset path: {midi_dir}")
    else:
        print(f"  - Dataset files should be in: {midi_dir}")

    return final_path


def verify_dataset(dataset_path: Path):
    """Verify that the dataset was downloaded correctly.

    Args:
        dataset_path (Path): Path to the dataset directory
    """
    print("\nVerifying dataset...")

    pop909_dir = dataset_path / "POP909"
    if not pop909_dir.exists():
        print(f"✗ POP909 directory not found at {pop909_dir}")
        return False

    # Check for MIDI and chord files
    midi_files = list(pop909_dir.glob("*/*.mid"))
    chord_files = list(pop909_dir.glob("*/*.txt"))

    expected_songs = 909  # Should be around 909 songs

    print(f"Found {len(midi_files)} MIDI files")
    print(f"Found {len(chord_files)} chord files")

    if len(midi_files) < 800:  # Allow some tolerance
        print(
            f"Warning: Expected around {expected_songs} MIDI files, found {len(midi_files)}"
        )

    if len(chord_files) < 800:
        print(
            f"Warning: Expected around {expected_songs} chord files, found {len(chord_files)}"
        )

    # Check a few files exist
    sample_dirs = list(pop909_dir.glob("*"))[:5]
    for sample_dir in sample_dirs:
        if sample_dir.is_dir():
            midi_file = sample_dir / f"{sample_dir.name}.mid"
            chord_file = sample_dir / "chord_midi.txt"

            if midi_file.exists() and chord_file.exists():
                print(
                    f"Sample verification: {sample_dir.name} has both MIDI and chord files"
                )
            else:
                print(f"Sample verification: {sample_dir.name} missing files")
                print(f"  - MIDI: {midi_file.exists()}")
                print(f"  - Chord: {chord_file.exists()}")

    print("Dataset verification completed")
    return True


def main():
    """Main function to handle command line arguments and run the download."""
    parser = argparse.ArgumentParser(description="Download POP909 dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/pop909",
        help="Output directory for the dataset",
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify dataset after download"
    )

    args = parser.parse_args()

    # Download dataset
    dataset_path = download_pop909_dataset(args.output_dir)

    # Verify if requested
    if args.verify:
        verify_dataset(dataset_path)

    print(f"\nPOP909 dataset setup completed!")
    print(f"Next steps:")
    print(f"  1. Run: python scripts/convert_pop909_to_cache.py")
    print(f"  2. Run: python scripts/test_pop909_dataset.py")


if __name__ == "__main__":
    main()
