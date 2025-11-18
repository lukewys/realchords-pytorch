"""Download and setup Hooktheory dataset.

This script downloads the Hooktheory dataset and sets up the directory structure
for processing with the RealChords codebase.
"""

import argparse
from pathlib import Path
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


def download_hooktheory_dataset(output_dir: str = "data/hooktheory"):
    """Download the Hooktheory dataset.

    Args:
        output_dir (str): Directory to save the dataset
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Hooktheory dataset to {output_path}")

    # Dataset URL
    dataset_url = "https://sheetsage.s3.amazonaws.com/hooktheory/Hooktheory.json.gz"
    gz_path = output_path / "Hooktheory.json.gz"

    if gz_path.exists():
        print(f"Dataset already exists at {gz_path}")
        print("Skipping download. Delete the file to re-download.")
    else:
        print(f"Downloading from {dataset_url}")
        download_file_with_progress(dataset_url, gz_path)
        print(f"Hooktheory dataset successfully downloaded to {gz_path}")

    # Print dataset info
    print("\nDataset information:")
    print(f"  - File: {gz_path}")
    print(f"  - Size: {gz_path.stat().st_size / (1024*1024):.1f} MB")
    print(f"  - Format: Gzipped JSON")

    return output_path


def verify_dataset(dataset_path: Path):
    """Verify that the dataset was downloaded correctly.

    Args:
        dataset_path (Path): Path to the dataset directory
    """
    print("\nVerifying dataset...")

    gz_file = dataset_path / "Hooktheory.json.gz"

    if not gz_file.exists():
        print(f"Dataset file not found at {gz_file}")
        return False

    # Basic file size check (should be around 350-400 MB)
    file_size_mb = gz_file.stat().st_size / (1024 * 1024)
    print(f"Found dataset file: {gz_file}")
    print(f"File size: {file_size_mb:.1f} MB")

    if file_size_mb < 100:
        print(f"Warning: File size seems too small ({file_size_mb:.1f} MB)")
        print("The file may be corrupted or incomplete.")
        return False

    # Try to open and read the first few bytes
    try:
        import gzip

        with gzip.open(gz_file, "rb") as f:
            # Read first 100 bytes to verify it's a valid gzip file
            header = f.read(100)
            if header:
                print("Dataset file appears to be valid (gzip compressed)")
            else:
                print("Warning: Could not read file contents")
                return False
    except Exception as e:
        print(f"Error reading dataset file: {e}")
        return False

    print("Dataset verification completed")
    return True


def main():
    """Main function to handle command line arguments and run the download."""
    parser = argparse.ArgumentParser(description="Download Hooktheory dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/hooktheory",
        help="Output directory for the dataset",
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify dataset after download"
    )

    args = parser.parse_args()

    # Download dataset
    dataset_path = download_hooktheory_dataset(args.output_dir)

    # Verify if requested
    if args.verify:
        verify_dataset(dataset_path)

    print(f"\nHooktheory dataset setup completed!")
    print(f"Next steps:")
    print(f"  1. Run: python scripts/convert_hooktheory_to_cache.py --augmentation")
    print(f"  2. Check: data/cache/hooktheory/")


if __name__ == "__main__":
    main()
