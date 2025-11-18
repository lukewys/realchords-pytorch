"""Download and setup Wikifonia dataset.

This script downloads the Wikifonia dataset from GitHub and sets up the directory structure
for processing with the RealChords codebase.
"""

import os
import argparse
import zipfile
from pathlib import Path
import urllib.request
from tqdm import tqdm


def download_file_with_progress(url: str, filepath: Path):
    """Download a file with progress bar."""

    def progress_hook(block_num, block_size, total_size):
        progress_bar.update(block_size)

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


def download_wikifonia_dataset(
    output_dir: str = "data/wikifonia",
    verify: bool = False,
):
    """Download the Wikifonia dataset from GitHub."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Wikifonia dataset to {output_path}")

    repo_url = "https://github.com/00sapo/OpenEWLD"
    zip_url = f"{repo_url}/archive/master.zip"
    zip_path = output_path / "OpenEWLD-master.zip"

    if zip_path.exists():
        print(f"Archive already exists at {zip_path}")
    else:
        print(f"Downloading from {zip_url}")
        download_file_with_progress(zip_url, zip_path)

    extracted_path = output_path / "OpenEWLD-master"
    if extracted_path.exists():
        print(f"Dataset already extracted at {extracted_path}")
    else:
        print(f"Extracting {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(output_path)

    # The dataset is in dataset/ subdirectory
    final_path = output_path / "wikifonia"
    if not final_path.exists():
        dataset_dir = extracted_path / "dataset"
        if dataset_dir.exists():
            print(f"Moving dataset to {final_path}")
            dataset_dir.rename(final_path)
        else:
            print("✗ Dataset directory not found in repository.")
            return None

    if zip_path.exists():
        print(f"Cleaning up {zip_path}")
        zip_path.unlink()

    print(f"Wikifonia dataset successfully downloaded to {final_path}")

    # Count MXL files recursively since they're in subdirectories
    mxl_files = list(final_path.glob("**/*.mxl"))
    print(f"  - Found {len(mxl_files)} MusicXML files (.mxl format)")
    print(f"  - Dataset path: {final_path}")

    return final_path


def main():
    parser = argparse.ArgumentParser(description="Download Wikifonia dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/wikifonia",
        help="Output directory for the dataset",
    )

    args = parser.parse_args()
    dataset_path = download_wikifonia_dataset(args.output_dir)

    if dataset_path:
        print(f"\nWikifonia dataset setup completed!")
        print(f"Next steps:")
        print(f"  1. Run: python scripts/convert_wikifonia_to_cache.py")


if __name__ == "__main__":
    main()
