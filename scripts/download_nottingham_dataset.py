"""Download and setup Nottingham dataset.

This script downloads the Nottingham folk music dataset from GitHub and sets up
the directory structure for processing with the RealChords codebase.
"""

import argparse
from pathlib import Path
import urllib.request
import zipfile
from tqdm import tqdm
import shutil


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


def download_nottingham_dataset(
    output_dir: str = "data/nottingham",
    verify: bool = False,
):
    """Download the Nottingham dataset from GitHub.

    Args:
        output_dir (str): Directory to save the dataset
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Nottingham dataset to {output_path}")

    # GitHub repository URL
    repo_url = "https://github.com/jukedeck/nottingham-dataset"
    zip_url = f"{repo_url}/archive/master.zip"

    # Download the zip file
    zip_path = output_path / "nottingham-dataset-master.zip"

    if zip_path.exists():
        print(f"Archive already exists at {zip_path}")
    else:
        print(f"Downloading from {zip_url}")
        download_file_with_progress(zip_url, zip_path)

    # Extract the zip file
    extracted_path = output_path / "nottingham-dataset-master"
    abc_cleaned_path = output_path / "nottingham_abc"

    if abc_cleaned_path.exists():
        print(f"Dataset already extracted at {abc_cleaned_path}")
    else:
        print(f"Extracting {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(output_path)

        # Move ABC_cleaned to a cleaner directory structure
        source_abc = extracted_path / "ABC_cleaned"
        if source_abc.exists():
            print(f"Moving ABC files to {abc_cleaned_path}")
            source_abc.rename(abc_cleaned_path)
        else:
            print(f"Warning: ABC_cleaned not found in {extracted_path}")

        # Clean up extracted master directory
        if extracted_path.exists():
            print(f"Cleaning up {extracted_path}")
            shutil.rmtree(extracted_path)

    # Clean up zip file
    if zip_path.exists():
        print(f"Cleaning up {zip_path}")
        zip_path.unlink()

    print(f"Nottingham dataset successfully downloaded to {abc_cleaned_path}")

    # Print dataset structure info
    print("\nDataset structure:")
    if abc_cleaned_path.exists():
        abc_files = list(abc_cleaned_path.glob("*.abc"))
        print(f"  - Found {len(abc_files)} ABC files")
        print(f"  - Dataset path: {abc_cleaned_path}")

        # Show some example files
        if abc_files:
            print(
                f"  - Example files: {', '.join([f.name for f in abc_files[:3]])}"
            )
    else:
        print(f"  - Dataset files should be in: {abc_cleaned_path}")

    return abc_cleaned_path


def verify_dataset(dataset_path: Path):
    """Verify that the dataset was downloaded correctly.

    Args:
        dataset_path (Path): Path to the dataset directory
    """
    print("\nVerifying dataset...")

    if not dataset_path.exists():
        print(f"Dataset directory not found at {dataset_path}")
        return False

    # Check for ABC files
    abc_files = list(dataset_path.glob("*.abc"))

    expected_files = 14  # Nottingham dataset has 14 ABC files
    print(f"Found {len(abc_files)} ABC files")

    if len(abc_files) < expected_files - 2:  # Allow some tolerance
        print(
            f"Warning: Expected around {expected_files} ABC files, found {len(abc_files)}"
        )

    # Check a few files for content
    sample_files = abc_files[:3] if len(abc_files) >= 3 else abc_files
    for abc_file in sample_files:
        try:
            with open(abc_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read(200)  # Read first 200 chars
                if "X:" in content or "T:" in content:
                    print(
                        f"Sample verification: {abc_file.name} appears valid (ABC format)"
                    )
                else:
                    print(
                        f"Warning: {abc_file.name} may not be valid ABC format"
                    )
        except Exception as e:
            print(f"Error reading {abc_file.name}: {e}")

    print("Dataset verification completed")
    return True


def main():
    """Main function to handle command line arguments and run the download."""
    parser = argparse.ArgumentParser(description="Download Nottingham dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/nottingham",
        help="Output directory for the dataset",
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify dataset after download"
    )

    args = parser.parse_args()

    # Download dataset
    dataset_path = download_nottingham_dataset(args.output_dir)

    # Verify if requested
    if args.verify:
        verify_dataset(dataset_path)

    print(f"\nNottingham dataset setup completed!")
    print(f"Next steps:")
    print(
        f"  1. Run: python scripts/convert_nottingham_to_cache.py --augmentation"
    )
    print(f"  2. Check: data/cache/nottingham/")


if __name__ == "__main__":
    main()
