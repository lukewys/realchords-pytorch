"""Automatic model checkpoint downloader for RealJam."""

import os
import requests
from pathlib import Path
from tqdm import tqdm
import json
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# TODO: Update these URLs when model checkpoints are released
CHECKPOINT_URLS = {
    "decoder_only_online_chord_3_datasets/step=11000.ckpt": "https://github.com/lukewys/realchords/releases/download/v0.1.0/decoder_only_online_chord_3_datasets_step=11000.ckpt",
    "decoder_only_online_chord_3_datasets/args.yml": "https://github.com/lukewys/realchords/releases/download/v0.1.0/decoder_only_online_chord_3_datasets_args.yml",
    "enc_dec_base_chord_random_truncate_3_datasets/step=20000.ckpt": "https://github.com/lukewys/realchords/releases/download/v0.1.0/enc_dec_base_chord_random_truncate_3_datasets_step=20000.ckpt",
    "enc_dec_base_chord_random_truncate_3_datasets/args.yml": "https://github.com/lukewys/realchords/releases/download/v0.1.0/enc_dec_base_chord_random_truncate_3_datasets_args.yml",
    "realchords_entropy_gail_3_datasets_melody_pred/actor.pth": "https://github.com/lukewys/realchords/releases/download/v0.1.0/realchords_entropy_gail_3_datasets_melody_pred_actor.pth",
    "rl_realchords_chord_ensemble_rhythm_reward_offline_anchor_invalid_output_inner_loop_fix_invalid_penalty/actor.pth": "https://github.com/lukewys/realchords/releases/download/v0.1.0/rl_realchords_actor.pth",
    "chord_names_augmented.json": "https://github.com/lukewys/realchords/releases/download/v0.1.0/chord_names_augmented.json",
}


def download_file(url: str, save_path: Path, desc: str = "Downloading"):
    """Download a file with progress bar.

    Args:
        url: URL to download from
        save_path: Path to save the file
        desc: Description for progress bar
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    block_size = 8192

    with open(save_path, "wb") as f:
        with tqdm(
            total=total_size, unit="B", unit_scale=True, desc=desc
        ) as pbar:
            for chunk in response.iter_content(block_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def check_file_exists(file_path: Path) -> bool:
    """Check if a file exists and is not empty."""
    return file_path.exists() and file_path.stat().st_size > 0


def ensure_checkpoints(
    checkpoint_dir: Path = None, force_download: bool = False
):
    """Ensure all required model checkpoints are downloaded.

    Args:
        checkpoint_dir: Directory to store checkpoints. If None, uses ~/.realjam/checkpoints
        force_download: If True, re-download even if files exist
    """
    if checkpoint_dir is None:
        checkpoint_dir = Path.home() / ".realjam" / "checkpoints"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Checkpoint directory: {checkpoint_dir}")

    missing_files = []
    existing_files = []

    # Check which files need to be downloaded
    for relative_path, url in CHECKPOINT_URLS.items():
        file_path = checkpoint_dir / relative_path
        if force_download or not check_file_exists(file_path):
            missing_files.append((relative_path, url, file_path))
        else:
            existing_files.append(relative_path)

    if existing_files and not force_download:
        logging.info(f"✓ Found {len(existing_files)} existing checkpoint files")

    if not missing_files:
        logging.info("✓ All checkpoints are ready!")
        return

    logging.info(f"Downloading {len(missing_files)} checkpoint files...")

    for i, (relative_path, url, file_path) in enumerate(missing_files, 1):
        try:
            desc = f"[{i}/{len(missing_files)}] {relative_path}"
            logging.info(f"Downloading {relative_path}...")
            download_file(url, file_path, desc)
            logging.info(f"✓ Downloaded {relative_path}")
        except requests.exceptions.RequestException as e:
            logging.error(f"✗ Failed to download {relative_path}: {e}")
            logging.error(f"  URL: {url}")
            logging.warning(
                "\n⚠️  Checkpoint download failed. This might be because:\n"
                "   1. The model checkpoints haven't been released yet\n"
                "   2. You need to download them manually from the release page\n"
                "   3. Network connection issues\n"
                f"\nPlease check: {url}\n"
            )
            raise

    logging.info("✓ All checkpoints downloaded successfully!")
