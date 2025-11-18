"""Generate a checkpoint for RealJam from a RealChords pytorch lightning checkpoint.

E.g.:
checkpoint_path = "logs/decoder_only_online_chord/step=11000.ckpt"
save_dir = "checkpoints"
"""

import argparse
import os
import shutil
from pathlib import Path

import torch
from realchords.constants import REALJAM_CHECKPOINT_DIR


def main(checkpoint_path: str, save_dir: str):
    """Generate a checkpoint for ReaLJam from a ReaLChords pytorch lightning checkpoint.

    Args:
        checkpoint_path: Path to the source checkpoint file (e.g. "logs/decoder_only_online_chord/step=11000.ckpt")
        save_dir: Directory to save the converted checkpoint (e.g. "checkpoints")
    """
    # Load the ReaLChords checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Extract model name from checkpoint path (e.g. "decoder_only_online_chord" from the example)
    model_name = os.path.basename(os.path.dirname(checkpoint_path))

    # Create target directory (e.g. "checkpoints/decoder_only_online_chord")
    target_dir = Path(save_dir) / model_name
    target_dir.mkdir(parents=True, exist_ok=True)

    # Extract model state dict
    state_dict = checkpoint["state_dict"]
    # Remove "model." prefix from keys if present
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    # Remove "_orig_mod." prefix from keys if present
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    # Save model parameters (e.g. "checkpoints/decoder_only_online_chord/step=11000.ckpt")
    target_checkpoint_path = target_dir / os.path.basename(checkpoint_path)
    torch.save(state_dict, target_checkpoint_path)

    # Copy args.yml if it exists (e.g. from "logs/decoder_only_online_chord/args.yml")
    args_path = os.path.join(os.path.dirname(checkpoint_path), "args.yml")
    if os.path.exists(args_path):
        shutil.copy2(args_path, target_dir / "args.yml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a checkpoint for ReaLJam from a ReaLChords pytorch lightning checkpoint."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help='Path to the source checkpoint file (e.g. "logs/decoder_only_online_chord/step=11000.ckpt")',
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help='Directory to save the converted checkpoint (e.g. "checkpoints")',
    )

    args = parser.parse_args()
    main(args.checkpoint_path, args.save_dir)
