"""Command-line interface for RealJam."""

import argparse
import sys
from pathlib import Path


def download_weights():
    """Download model checkpoints."""
    parser = argparse.ArgumentParser(
        description="Download RealJam model checkpoints"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to store checkpoints (default: ~/.realjam/checkpoints)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist",
    )
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir else None

    print("=" * 60)
    print("RealJam Model Checkpoint Downloader")
    print("=" * 60)

    try:
        from realjam.agent_interface import download_checkpoints

        if download_checkpoints(
            checkpoint_dir=checkpoint_dir, force=args.force
        ):
            print("\n✓ Success! All checkpoints are ready.")
            return 0
        else:
            print("\n✗ Download failed.")
            return 1
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1


def start_server():
    """Start the RealJam server with automatic checkpoint download."""
    print("Starting RealJam server...")
    print("Note: Checkpoints will be automatically downloaded if needed.")

    # Import here to avoid loading heavy dependencies if not needed
    from realjam.server import main as server_main

    try:
        server_main()
        return 0
    except KeyboardInterrupt:
        print("\n\nServer stopped by user")
        return 0
    except Exception as e:
        print(f"\n✗ Server error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(start_server())
