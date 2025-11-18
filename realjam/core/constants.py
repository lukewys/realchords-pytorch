"""Constants for RealJam inference."""

from pathlib import Path

# Use user home directory for storing checkpoints
REALJAM_HOME = Path.home() / ".realjam"
REALJAM_CHECKPOINT_DIR = REALJAM_HOME / "checkpoints"
CACHE_DIR = REALJAM_HOME / "cache"

# Convert to strings for backward compatibility
REALJAM_CHECKPOINT_DIR = str(REALJAM_CHECKPOINT_DIR)
CACHE_DIR = str(CACHE_DIR)

# Constants from hooktheory dataset
ZERO_OCTAVE = 60

# Tokenization parameters
PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
FRAME_PER_BEAT = 4

# MIDI synthesis parameters
CHORD_OCTAVE = 4
BASS_OCTAVE = CHORD_OCTAVE - 1
