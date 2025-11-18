"""Constants for the ReaLchords."""

from pathlib import Path

# Dataset paths - use current working directory
# This allows the package to work in both development (pip install -e .)
# and installed (pip install .) modes
DATA_DIR = Path.cwd() / "data" / "hooktheory"
DATA_PATH = DATA_DIR / "Hooktheory.json.gz"
CACHE_DIR = Path.cwd() / "data" / "cache"
CHORD_NAMES_AUG_PATH = CACHE_DIR / "chord_names_augmented.json"

# Checkpoint directory
REALJAM_CHECKPOINT_DIR = Path.cwd() / "checkpoints"

# Sound font from https://sites.google.com/site/soundfonts4u/
SF2_PATH = Path.cwd() / "soundfonts" / "Yamaha C5 Grand-v2.4.sf2"

# Convert to strings for backward compatibility where needed
DATA_DIR = str(DATA_DIR)
DATA_PATH = str(DATA_PATH)
CACHE_DIR = str(CACHE_DIR)
CHORD_NAMES_AUG_PATH = str(CHORD_NAMES_AUG_PATH)
REALJAM_CHECKPOINT_DIR = str(REALJAM_CHECKPOINT_DIR)
SF2_PATH = str(SF2_PATH)

# Constants of the hooktheory dataset
ZERO_OCTAVE = 60

# Tokenization parameters
PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
FRAME_PER_BEAT = 4

# MIDI synthesis parameters
MELODY_VELOCITY = 62 + 14
CHORD_VELOCITY = 50
BASS_VELOCITY = 56
CHORD_OCTAVE = 4
BASS_OCTAVE = CHORD_OCTAVE - 1
MIDI_SYNTH_SR = 44100
