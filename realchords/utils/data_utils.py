"""Utilities for data processing and chord manipulation.

This module contains functions for:
- Transposing melody notes and chords
- Managing global chord names across datasets
- Chord name postprocessing and transposition
"""

import json
import os
from pathlib import Path
from typing import List, Optional
from copy import deepcopy

from note_seq import chord_symbols_lib


# ============================================================================
# Chord Name Conversion
# ============================================================================


def to_chord_name(
    root_pitch_class: int,
    root_position_intervals: List[int],
    inversion: Optional[int] = None,
) -> str:
    """Convert root pitch class and intervals to chord name.

    Args:
        root_pitch_class (int): The root pitch class (0-11).
        root_position_intervals (List[int]): The intervals from the root.
        inversion (Optional[int]): The inversion number.

    Returns:
        str: The chord name.

    Example:
        >>> to_chord_name(0, [4, 7])  # C major triad
        'C'
        >>> to_chord_name(0, [3, 7])  # C minor triad
        'Cm'
    """
    pitches = [root_pitch_class]
    curr = root_pitch_class
    for pitch in root_position_intervals:
        curr += pitch
        pitches.append(curr)
    pitch_name = chord_symbols_lib.pitches_to_chord_symbol(pitches)
    if inversion is not None:
        for i in range(inversion):
            pitches = pitches[1:] + [pitches[0] + 12]
        while pitches[0] > 11:
            pitches = [pitch - 12 for pitch in pitches]
    return pitch_name


# ============================================================================
# Melody and Chord Transposition
# ============================================================================


def transpose_melody(note: dict, semitone: int) -> dict:
    """Transpose a melody note by a given number of semitones.

    Args:
        note (dict): Dictionary containing note information with keys:
            - pitch_class (int): The pitch class of the note (0-11)
            - octave (int): The octave number
        semitone (int): Number of semitones to transpose by (can be negative)

    Returns:
        dict: A new dictionary containing the transposed note information with the same structure
            as the input note dictionary.

    Example:
        >>> note = {"pitch_class": 0, "octave": 4}  # Middle C
        >>> transposed = transpose_melody(note, 2)  # Transpose up 2 semitones
        >>> print(transposed)
        {"pitch_class": 2, "octave": 4}  # D4
    """
    if semitone == 0:
        return note
    note_transposed = deepcopy(note)
    pitch_class = note["pitch_class"]
    octave = note["octave"]
    pitch_class_transposed = pitch_class + semitone
    octave_transposed = octave + pitch_class_transposed // 12
    pitch_class_transposed = pitch_class_transposed % 12
    note_transposed["pitch_class"] = pitch_class_transposed
    note_transposed["octave"] = octave_transposed
    return note_transposed


def transpose_chord(chord: dict, semitone: int) -> dict:
    """Transpose a chord by a given number of semitones.

    Args:
        chord (dict): Dictionary containing chord information with key:
            - root_pitch_class (int): The root pitch class of the chord (0-11)
        semitone (int): Number of semitones to transpose by (can be negative)

    Returns:
        dict: A new dictionary containing the transposed chord information with the same structure
            as the input chord dictionary.

    Example:
        >>> chord = {"root_pitch_class": 0}  # C chord
        >>> transposed = transpose_chord(chord, 7)  # Transpose up a perfect fifth
        >>> print(transposed)
        {"root_pitch_class": 7}  # G chord
    """
    if semitone == 0:
        return chord
    chord_transposed = deepcopy(chord)
    root_pitch_class = chord["root_pitch_class"]
    root_pitch_class_transposed = (root_pitch_class + semitone) % 12
    chord_transposed["root_pitch_class"] = root_pitch_class_transposed
    return chord_transposed


# ============================================================================
# Global Chord Names Management
# ============================================================================


def update_global_chord_names(
    new_chord_names: List[str],
    cache_dir: str,
    augmented: bool = False,
) -> None:
    """Update global chord names file by merging with new chord names.

    This function reads the existing global chord_names.json (or chord_names_augmented.json),
    takes the union with new_chord_names, sorts the result, and saves it back.

    Args:
        new_chord_names (List[str]): List of chord names to add
        cache_dir (str): Path to the cache directory (should be data/cache)
        augmented (bool, optional): Whether to update the augmented chord names file.
            Defaults to False.

    Example:
        >>> update_global_chord_names(
        ...     ["C", "Dm", "G7"],
        ...     "data/cache",
        ...     augmented=False
        ... )
        Updated global chord_names.json: added 3 new chords (total: 150)
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Determine filename
    filename = "chord_names_augmented.json" if augmented else "chord_names.json"
    global_path = cache_path / filename

    # Read existing chord names if file exists
    existing_chord_names = set()
    if global_path.exists():
        try:
            with open(global_path, "r") as f:
                existing_chord_names = set(json.load(f))
        except (json.JSONDecodeError, TypeError):
            print(
                f"Warning: Could not read existing {filename}, starting fresh"
            )
            existing_chord_names = set()

    # Merge with new chord names
    original_count = len(existing_chord_names)
    merged_chord_names = existing_chord_names.union(set(new_chord_names))
    new_count = len(merged_chord_names) - original_count

    # Sort and save
    sorted_chord_names = sorted(list(merged_chord_names))
    with open(global_path, "w") as f:
        json.dump(sorted_chord_names, f, indent=2)

    # Print update information
    print(f"\n{'='*60}")
    print(f"Updated global {filename}:")
    print(f"  - Added {new_count} new chord(s)")
    print(f"  - Total unique chords: {len(sorted_chord_names)}")
    print(f"  - File: {global_path}")
    print(f"{'='*60}\n")
