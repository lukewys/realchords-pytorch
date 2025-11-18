"""Utilities for data processing - minimal version for inference."""

from typing import List, Optional
from note_seq import chord_symbols_lib


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
