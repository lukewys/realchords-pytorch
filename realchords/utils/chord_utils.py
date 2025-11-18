from note_seq import chord_symbols_lib
from typing import List, Optional


def postprocess_chord_name(chord_name: str) -> str:
    """Postprocess a chord name to handle the bug of note_seq.chord_symbols_lib.transpose_chord_symbol.

    Args:
        chord_name (str): The chord name to postprocess.

    Returns:
        str: The postprocessed chord name.
    """
    if "Cb" in chord_name:
        chord_name = chord_name.replace("Cb", "B")
    elif "B#" in chord_name:
        chord_name = chord_name.replace("B#", "C")
    elif "Fb" in chord_name:
        chord_name = chord_name.replace("Fb", "E")
    elif "E#" in chord_name:
        chord_name = chord_name.replace("E#", "F")
    return chord_name
