"""Convert POP909 dataset to Hooktheory-compatible cache format.

This script processes MIDI files and chord annotations from the POP909 dataset and converts
them to the same cache format used by the Hooktheory dataset, allowing them to be
loaded by the existing HooktheoryDataset dataloader.
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import argparse
import pretty_midi
import numpy as np
from copy import deepcopy
import note_seq.chord_symbols_lib as chord_symbols_lib
import random

from realchords.constants import ZERO_OCTAVE
from realchords.utils.io_utils import save_jsonl
from realchords.utils.data_utils import (
    to_chord_name,
    transpose_melody,
    transpose_chord,
    update_global_chord_names,
)


def parse_chord_symbol(chord_symbol: str) -> Tuple[int, List[int], int]:
    """Convert POP909 chord symbol to root pitch class, intervals, and inversion using note_seq.

    POP909 uses format like "C:maj", "F#:min", "Bb:7", "N" (no chord)

    Args:
        chord_symbol (str): Chord symbol from POP909 (e.g., "C:maj", "F#:min")

    Returns:
        Tuple[int, List[int], int]: (root_pitch_class, intervals, inversion)
    """

    # Handle no chord case
    if chord_symbol.strip() in ["N", "X", ""]:
        return 0, [], 0  # Will be filtered out later

    # Convert POP909 format to standard chord notation
    standard_chord = pop909_to_standard_chord(chord_symbol)
    if standard_chord is None:
        # Fallback to C major for unparseable chords
        return 0, [4, 3], 0

    try:
        # Get pitches from note_seq
        chord_pitches = chord_symbols_lib.chord_symbol_pitches(standard_chord)

        if not chord_pitches:
            # Fallback to C major
            return 0, [4, 3], 0

        # For slash chords, note-seq returns bass note first, so adjust
        if "/" in standard_chord:
            # Remove bass note from pitches and use base chord root
            base_chord_name = standard_chord.split("/")[0]
            base_chord_pitches = chord_symbols_lib.chord_symbol_pitches(base_chord_name)
            chord_pitches = base_chord_pitches

        # Extract root pitch class (first pitch)
        root_pitch_class = chord_pitches[0] % 12

        # Calculate intervals between consecutive pitches
        intervals = []
        for i in range(1, len(chord_pitches)):
            interval = (chord_pitches[i] - chord_pitches[i - 1]) % 12
            intervals.append(interval)

        # Calculate inversion based on bass note position
        inversion = 0
        if "/" in standard_chord:
            # For slash chords, find which chord tone is in bass
            base_chord_name = standard_chord.split("/")[0]
            bass_note_name = standard_chord.split("/")[1]

            try:
                # Get pitches for base chord (without bass)
                base_chord_pitches = chord_symbols_lib.chord_symbol_pitches(
                    base_chord_name
                )
                bass_pitch = (
                    chord_symbols_lib.chord_symbol_pitches(bass_note_name)[0] % 12
                )

                # Find bass note position in chord tones
                for i, pitch in enumerate(base_chord_pitches):
                    if pitch % 12 == bass_pitch:
                        inversion = i
                        break
            except:
                inversion = 0  # fallback to root position

        return root_pitch_class, intervals, inversion

    except Exception as e:
        print(f"Error parsing chord {chord_symbol} -> {standard_chord}: {e}")
        # Fallback to C major
        return 0, [4, 3], 0


def scale_degree_to_note(root_note: str, scale_degree: str) -> str:
    """Convert scale degree to actual note name based on root.

    Args:
        root_note (str): Root note (e.g., 'F#', 'Eb')
        scale_degree (str): Scale degree (e.g., '3', 'b3', '5', '7', 'b7')

    Returns:
        str: Actual note name (e.g., 'A#', 'G')
    """
    # Convert root to pitch class
    note_map = {
        "C": 0,
        "C#": 1,
        "Db": 1,
        "D": 2,
        "D#": 3,
        "Eb": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "Gb": 6,
        "G": 7,
        "G#": 8,
        "Ab": 8,
        "A": 9,
        "A#": 10,
        "Bb": 10,
        "B": 11,
    }

    # Reverse mapping from pitch class to note name (prefer sharps for consistency)
    pitch_to_note = {
        0: "C",
        1: "C#",
        2: "D",
        3: "D#",
        4: "E",
        5: "F",
        6: "F#",
        7: "G",
        8: "G#",
        9: "A",
        10: "A#",
        11: "B",
    }

    root_pitch = note_map.get(root_note, 0)

    # Scale degree intervals (semitones from root)
    degree_intervals = {
        "3": 4,  # major 3rd
        "b3": 3,  # minor 3rd
        "5": 7,  # perfect 5th
        "7": 11,  # major 7th
        "b7": 10,  # minor 7th
    }

    if scale_degree not in degree_intervals:
        return root_note  # fallback to root

    target_pitch = (root_pitch + degree_intervals[scale_degree]) % 12
    return pitch_to_note[target_pitch]


def pop909_to_standard_chord(chord_symbol: str) -> Optional[str]:
    """Convert POP909 chord symbol to standard notation for note_seq.

    Args:
        chord_symbol (str): POP909 chord symbol (e.g., "C:maj", "F#:min7")

    Returns:
        Optional[str]: Standard chord notation or None if invalid
    """
    # Handle no chord case
    if chord_symbol.strip() in ["N", "X", ""]:
        return None

    # Handle root-only chords (no colon)
    if ":" not in chord_symbol:
        return chord_symbol.strip()

    # Split by colon
    root_str, quality = chord_symbol.split(":", 1)
    root_str = root_str.strip()
    quality = quality.strip()

    # Handle inversions - extract bass note information if present
    bass_note = None
    if "/" in quality:
        quality, bass_part = quality.split("/", 1)
        quality = quality.strip()
        bass_part = bass_part.strip()
        # Convert scale degree to actual note name
        bass_note = scale_degree_to_note(root_str, bass_part)

    # Convert POP909 quality to standard notation
    base_chord = None
    if quality == "maj" or quality == "":
        base_chord = root_str
    elif quality == "min":
        base_chord = root_str + "m"
    elif quality == "7":
        base_chord = root_str + "7"
    elif quality == "min7":
        base_chord = root_str + "m7"
    elif quality == "maj7":
        base_chord = root_str + "maj7"
    elif quality == "dim":
        base_chord = root_str + "dim"
    elif quality == "dim7":
        base_chord = root_str + "dim7"
    elif quality == "aug":
        base_chord = root_str + "aug"
    elif quality == "sus4":
        base_chord = root_str + "sus4"
    elif quality == "sus2":
        base_chord = root_str + "sus2"
    elif quality == "maj6":
        base_chord = root_str + "6"  # Standard notation for major 6th
    elif quality == "min6":
        base_chord = root_str + "m6"
    elif quality == "hdim7":
        base_chord = root_str + "m7b5"  # Half-diminished 7th
    elif quality == "minmaj7":
        base_chord = root_str + "mM7"  # Minor-major 7th
    elif quality == "sus4(b7)":
        base_chord = root_str + "7sus"  # Sus4 with b7
    else:
        # Return None for unknown qualities
        return None

    # Add bass note if present (inversion)
    if bass_note and bass_note != root_str:
        return f"{base_chord}/{bass_note}"
    else:
        return base_chord


def load_chord_annotations(chord_file: Path) -> List[Dict]:
    """Load chord annotations from POP909 chord file.

    Args:
        chord_file (Path): Path to chord_midi.txt file

    Returns:
        List[Dict]: List of chord dictionaries with timing and chord info
    """
    chords = []

    with open(chord_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) >= 3:
                start_time = float(parts[0])
                end_time = float(parts[1])
                chord_symbol = parts[2]

                # Skip "N" (no chord) entries
                if chord_symbol in ["N", "X"]:
                    continue

                # Parse chord symbol
                root_pitch_class, intervals, inversion = parse_chord_symbol(
                    chord_symbol
                )

                # Skip if no intervals (invalid chord)
                if not intervals:
                    continue

                # Skip zero-duration chords (chord change markers)
                if start_time >= end_time:
                    continue

                chords.append(
                    {
                        "onset": start_time,
                        "offset": end_time,
                        "root_pitch_class": root_pitch_class,
                        "root_position_intervals": intervals,
                        "inversion": inversion,
                    }
                )

    return chords


def load_key_info(key_file: Path) -> List[str]:
    """Load key information from POP909 key file.

    Args:
        key_file (Path): Path to key_audio.txt file

    Returns:
        List[str]: List of keys in chronological order
    """
    keys = []

    if not key_file.exists():
        return keys

    with open(key_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 3:
                    # Format: start_time end_time key
                    key = parts[2]
                    keys.append(key)

    return keys


def estimate_bpm_and_meter(
    beat_times: List[float],
    downbeat_flags: List[bool],
    bar_start_flags: List[bool],
) -> Tuple[float, str]:
    """Estimate BPM and time signature from beat information.

    Args:
        beat_times (List[float]): Beat timestamps in seconds
        downbeat_flags (List[bool]): Downbeat flags (not used for meter)
        bar_start_flags (List[bool]): Bar start flags (used for meter estimation)

    Returns:
        Tuple[float, str]: (BPM, time_signature)
    """

    # Calculate BPM from beat intervals
    if len(beat_times) > 1:
        intervals = np.diff(beat_times)
        # Remove outliers (very short or long intervals)
        valid_intervals = intervals[(intervals > 0.2) & (intervals < 2.0)]
        if len(valid_intervals) > 0:
            avg_beat_interval = np.median(valid_intervals)
            bpm = 60.0 / avg_beat_interval
        else:
            bpm = 120.0  # fallback
    else:
        bpm = 120.0  # fallback

    # Estimate time signature from bar start patterns
    if sum(bar_start_flags) > 1:
        bar_start_indices = [
            i for i, is_bar_start in enumerate(bar_start_flags) if is_bar_start
        ]
        if len(bar_start_indices) > 1:
            bar_intervals = np.diff(bar_start_indices)
            # Filter reasonable meter values (2-8 beats per measure)
            valid_meters = bar_intervals[(bar_intervals >= 2) & (bar_intervals <= 8)]
            if len(valid_meters) > 0:
                meter_beats = int(np.median(valid_meters))
                time_signature = f"{meter_beats}/4"
            else:
                time_signature = "4/4"  # fallback
        else:
            time_signature = "4/4"  # fallback
    else:
        time_signature = "4/4"  # fallback

    return float(bpm), time_signature


def load_beat_info(
    beat_file: Path,
) -> Tuple[List[float], List[bool], List[bool]]:
    """Load beat information from POP909 beat file with all columns.

    Args:
        beat_file (Path): Path to beat_midi.txt file

    Returns:
        Tuple containing:
        - List[float]: Beat times in seconds (quarter-note grid)
        - List[bool]: Downbeat flags (True if downbeat)
        - List[bool]: Bar start flags (True if first beat of bar)
    """
    beat_times = []
    downbeat_flags = []
    bar_start_flags = []

    with open(beat_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                # Beat file format: "timestamp downbeat_flag bar_start_flag"
                # Column 1: timestamp (seconds)
                # Column 2: downbeat flag (1 = downbeat, 0 = not)
                # Column 3: bar start flag (1 = first beat of bar, 0 = not)
                parts = line.split()
                if len(parts) >= 3:
                    timestamp = float(parts[0])
                    is_downbeat = bool(float(parts[1]) > 0.5)
                    is_bar_start = bool(float(parts[2]) > 0.5)

                    beat_times.append(timestamp)
                    downbeat_flags.append(is_downbeat)
                    bar_start_flags.append(is_bar_start)

    return beat_times, downbeat_flags, bar_start_flags


def find_trim_offset(
    melody_notes: List[Dict],
    beat_times: List[float],
    bar_start_flags: List[bool],
    threshold_beats: float = 4.0,
) -> float:
    """Find the time offset to trim beginning silence if >threshold_beats.

    Args:
        melody_notes (List[Dict]): List of melody notes with onset/offset
        beat_times (List[float]): Beat times in seconds
        bar_start_flags (List[bool]): Flags indicating bar starts
        threshold_beats (float): Threshold in beats (quarter notes) to trigger trimming

    Returns:
        float: Time offset to subtract from all timings (0.0 if no trimming needed)
    """
    if not melody_notes:
        return 0.0

    # Find the first melody note onset
    first_melody_onset = min(note["onset"] for note in melody_notes)

    # Find the onset in terms of beats
    if not beat_times:
        return 0.0

    # Find closest beat index for first melody onset
    closest_beat_idx = 0
    min_diff = float("inf")
    for i, beat_time in enumerate(beat_times):
        diff = abs(first_melody_onset - beat_time)
        if diff < min_diff:
            min_diff = diff
            closest_beat_idx = i

    # If the first melody note is more than threshold_beats from start
    if closest_beat_idx >= threshold_beats:
        # Find the nearest bar start at or before the first melody note
        bar_start_idx = None
        for i in range(closest_beat_idx, -1, -1):
            if i < len(bar_start_flags) and bar_start_flags[i]:
                bar_start_idx = i
                break

        if bar_start_idx is not None:
            return beat_times[bar_start_idx]

    return 0.0


def create_beat_sequences(
    beat_times: List[float],
    downbeat_flags: List[bool],
    bar_start_flags: List[bool],
    total_duration: float,
    trim_offset: float = 0.0,
    resolution: float = 0.25,
) -> Tuple[List[int], List[int], List[int]]:
    """Create frame-wise binary sequences for beat information.

    Args:
        beat_times (List[float]): Beat times in seconds
        downbeat_flags (List[bool]): Downbeat flags
        bar_start_flags (List[bool]): Bar start flags
        total_duration (float): Total duration in quarter-note units after trimming
        trim_offset (float): Time offset to subtract from beat times (in seconds)
        resolution (float): Frame resolution in quarter-note units (0.25 = 16th note)

    Returns:
        Tuple of three binary sequences (beat, downbeat, bar_start)
    """
    # Calculate total frames needed
    total_frames = int(np.ceil(total_duration / resolution))

    # Initialize sequences
    beat_sequence = [0] * total_frames
    downbeat_sequence = [0] * total_frames
    bar_start_sequence = [0] * total_frames

    # Estimate quarter note duration from beat intervals
    if len(beat_times) > 1:
        avg_beat_interval = np.mean(
            [beat_times[i + 1] - beat_times[i] for i in range(len(beat_times) - 1)]
        )
    else:
        avg_beat_interval = 0.5  # fallback: 120 BPM

    # Convert each beat to frame index and set flags
    for i, (beat_time, is_downbeat, is_bar_start) in enumerate(
        zip(beat_times, downbeat_flags, bar_start_flags)
    ):
        # Apply trim offset
        adjusted_time = beat_time - trim_offset

        # Skip negative times (before trim point)
        if adjusted_time < 0:
            continue

        # Convert seconds to quarter-note units
        beat_position = (
            adjusted_time / avg_beat_interval if avg_beat_interval > 0 else 0
        )

        # Convert to frame index (16th note resolution)
        frame_idx = int(round(beat_position / resolution))

        # Set flags if within bounds
        if 0 <= frame_idx < total_frames:
            beat_sequence[frame_idx] = 1
            if is_downbeat:
                downbeat_sequence[frame_idx] = 1
            if is_bar_start:
                bar_start_sequence[frame_idx] = 1

    return beat_sequence, downbeat_sequence, bar_start_sequence


def apply_time_offset(items: List[Dict], offset: float) -> List[Dict]:
    """Apply time offset to list of notes or chords.

    Args:
        items (List[Dict]): List of notes or chords with onset/offset
        offset (float): Time offset to subtract

    Returns:
        List[Dict]: Items with adjusted timing
    """
    if offset == 0.0:
        return items

    adjusted_items = []
    for item in items:
        adjusted_item = item.copy()
        adjusted_item["onset"] = max(0.0, item["onset"] - offset)
        adjusted_item["offset"] = max(adjusted_item["onset"], item["offset"] - offset)

        # Only keep items that have positive duration after trimming
        # Use small tolerance to handle floating point precision issues
        min_duration = 0.001  # 1ms minimum duration
        if adjusted_item["offset"] > adjusted_item["onset"] + min_duration:
            adjusted_items.append(adjusted_item)

    return adjusted_items


def extract_melody_from_midi(midi_file: Path) -> List[Dict]:
    """Extract melody notes from MIDI file - only from MELODY track.

    Args:
        midi_file (Path): Path to MIDI file

    Returns:
        List[Dict]: List of melody note dictionaries
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(str(midi_file))
    except Exception as e:
        print(f"Error loading MIDI file {midi_file}: {e}")
        return []

    melody_notes = []

    # Find the MELODY track specifically
    melody_instrument = None
    for instrument in midi_data.instruments:
        if instrument.name == "MELODY":
            melody_instrument = instrument
            break

    if melody_instrument is None:
        print(f"No MELODY track found in {midi_file}")
        return []

    # Extract notes only from the MELODY track
    for note in melody_instrument.notes:
        # Convert MIDI pitch to octave and pitch class
        relative_pitch = note.pitch - ZERO_OCTAVE
        octave = relative_pitch // 12
        pitch_class = relative_pitch % 12

        melody_notes.append(
            {
                "onset": note.start,
                "offset": note.end,
                "pitch_class": pitch_class,
                "octave": octave,
            }
        )

    return melody_notes


def quantize_to_beats(
    notes_or_chords: List[Dict],
    beat_times: List[float],
    beats_per_measure: int = 4,
    subdivisions_per_beat: int = 4,  # 16th note resolution (4 subdivisions per quarter note)
) -> List[Dict]:
    """Quantize note/chord timings to 16th note resolution within quarter-note grid.

    Uses Hooktheory time notation where 1.0 = quarter note (beat), 0.25 = 16th note.

    Args:
        notes_or_chords (List[Dict]): List of notes or chords with onset/offset in seconds
        beat_times (List[float]): Quarter-note grid times in seconds
        beats_per_measure (int): Number of beats per measure
        subdivisions_per_beat (int): Number of subdivisions per beat (4 = 16th notes)

    Returns:
        List[Dict]: Quantized notes/chords with 16th note resolution in Hooktheory time units
                   (1.0 = quarter note, 0.25 = 16th note, minimal grid = 0.25)
    """
    if not beat_times:
        return notes_or_chords

    quantized = []

    if len(beat_times) < 2:
        return notes_or_chords

    for item in notes_or_chords:
        # Find which quarter-note interval this note's onset falls in
        onset_beat_idx = 0
        for i in range(len(beat_times) - 1):
            if beat_times[i] <= item["onset"] < beat_times[i + 1]:
                onset_beat_idx = i
                break
        else:
            # Handle edge case: note after last beat
            onset_beat_idx = len(beat_times) - 1

        # Calculate subdivision within the quarter-note for onset
        if onset_beat_idx < len(beat_times) - 1:
            beat_start = beat_times[onset_beat_idx]
            beat_end = beat_times[onset_beat_idx + 1]
            beat_duration = beat_end - beat_start

            # Position within this quarter-note (0.0 to 1.0)
            relative_position = (item["onset"] - beat_start) / beat_duration
            # Which 16th note subdivision (0, 1, 2, or 3)
            subdivision = round(relative_position * subdivisions_per_beat)
            subdivision = max(
                0, min(subdivisions_per_beat, subdivision)
            )  # Clamp to valid range
        else:
            subdivision = 0

        # Calculate onset in quarter-note units with 16th note precision
        onset_quarters = onset_beat_idx + (subdivision / subdivisions_per_beat)

        # Find which quarter-note interval this note's offset falls in
        offset_beat_idx = onset_beat_idx
        for i in range(len(beat_times) - 1):
            if beat_times[i] <= item["offset"] < beat_times[i + 1]:
                offset_beat_idx = i
                break
        else:
            # Handle edge case: note after last beat
            offset_beat_idx = len(beat_times) - 1

        # Calculate subdivision within the quarter-note for offset
        if offset_beat_idx < len(beat_times) - 1:
            beat_start = beat_times[offset_beat_idx]
            beat_end = beat_times[offset_beat_idx + 1]
            beat_duration = beat_end - beat_start

            # Position within this quarter-note (0.0 to 1.0)
            relative_position = (item["offset"] - beat_start) / beat_duration
            # Which 16th note subdivision (0, 1, 2, or 3)
            subdivision = round(relative_position * subdivisions_per_beat)
            subdivision = max(
                0, min(subdivisions_per_beat, subdivision)
            )  # Clamp to valid range
        else:
            subdivision = 0

        # Calculate offset in quarter-note units with 16th note precision
        offset_quarters = offset_beat_idx + (subdivision / subdivisions_per_beat)

        # Ensure minimum duration of 1 sixteenth note (0.25 quarter notes)
        if offset_quarters <= onset_quarters:
            offset_quarters = onset_quarters + (1 / subdivisions_per_beat)

        # Create quantized item
        quantized_item = deepcopy(item)
        quantized_item["onset"] = onset_quarters
        quantized_item["offset"] = offset_quarters

        quantized.append(quantized_item)

    return quantized


def resolve_melody_overlaps(melody_notes: List[Dict]) -> List[Dict]:
    """Resolve melody note overlaps by truncating earlier notes.

    Args:
        melody_notes: List of melody notes with onset/offset

    Returns:
        List of melody notes with overlaps resolved
    """
    if not melody_notes:
        return melody_notes

    # Sort by onset time, then by offset time for stable sorting
    sorted_notes = sorted(melody_notes, key=lambda x: (x["onset"], x["offset"]))
    resolved_notes = []

    i = 0
    while i < len(sorted_notes):
        current_note = deepcopy(sorted_notes[i])

        # Check if there are notes with the same onset
        same_onset_notes = [current_note]
        j = i + 1
        while (
            j < len(sorted_notes) and sorted_notes[j]["onset"] == current_note["onset"]
        ):
            same_onset_notes.append(deepcopy(sorted_notes[j]))
            j += 1

        if len(same_onset_notes) == 1:
            # No same-onset conflicts, just check for regular overlaps with later notes
            for k in range(j, len(sorted_notes)):
                later_note = sorted_notes[k]
                if later_note["onset"] < current_note["offset"]:
                    current_note["offset"] = later_note["onset"]
                    break

            # Ensure minimum duration
            if current_note["offset"] <= current_note["onset"]:
                current_note["offset"] = current_note["onset"] + 0.001

            resolved_notes.append(current_note)
        else:
            # Multiple notes with same onset - keep only the longest one
            longest_note = max(same_onset_notes, key=lambda x: x["offset"] - x["onset"])

            # Check for overlaps with later notes
            for k in range(j, len(sorted_notes)):
                later_note = sorted_notes[k]
                if later_note["onset"] < longest_note["offset"]:
                    longest_note["offset"] = later_note["onset"]
                    break

            # Ensure minimum duration
            if longest_note["offset"] <= longest_note["onset"]:
                longest_note["offset"] = longest_note["onset"] + 0.001

            resolved_notes.append(longest_note)

        i = j

    return resolved_notes


def process_pop909_song(song_dir: Path) -> Optional[Dict]:
    """Process a single POP909 song directory.

    Args:
        song_dir (Path): Path to song directory

    Returns:
        Optional[Dict]: Processed song dictionary or None if processing failed
    """
    song_id = song_dir.name

    # Find required files
    midi_file = song_dir / f"{song_id}.mid"
    chord_file = song_dir / "chord_midi.txt"
    beat_file = song_dir / "beat_midi.txt"
    key_file = song_dir / "key_audio.txt"

    # Check if all required files exist
    if not all(f.exists() for f in [midi_file, chord_file, beat_file]):
        print(f"Missing files for song {song_id}")
        return None

    try:
        # Load annotations with full beat information
        chords = load_chord_annotations(chord_file)
        beat_times, downbeat_flags, bar_start_flags = load_beat_info(beat_file)

        # Load key information and estimate BPM/meter
        keys = load_key_info(key_file)
        bpm, time_signature = estimate_bpm_and_meter(
            beat_times, downbeat_flags, bar_start_flags
        )

        melody_notes = extract_melody_from_midi(midi_file)

        # Skip if no valid data
        if not chords or not melody_notes:
            print(f"No valid chords or melody for song {song_id}")
            return None

        # Find trim offset if beginning silence >4 beats
        trim_offset = find_trim_offset(
            melody_notes, beat_times, bar_start_flags, threshold_beats=4.0
        )

        # Apply trim offset to melody and chords
        if trim_offset > 0:
            melody_notes = apply_time_offset(melody_notes, trim_offset)
            chords = apply_time_offset(chords, trim_offset)
            # Also adjust beat times for quantization
            adjusted_beat_times = [
                bt - trim_offset for bt in beat_times if bt >= trim_offset
            ]
        else:
            adjusted_beat_times = beat_times

        # Quantize to beat grid
        quantized_chords = quantize_to_beats(chords, adjusted_beat_times)
        quantized_melody = quantize_to_beats(melody_notes, adjusted_beat_times)

        # Resolve overlaps
        # For melody: truncate earlier notes to avoid collisions
        quantized_melody = resolve_melody_overlaps(quantized_melody)

        # Calculate total duration in beats
        max_offset = 0
        if quantized_chords:
            max_offset = max(max_offset, max(c["offset"] for c in quantized_chords))
        if quantized_melody:
            max_offset = max(max_offset, max(n["offset"] for n in quantized_melody))

        # Create beat sequences with 16th note resolution
        beat_sequence, downbeat_sequence, bar_start_sequence = create_beat_sequences(
            beat_times,
            downbeat_flags,
            bar_start_flags,
            total_duration=max_offset,
            trim_offset=trim_offset,
            resolution=0.25,
        )

        # Create song dictionary in Hooktheory format
        song_dict = {
            "tags": ["MELODY", "HARMONY", "NO_SWING"],
            "split": "TRAIN",  # Will be reassigned later
            "pop909": {
                "id": song_id,
                "title": f"POP909 Song {song_id}",
                "source": "POP909 Dataset",
                "file": midi_file.name,
            },
            "annotations": {
                "num_beats": int(max_offset) if max_offset > 0 else 32,
                "meters": [{"beat": 0, "beats_per_bar": 4, "beat_unit": 4}],
                "keys": (keys if keys else ["C:maj"]),  # Use detected keys or default
                "bpm": bpm,
                "time_signature": time_signature,
                "melody": quantized_melody,
                "harmony": quantized_chords,
                "beat_sequence": beat_sequence,
                "downbeat_sequence": downbeat_sequence,
                "bar_start_sequence": bar_start_sequence,
            },
        }

        return song_dict

    except Exception as e:
        print(f"Error processing song {song_id}: {e}")
        return None


def split_dataset(
    songs: List[Dict], train_ratio: float = 0.8, valid_ratio: float = 0.1
) -> Dict[str, List[Dict]]:
    """Split the dataset into train/valid/test splits.

    Args:
        songs (List[Dict]): List of song dictionaries
        train_ratio (float): Ratio for training set
        valid_ratio (float): Ratio for validation set

    Returns:
        Dict[str, List[Dict]]: Dictionary with train/valid/test splits
    """
    total = len(songs)
    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)

    # Shuffle songs deterministically

    random.seed(42)
    random.shuffle(songs)

    splits = {
        "train": songs[:train_end],
        "valid": songs[train_end:valid_end],
        "test": songs[valid_end:],
    }

    # Update split tags
    for split_name, split_songs in splits.items():
        for song in split_songs:
            song["split"] = split_name.upper()

    return splits


def create_augmented_dataset(songs: List[Dict]) -> List[Dict]:
    """Create augmented dataset with transposition [-6, +6] semitones.

    Args:
        songs (List[Dict]): Original songs

    Returns:
        List[Dict]: Augmented songs (original + 12 transposed versions each)
    """
    augmented_songs = []

    for song in tqdm(songs, desc="Augmenting training data"):
        for semitone in range(-6, 7):  # -6 to +6 semitones
            augmented_song = augment_song_data(song, semitone)
            augmented_songs.append(augmented_song)

    print(
        f"Augmented training data. New number of training samples: {len(augmented_songs)}"
    )
    return augmented_songs


def augment_song_data(song_data: Dict, semitones: int = 0) -> Dict:
    """Apply transposition augmentation to a song.

    Args:
        song_data (Dict): Original song data
        semitones (int): Number of semitones to transpose (0 = no change)

    Returns:
        Dict: Augmented song data
    """
    if semitones == 0:
        return song_data

    augmented_data = deepcopy(song_data)

    # Transpose melody
    augmented_melody = []
    for note in song_data["annotations"]["melody"]:
        note_transposed = transpose_melody(note, semitones)
        augmented_melody.append(note_transposed)
    augmented_data["annotations"]["melody"] = augmented_melody

    # Transpose harmony
    augmented_harmony = []
    for chord in song_data["annotations"]["harmony"]:
        chord_transposed = transpose_chord(chord, semitones)
        augmented_harmony.append(chord_transposed)
    augmented_data["annotations"]["harmony"] = augmented_harmony

    return augmented_data


def collect_chord_names(songs: List[Dict]) -> List[str]:
    """Collect all unique chord names from the dataset.

    Args:
        songs (List[Dict]): List of song dictionaries

    Returns:
        List[str]: Sorted list of unique chord names
    """

    chord_names = set()

    for song in songs:
        for chord in song["annotations"]["harmony"]:
            chord_name = to_chord_name(
                chord["root_pitch_class"],
                chord["root_position_intervals"],
                chord["inversion"],
            )
            chord_names.add(chord_name)

    return sorted(list(chord_names))


def main():
    parser = argparse.ArgumentParser(
        description="Convert POP909 dataset to Hooktheory cache format"
    )
    parser.add_argument(
        "--pop909_path",
        type=str,
        default="data/pop909/POP909/POP909",
        help="Path to POP909 dataset directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/cache/pop909",
        help="Output directory for cache files",
    )
    parser.add_argument(
        "--max_songs",
        type=int,
        default=None,
        help="Maximum number of songs to process (for testing)",
    )
    parser.add_argument(
        "--augmentation",
        action="store_true",
        help="Create augmented dataset with transposition",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all song directories
    pop909_path = Path(args.pop909_path)
    song_dirs = [d for d in pop909_path.iterdir() if d.is_dir()]

    if args.max_songs:
        song_dirs = song_dirs[: args.max_songs]

    print(f"Found {len(song_dirs)} song directories to process")

    # Process all songs
    all_songs = []

    for song_dir in tqdm(song_dirs, desc="Processing POP909 songs"):
        try:
            song_dict = process_pop909_song(song_dir)
            if song_dict:
                all_songs.append(song_dict)
        except Exception as e:
            print(f"Error processing {song_dir.name}: {e}")
            continue

    print(f"Successfully processed {len(all_songs)} songs")

    if not all_songs:
        print("No songs were successfully processed!")
        return

    # Split dataset
    splits = split_dataset(all_songs)

    print(f"Dataset splits:")
    print(f"  Train: {len(splits['train'])} songs")
    print(f"  Valid: {len(splits['valid'])} songs")
    print(f"  Test: {len(splits['test'])} songs")

    # Collect chord names (before augmentation)
    chord_names = collect_chord_names(all_songs)
    print(f"Found {len(chord_names)} unique chord names")

    # Define cache_dir early for use in augmentation
    cache_dir = str(output_dir.parent)  # Go up to data/cache/

    # Create augmented dataset if requested
    if args.augmentation:
        print("\n=== Creating Augmented Dataset ===")
        # Augment training data
        augmented_train = create_augmented_dataset(splits["train"])

        # For augmented datasets, we need to collect chord names from all data including augmentation
        print("Collecting chord names from augmented training data...")
        augmented_chord_names = collect_chord_names(
            augmented_train + splits["valid"] + splits["test"]
        )
        print(
            f"Found {len(augmented_chord_names)} unique chord names in augmented dataset"
        )

        # Save augmented splits
        augmented_splits = {
            "train": augmented_train,
            "valid": splits["valid"],  # Keep valid/test unaugmented
            "test": splits["test"],
        }

        for split_name, split_songs in augmented_splits.items():
            cache_path = output_dir / f"{split_name}_augmented.jsonl"
            save_jsonl(split_songs, cache_path)
            print(f"Saved {split_name} augmented split to {cache_path}")

        # Save augmented chord names
        chord_names_aug_path = output_dir / "chord_names_augmented.json"
        with open(chord_names_aug_path, "w") as f:
            json.dump(augmented_chord_names, f, indent=2)
        print(f"Saved augmented chord names to {chord_names_aug_path}")

        # Update global augmented chord names
        print(f"\nUpdating global augmented chord_names in {cache_dir}...")
        update_global_chord_names(augmented_chord_names, cache_dir, augmented=True)

    # Save regular (non-augmented) splits
    for split_name, split_songs in splits.items():
        cache_path = output_dir / f"{split_name}.jsonl"
        save_jsonl(split_songs, cache_path)
        print(f"Saved {split_name} split to {cache_path}")

    # Save regular chord names
    chord_names_path = output_dir / "chord_names.json"
    with open(chord_names_path, "w") as f:
        json.dump(chord_names, f, indent=2)
    print(f"Saved chord names to {chord_names_path}")

    # Update global chord names
    print(f"\nUpdating global chord_names in {cache_dir}...")
    update_global_chord_names(chord_names, cache_dir, augmented=False)

    print("POP909 dataset conversion completed!")


if __name__ == "__main__":
    main()
