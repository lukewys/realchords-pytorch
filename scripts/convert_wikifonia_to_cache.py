"""Convert Wikifonia dataset to Hooktheory-compatible cache format.

This script processes MusicXML files from the Wikifonia dataset and converts
them to the same cache format used by the Hooktheory dataset, allowing them to be
loaded by the existing HooktheoryDataset dataloader.
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import music21
from music21 import converter, meter, key, note, chord as music21_chord
from music21 import stream as music21_stream, harmony
import argparse
import random
import numpy as np
from copy import deepcopy
import note_seq.chord_symbols_lib as chord_symbols_lib

from realchords.constants import ZERO_OCTAVE
from realchords.utils.io_utils import save_jsonl
from realchords.utils.data_utils import (
    to_chord_name,
    transpose_melody,
    transpose_chord,
    update_global_chord_names,
)


def transform_wikifonia_chord_symbol(chord_symbol: str) -> str:
    """Transform Wikifonia chord notation to note_seq compatible format.

    Args:
        chord_symbol (str): Original Wikifonia chord symbol

    Returns:
        str: Transformed chord symbol compatible with note_seq
    """
    if not chord_symbol or chord_symbol.strip() in ["N", "X", "", "NC"]:
        return ""

    chord = chord_symbol.strip()

    # Handle flat notation: E- → Eb, B- → Bb, etc.
    chord = re.sub(r"([A-G])-", r"\1b", chord)

    # Handle diminished notation
    chord = re.sub(r"dim", r"dim", chord)  # Keep dim as is
    chord = re.sub(r"([A-G][#b]?)o7", r"\1dim7", chord)  # Ao7 → Adim7
    chord = re.sub(r"([A-G][#b]?)ø7", r"\1m7b5", chord)  # Aø7 → Am7b5 (half-diminished)

    # Handle augmented notation
    chord = re.sub(r"([A-G][#b]?)\+$", r"\1aug", chord)  # G+ → Gaug
    chord = re.sub(r"([A-G][#b]?)7\+", r"\1aug7", chord)  # G7+ → Gaug7

    # Handle complex notation (order matters!)
    chord = re.sub(r"sus add 7 add 9", r"9sus4", chord)  # sus add 7 add 9 → 9sus4
    chord = re.sub(r"sus add 7", r"7sus4", chord)  # sus add 7 → 7sus4
    chord = re.sub(r"sus add 9", r"9sus4", chord)  # sus add 9 → 9sus4
    chord = re.sub(r"alter b5", r"b5", chord)  # alter b5 → b5

    # Handle add notation
    chord = re.sub(r"add b9", r"b9", chord)  # add b9 → b9
    chord = re.sub(r"add 9", r"add9", chord)  # add 9 → add9

    # Remove extra spaces that might cause parsing issues
    chord = re.sub(r"\s+", "", chord)  # Remove all spaces

    # Fix sus chord format issues
    chord = re.sub(
        r"([A-G][#b]?)7sus4", r"\1sus4", chord
    )  # D7sus4 → Dsus4 (note_seq prefers this)

    return chord


def simplify_complex_chord(chord_symbol: str) -> str:
    """Simplify complex chords that note_seq cannot parse to basic chords.

    Args:
        chord_symbol (str): Complex chord symbol that failed parsing

    Returns:
        str: Simplified chord symbol
    """
    chord = chord_symbol.strip()

    # Extract root note (with accidentals)
    root_match = re.match(r"^([A-G][#b]?)", chord)
    if not root_match:
        return "C"  # Fallback to C major

    root = root_match.group(1)

    # Handle pedal tones - just use the root as a major chord
    if "pedal" in chord.lower():
        return root

    # Handle power chords - convert to major triad
    if "power" in chord.lower():
        return root

    # Handle complex alterations - simplify to basic chord types
    if any(x in chord for x in ["alter", "#5", "b5", "#9", "b9", "#11", "b13"]):
        # Check if it's minor or major based on existing chord structure
        if "m" in chord and not "maj" in chord:
            if "7" in chord:
                return f"{root}m7"
            else:
                return f"{root}m"
        else:
            if "7" in chord:
                return f"{root}7"
            else:
                return root

    # Handle complex sus chords - simplify to sus4
    if "sus" in chord:
        return f"{root}sus4"

    # Handle complex slash chords with double flats/sharps
    if "/" in chord:
        base_chord, bass = chord.split("/", 1)
        # Clean up bass note (remove double accidentals)
        bass = re.sub(r"([A-G])[#b]{2,}", r"\1", bass)  # Remove double accidentals
        bass = re.sub(r"[^A-G#b]", "", bass)  # Remove non-note characters

        if bass and bass != root:
            # Simplify base chord first
            simplified_base = simplify_complex_chord(base_chord)
            return f"{simplified_base}/{bass}"
        else:
            return simplify_complex_chord(base_chord)

    # Handle extended chords (9, 11, 13) - simplify to 7th chords
    chord = re.sub(r"(9|11|13)sus4", r"7sus4", chord)  # 9sus4 -> 7sus4
    chord = re.sub(r"(9|11|13)", r"7", chord)  # 9/11/13 -> 7

    # If still complex, fall back to basic triads
    if "m" in chord and not "maj" in chord:
        return f"{root}m"
    elif "7" in chord:
        return f"{root}7"
    else:
        return root


def parse_chord_symbol_with_noteseq(
    chord_symbol: str,
) -> Tuple[int, List[int], int]:
    """Convert chord symbol to root pitch class, intervals, and inversion using note_seq.

    Args:
        chord_symbol (str): Chord symbol (e.g., "C", "Dm", "G7", "C/E")

    Returns:
        Tuple[int, List[int], int]: (root_pitch_class, intervals, inversion)
    """
    # Transform Wikifonia chord notation to note_seq compatible format
    transformed_chord = transform_wikifonia_chord_symbol(chord_symbol)

    # Handle empty or invalid chord symbols
    if not transformed_chord:
        return 0, [], 0  # Will be filtered out later

    try:
        # Get pitches from note_seq using transformed chord
        chord_pitches = chord_symbols_lib.chord_symbol_pitches(transformed_chord)

        if not chord_pitches:
            # Try to simplify the chord and parse again
            simplified_chord = simplify_complex_chord(transformed_chord)
            try:
                chord_pitches = chord_symbols_lib.chord_symbol_pitches(simplified_chord)
                if not chord_pitches:
                    # Final fallback to C major
                    return 0, [4, 3], 0
                print(
                    f"Simplified '{chord_symbol}' -> '{simplified_chord}' successfully"
                )
            except:
                # Final fallback to C major
                return 0, [4, 3], 0

        # Extract root pitch class (first pitch)
        root_pitch_class = chord_pitches[0] % 12

        # Calculate intervals between consecutive pitches
        intervals = []
        for i in range(1, len(chord_pitches)):
            interval = (chord_pitches[i] - chord_pitches[i - 1]) % 12
            intervals.append(interval)

        # Calculate inversion based on bass note position
        inversion = 0
        if "/" in transformed_chord:
            # For slash chords, find which chord tone is in bass
            base_chord_name = transformed_chord.split("/")[0]
            bass_note_name = transformed_chord.split("/")[1]

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
        print(f"Error parsing chord '{chord_symbol}' -> '{transformed_chord}': {e}")
        # Try to simplify the chord and parse again
        try:
            simplified_chord = simplify_complex_chord(transformed_chord)
            chord_pitches = chord_symbols_lib.chord_symbol_pitches(simplified_chord)
            if chord_pitches:
                print(
                    f"Simplified '{chord_symbol}' -> '{simplified_chord}' successfully"
                )

                # Extract root pitch class and intervals from simplified chord
                root_pitch_class = chord_pitches[0] % 12
                intervals = []
                for i in range(1, len(chord_pitches)):
                    interval = (chord_pitches[i] - chord_pitches[i - 1]) % 12
                    intervals.append(interval)

                # For simplified chords, assume root position
                return root_pitch_class, intervals, 0
            else:
                # Final fallback to C major
                return 0, [4, 3], 0
        except:
            # Final fallback to C major
            return 0, [4, 3], 0


def extract_melody_and_chords_from_musicxml(xml_file: Path) -> Optional[Dict]:
    """Extract melody and chord symbols from a MusicXML file.

    Args:
        xml_file (Path): Path to MusicXML file

    Returns:
        Optional[Dict]: Dictionary with melody, chords, and metadata or None if parsing failed
    """
    try:
        score = music21.converter.parse(str(xml_file))
        if score is None:
            return None

        # Get metadata
        metadata = {
            "title": (
                score.metadata.title
                if score.metadata and score.metadata.title
                else xml_file.stem
            ),
            "composer": (
                score.metadata.composer
                if score.metadata and score.metadata.composer
                else None
            ),
            "source": str(xml_file),
        }

        # Extract time signature and key signature for metadata
        time_sigs = score.flat.getElementsByClass(meter.TimeSignature)
        key_sigs = score.flat.getElementsByClass(key.KeySignature)

        if time_sigs:
            metadata["time_signature"] = str(time_sigs[0])
        if key_sigs:
            metadata["key_signature"] = str(key_sigs[0])

        # Extract melody from the first part (assuming single part as specified)
        if not score.parts:
            # No parts, try to get notes from the score directly
            melody_notes_raw = score.flat.notes
        else:
            # Get notes from first part
            melody_notes_raw = score.parts[0].flat.notes

        melody_notes = []
        for element in melody_notes_raw:
            if isinstance(element, note.Note):
                # Convert to onset/offset format with proper pitch handling
                onset = float(element.offset)
                duration = float(element.quarterLength)
                offset = onset + duration

                # Convert MIDI pitch to octave and pitch class (following Hooktheory format)
                midi_number = element.pitch.midi
                relative_pitch = midi_number - ZERO_OCTAVE  # Remove ZERO_OCTAVE offset
                octave = relative_pitch // 12
                pitch_class = relative_pitch % 12

                melody_notes.append(
                    {
                        "onset": onset,
                        "offset": offset,
                        "pitch_class": pitch_class,
                        "octave": octave,
                    }
                )

        # Extract chord symbols
        chord_symbols = score.flat.getElementsByClass(harmony.ChordSymbol)
        chords = []

        if chord_symbols:
            # Use chord symbols with proper durations
            harmony.realizeChordSymbolDurations(score)

            for chord_symbol in chord_symbols:
                try:
                    onset = float(chord_symbol.offset)
                    duration = float(chord_symbol.quarterLength)
                    offset = onset + duration

                    # Get the chord symbol text
                    chord_text = (
                        str(chord_symbol.figure)
                        if hasattr(chord_symbol, "figure")
                        else str(chord_symbol)
                    )

                    # Parse chord symbol
                    (
                        root_pitch_class,
                        intervals,
                        inversion,
                    ) = parse_chord_symbol_with_noteseq(chord_text)

                    # Skip if no intervals (invalid chord)
                    if not intervals:
                        continue

                    chords.append(
                        {
                            "onset": onset,
                            "offset": offset,
                            "root_pitch_class": root_pitch_class,
                            "root_position_intervals": intervals,
                            "inversion": inversion,
                        }
                    )

                except Exception as e:
                    print(f"Error processing chord symbol {chord_symbol}: {e}")
                    continue

        # Return parsed data
        return {
            "melody": melody_notes,
            "chords": chords,
            "metadata": metadata,
        }

    except Exception as e:
        print(f"Error parsing {xml_file}: {e}")
        return None


def quantize_timing_to_beat_grid(
    notes_or_chords: List[Dict], resolution: float = 0.25
) -> List[Dict]:
    """Quantize note/chord timings to beat grid with 16th note resolution.

    Args:
        notes_or_chords (List[Dict]): List of notes or chords with onset/offset
        resolution (float): Grid resolution in quarter-note units (0.25 = 16th note)

    Returns:
        List[Dict]: Quantized notes/chords
    """
    quantized = []

    for item in notes_or_chords:
        # Quantize onset and offset to the grid
        quantized_onset = round(item["onset"] / resolution) * resolution
        quantized_offset = round(item["offset"] / resolution) * resolution

        # Ensure minimum duration of one grid unit
        if quantized_offset <= quantized_onset:
            quantized_offset = quantized_onset + resolution

        # Create quantized item
        quantized_item = deepcopy(item)
        quantized_item["onset"] = quantized_onset
        quantized_item["offset"] = quantized_offset

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
            min_duration = 0.25  # 16th note
            if current_note["offset"] <= current_note["onset"]:
                current_note["offset"] = current_note["onset"] + min_duration

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
            min_duration = 0.25  # 16th note
            if longest_note["offset"] <= longest_note["onset"]:
                longest_note["offset"] = longest_note["onset"] + min_duration

            resolved_notes.append(longest_note)

        i = j

    return resolved_notes


def filter_zero_duration_chords(chords: List[Dict]) -> List[Dict]:
    """Filter out zero or near-zero duration chords.

    Args:
        chords: List of chord dictionaries with onset/offset

    Returns:
        List of chords with positive duration
    """
    min_duration = 0.001  # 1ms minimum duration

    filtered_chords = []
    for chord in chords:
        duration = chord["offset"] - chord["onset"]
        if duration > min_duration:
            filtered_chords.append(chord)

    return filtered_chords


def process_wikifonia_file(xml_file: Path) -> Optional[Dict]:
    """Process a single Wikifonia MusicXML file.

    Args:
        xml_file (Path): Path to MusicXML file

    Returns:
        Optional[Dict]: Processed song dictionary or None if processing failed
    """
    song_id = xml_file.stem

    # Extract melody and chords
    parsed_data = extract_melody_and_chords_from_musicxml(xml_file)
    if not parsed_data:
        return None

    melody_notes = parsed_data["melody"]
    chords = parsed_data["chords"]
    metadata = parsed_data["metadata"]

    # Skip if no valid data
    if not melody_notes or not chords:
        print(f"No valid melody or chords for {song_id}")
        return None

    # Quantize timing to 16th note grid
    quantized_melody = quantize_timing_to_beat_grid(melody_notes, resolution=0.25)
    quantized_chords = quantize_timing_to_beat_grid(chords, resolution=0.25)

    # Resolve overlaps
    quantized_melody = resolve_melody_overlaps(quantized_melody)
    quantized_chords = filter_zero_duration_chords(quantized_chords)

    # Calculate total duration
    max_offset = 0
    if quantized_chords:
        max_offset = max(max_offset, max(c["offset"] for c in quantized_chords))
    if quantized_melody:
        max_offset = max(max_offset, max(n["offset"] for n in quantized_melody))

    # Create song dictionary in Hooktheory format
    song_dict = {
        "tags": ["MELODY", "HARMONY", "NO_SWING"],
        "split": "TRAIN",  # Will be reassigned later
        "wikifonia": {
            "id": song_id,
            "title": metadata["title"],
            "composer": metadata.get("composer"),
            "source": "Wikifonia Dataset",
            "file": xml_file.name,
            "time_signature": metadata.get("time_signature"),
            "key_signature": metadata.get("key_signature"),
        },
        "annotations": {
            "num_beats": int(max_offset) if max_offset > 0 else 32,
            "meters": [{"beat": 0, "beats_per_bar": 4, "beat_unit": 4}],
            "keys": [
                {
                    "beat": 0,
                    "tonic_pitch_class": 0,  # Default to C major
                    "scale_degree_intervals": [2, 2, 1, 2, 2, 2],  # Major scale
                }
            ],
            "melody": quantized_melody,
            "harmony": quantized_chords,
        },
    }

    return song_dict


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


def main():
    parser = argparse.ArgumentParser(
        description="Convert Wikifonia dataset to Hooktheory cache format"
    )
    parser.add_argument(
        "--wikifonia_path",
        type=str,
        default="data/wikifonia/wikifonia",
        help="Path to Wikifonia directory containing MusicXML files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/cache/wikifonia",
        help="Output directory for cache files",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of XML files to process (for testing)",
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

    # Find all MusicXML files (recursively search for .mxl files)
    wikifonia_path = Path(args.wikifonia_path)
    xml_files = list(wikifonia_path.glob("**/*.mxl"))

    if not xml_files:
        print(f"No MusicXML files found in {wikifonia_path}")
        return

    if args.max_files:
        xml_files = xml_files[: args.max_files]

    print(f"Found {len(xml_files)} MusicXML files to process")

    # Process all XML files
    all_songs = []

    for xml_file in tqdm(xml_files, desc="Processing Wikifonia files"):
        try:
            song_dict = process_wikifonia_file(xml_file)
            if song_dict:
                all_songs.append(song_dict)
        except Exception as e:
            print(f"Error processing {xml_file.name}: {e}")
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

    print("Wikifonia dataset conversion completed!")


if __name__ == "__main__":
    main()
