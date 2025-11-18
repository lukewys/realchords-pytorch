"""Convert Nottingham dataset ABC files to Hooktheory-compatible cache format.

This script processes ABC notation files from the Nottingham dataset and converts
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
from copy import deepcopy

from realchords.utils.io_utils import save_jsonl
from realchords.utils.data_utils import (
    to_chord_name,
    transpose_melody,
    transpose_chord,
    update_global_chord_names,
)


def parse_abc_header(abc_content: str) -> Dict:
    """Parse ABC notation header to extract metadata.

    Args:
        abc_content (str): Raw ABC notation content

    Returns:
        Dict: Parsed metadata including title, meter, key, etc.
    """
    metadata = {}
    lines = abc_content.strip().split("\n")

    for line in lines:
        line = line.strip()
        if line.startswith("X:"):
            metadata["index"] = line[2:].strip()
        elif line.startswith("T:"):
            metadata["title"] = line[2:].strip()
        elif line.startswith("M:"):
            metadata["meter"] = line[2:].strip()
        elif line.startswith("K:"):
            metadata["key"] = line[2:].strip()
        elif line.startswith("L:"):
            metadata["note_length"] = line[2:].strip()
        elif line.startswith("S:"):
            metadata["source"] = line[2:].strip()
        elif line.startswith("Y:"):
            metadata["year"] = line[2:].strip()
        elif line.startswith("P:"):
            metadata["part"] = line[2:].strip()

    return metadata


def extract_chord_symbols(abc_content: str) -> List[Tuple[str, int]]:
    """Extract chord symbols and their positions from ABC notation.

    Args:
        abc_content (str): ABC notation content

    Returns:
        List[Tuple[str, int]]: List of (chord_symbol, position) tuples
    """
    # Find all chord symbols in quotes
    chord_pattern = r'"([^"]+)"'
    chords = []

    # Remove header lines and get only the music content
    lines = abc_content.strip().split("\n")
    music_lines = []
    in_header = True

    for line in lines:
        line = line.strip()
        if line.startswith("K:"):
            in_header = False
            continue
        if not in_header and line and not line.startswith("%"):
            music_lines.append(line)

    # Join all music lines and find chord positions
    music_content = " ".join(music_lines)

    position = 0
    for match in re.finditer(chord_pattern, music_content):
        chord_symbol = match.group(1)
        chords.append((chord_symbol, position))
        position += 1

    return chords


def chord_symbol_to_intervals(chord_symbol: str) -> Tuple[int, List[int], int]:
    """Convert chord symbol to root pitch class, intervals, and inversion.

    Args:
        chord_symbol (str): Chord symbol (e.g., "Am", "G7", "F#m")

    Returns:
        Tuple[int, List[int], int]: (root_pitch_class, intervals, inversion)
    """
    # Map note names to pitch classes
    note_to_pitch = {
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

    # Clean the chord symbol
    chord_symbol = chord_symbol.strip()

    # Handle basic major/minor chords
    if chord_symbol.endswith("m"):
        # Minor chord
        root_name = chord_symbol[:-1]
        intervals = [3, 4]  # Minor third, major third
    elif chord_symbol.endswith("7"):
        # Dominant 7th chord
        root_name = chord_symbol[:-1]
        intervals = [4, 3, 3]  # Major third, minor third, minor third
    elif chord_symbol.endswith("m7"):
        # Minor 7th chord
        root_name = chord_symbol[:-2]
        intervals = [3, 4, 3]  # Minor third, major third, minor third
    elif chord_symbol.endswith("maj7"):
        # Major 7th chord
        root_name = chord_symbol[:-4]
        intervals = [4, 3, 4]  # Major third, minor third, major third
    else:
        # Major chord (default)
        root_name = chord_symbol
        intervals = [4, 3]  # Major third, minor third

    # Handle accidentals
    if len(root_name) > 1 and root_name[1] in ["#", "b"]:
        root_pitch_class = note_to_pitch.get(root_name[:2], 0)
    else:
        root_pitch_class = note_to_pitch.get(root_name[0], 0)

    return root_pitch_class, intervals, 0  # Assuming root position for now


def has_pickup_bars(stream):
    """Check if a stream has pickup bars (anacrusis).

    Args:
        stream: music21 stream object

    Returns:
        bool: True if pickup bars are detected
    """
    try:
        # Convert to measures if not already
        if not stream.hasMeasures():
            stream = stream.makeMeasures()

        # Check first few measures for pickup characteristics
        measures = stream.getElementsByClass("Measure")[:3]  # Check first 3 measures

        for measure in measures:
            try:
                bar_duration = measure.barDuration.quarterLength
                filled_duration = sum(
                    element.quarterLength for element in measure.notesAndRests
                )
                padding_left = getattr(measure, "paddingLeft", 0)

                # If there's significant padding or the measure is significantly underfilled
                if padding_left > 0 or filled_duration < (bar_duration * 0.5):
                    return True

            except Exception:
                continue

        return False

    except Exception:
        return False


def remove_pickup_bars(stream):
    """Remove pickup bars (anacrusis) from a music21 stream.

    Args:
        stream: music21 stream object

    Returns:
        music21 stream with pickup bars removed, or None if no valid measures remain
    """

    try:
        # Convert to measures if not already
        if not stream.hasMeasures():
            # Use stream.makeMeasures() method directly
            stream = stream.makeMeasures()

        # Find measures with pickup content
        measures_to_keep = []

        for measure in stream.getElementsByClass("Measure"):
            # Check if this is a pickup measure
            try:
                bar_duration = measure.barDuration.quarterLength
                filled_duration = sum(
                    element.quarterLength for element in measure.notesAndRests
                )
                padding_left = getattr(measure, "paddingLeft", 0)

                # If there's padding or the measure is significantly underfilled, it's likely a pickup
                is_pickup = padding_left > 0 or filled_duration < (bar_duration - 1e-6)

                if not is_pickup:
                    measures_to_keep.append(measure)

            except Exception as e:
                # If we can't determine if it's a pickup, keep the measure
                measures_to_keep.append(measure)

        if not measures_to_keep:
            return None

        # Create new stream with non-pickup measures
        new_stream = music21_stream.Stream()

        # Copy metadata once (avoid duplicates)
        copied_elements = set()
        for element in stream.flat:
            element_type = type(element).__name__
            if (
                element_type in ["Metadata", "TimeSignature", "KeySignature"]
                and element_type not in copied_elements
            ):
                new_stream.append(element)
                copied_elements.add(element_type)

        # Add valid measures with their content
        for measure in measures_to_keep:
            # Create a new measure to avoid object duplication issues
            new_measure = music21_stream.Measure()
            new_measure.offset = measure.offset

            # Copy notes and other elements from the measure
            for element in measure:
                if hasattr(element, "offset"):
                    element_copy = element
                    new_measure.append(element_copy)

            new_stream.append(new_measure)

        return new_stream

    except Exception as e:
        print(f"Error removing pickup bars: {e}")
        return stream  # Return original stream if processing fails


def resolve_melody_overlaps(notes: List[Dict]) -> List[Dict]:
    """Resolve melody note overlaps by truncating earlier notes.

    Args:
        notes: List of note dictionaries with onset/offset

    Returns:
        List of notes with overlaps resolved
    """
    if len(notes) <= 1:
        return notes

    min_duration = 0.25  # Minimum duration = one 16th note

    # Sort notes by onset time
    sorted_notes = sorted(notes, key=lambda x: x["onset"])
    resolved_notes = []

    for i, note in enumerate(sorted_notes):
        current_note = note.copy()

        # Check for overlaps with subsequent notes
        for j in range(i + 1, len(sorted_notes)):
            next_note = sorted_notes[j]

            # If next note starts after current note ends, no overlap
            if next_note["onset"] >= current_note["offset"]:
                break

            # Overlap detected - truncate current note
            current_note["offset"] = next_note["onset"]

        # Ensure minimum duration
        if current_note["offset"] <= current_note["onset"]:
            current_note["offset"] = current_note["onset"] + min_duration

        resolved_notes.append(current_note)

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


def parse_chord_sequence(stream):
    """Parse chord symbols from a music21 stream with proper durations.

    Args:
        stream: music21 stream object

    Returns:
        List of chord dictionaries with onset, offset, root_pitch_class, intervals, inversion
    """

    try:
        # Get all chord symbols from the stream
        chord_symbols = stream.flat.getElementsByClass(harmony.ChordSymbol)

        if not chord_symbols:
            return []

        # Realize chord symbol durations
        harmony.realizeChordSymbolDurations(stream)

        # Extract chord information with proper timing
        harmony_chords = []

        for chord_symbol in chord_symbols:
            try:
                onset = float(chord_symbol.offset)
                duration = float(chord_symbol.quarterLength)
                offset = onset + duration

                # Get the actual chord symbol text and convert it
                chord_text = (
                    str(chord_symbol.figure)
                    if hasattr(chord_symbol, "figure")
                    else str(chord_symbol)
                )

                # Convert chord symbol to intervals
                root_pitch_class, intervals, inversion = chord_symbol_to_intervals(
                    chord_text
                )

                harmony_chords.append(
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

        return harmony_chords

    except Exception as e:
        print(f"Error parsing chord sequence: {e}")
        return []


def parse_abc_file(abc_file_path: str) -> List[Dict]:
    """Parse an ABC file and extract all tunes.

    Args:
        abc_file_path (str): Path to ABC file

    Returns:
        List[Dict]: List of parsed tune dictionaries
    """
    with open(abc_file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Split content by X: field to separate individual tunes
    tunes = []
    tune_sections = re.split(r"\n(?=X:\s*\d+)", content)

    for i, section in enumerate(tune_sections):
        if not section.strip():
            continue

        try:
            # Parse with music21
            stream = converter.parse(section, format="abc")
            if stream is None:
                continue

            # Extract metadata
            metadata = parse_abc_header(section)

            # Check for pickup bars and remove them if found
            original_stream = stream
            try:
                # Only attempt pickup removal if we detect potential pickup measures
                if has_pickup_bars(stream):
                    stream = remove_pickup_bars(stream)
                    if stream is None or len(stream.flat.notesAndRests) == 0:
                        stream = original_stream
            except Exception as e:
                print(
                    f"Error in pickup removal for {metadata.get('title', 'Unknown')}: {e}, using original stream"
                )
                stream = original_stream

            # Process the stream to extract melody and harmony with proper chord durations
            melody_notes = []
            harmony_chords = []

            # Get time signature and key signature
            time_sig = stream.getElementsByClass(meter.TimeSignature)
            key_sig = stream.getElementsByClass(key.KeySignature)

            # Extract melody notes with proper timing after pickup removal
            notes_and_rests = stream.flat.notesAndRests
            current_offset = 0.0

            for element in notes_and_rests:
                if isinstance(element, note.Note):
                    onset = current_offset
                    duration = float(element.quarterLength)
                    offset = onset + duration

                    # Get MIDI number directly from music21 and extract octave and pitch class
                    midi_number = element.pitch.midi
                    # Convert to relative octave and pitch class to match Hooktheory format
                    # where ZERO_OCTAVE = 60 means C4 (MIDI 60) has octave=0, pitch_class=0
                    relative_pitch = midi_number - 60  # Remove ZERO_OCTAVE offset
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

                current_offset += float(element.quarterLength)

            # Extract and process chord symbols with proper durations
            harmony_chords = parse_chord_sequence(stream)

            # Fallback to ABC text extraction if music21 parsing failed
            if not harmony_chords:
                chord_symbols = extract_chord_symbols(section)

                if chord_symbols:
                    chord_duration = 2.0  # Default chord duration
                    current_chord_offset = 0.0

                    for chord_symbol, _ in chord_symbols:
                        try:
                            (
                                root_pitch_class,
                                intervals,
                                inversion,
                            ) = chord_symbol_to_intervals(chord_symbol)

                            harmony_chords.append(
                                {
                                    "onset": current_chord_offset,
                                    "offset": current_chord_offset + chord_duration,
                                    "root_pitch_class": root_pitch_class,
                                    "root_position_intervals": intervals,
                                    "inversion": inversion,
                                }
                            )

                            current_chord_offset += chord_duration
                        except Exception as e:
                            print(f"Error converting chord {chord_symbol}: {e}")
                            continue

            # Apply overlap resolution
            # For melody: truncate earlier notes to avoid collisions
            melody_notes = resolve_melody_overlaps(melody_notes)

            # For chords: filter out zero-duration chords
            harmony_chords = filter_zero_duration_chords(harmony_chords)

            # Create tune dictionary in Hooktheory format
            tune_id = (
                f"nottingham_{Path(abc_file_path).stem}_{metadata.get('index', i)}"
            )

            tune_dict = {
                "tags": ["MELODY", "HARMONY", "NO_SWING"],
                "split": "TRAIN",  # Will be reassigned later
                "nottingham": {
                    "id": tune_id,
                    "title": metadata.get("title", "Unknown"),
                    "source": metadata.get("source", "Nottingham Database"),
                    "file": Path(abc_file_path).name,
                    "index": metadata.get("index", str(i)),
                },
                "annotations": {
                    "num_beats": (int(current_offset) if current_offset > 0 else 32),
                    "meters": [{"beat": 0, "beats_per_bar": 4, "beat_unit": 4}],
                    "keys": [
                        {
                            "beat": 0,
                            "tonic_pitch_class": 0,  # Default to C major
                            "scale_degree_intervals": [
                                2,
                                2,
                                1,
                                2,
                                2,
                                2,
                            ],  # Major scale
                        }
                    ],
                    "melody": melody_notes,
                    "harmony": harmony_chords,
                },
            }

            # Only add tunes that have both melody and harmony
            if melody_notes and harmony_chords:
                tunes.append(tune_dict)

        except Exception as e:
            print(f"Error processing tune in {abc_file_path}: {e}")
            continue

    return tunes


def split_dataset(
    tunes: List[Dict], train_ratio: float = 0.8, valid_ratio: float = 0.1
) -> Dict[str, List[Dict]]:
    """Split the dataset into train/valid/test splits.

    Args:
        tunes (List[Dict]): List of tune dictionaries
        train_ratio (float): Ratio for training set
        valid_ratio (float): Ratio for validation set

    Returns:
        Dict[str, List[Dict]]: Dictionary with train/valid/test splits
    """
    total = len(tunes)
    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)

    # Shuffle tunes deterministically

    random.seed(42)
    random.shuffle(tunes)

    splits = {
        "train": tunes[:train_end],
        "valid": tunes[train_end:valid_end],
        "test": tunes[valid_end:],
    }

    # Update split tags
    for split_name, split_tunes in splits.items():
        for tune in split_tunes:
            tune["split"] = split_name.upper()

    return splits


def collect_chord_names(tunes: List[Dict]) -> List[str]:
    """Collect all unique chord names from the dataset.

    Args:
        tunes (List[Dict]): List of tune dictionaries

    Returns:
        List[str]: Sorted list of unique chord names
    """

    chord_names = set()

    for tune in tunes:
        for chord in tune["annotations"]["harmony"]:
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
        description="Convert Nottingham dataset to Hooktheory cache format"
    )
    parser.add_argument(
        "--nottingham_path",
        type=str,
        default="data/nottingham/nottingham-dataset/ABC_cleaned",
        help="Path to Nottingham ABC_cleaned directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/cache/nottingham",
        help="Output directory for cache files",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of ABC files to process (for testing)",
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

    # Find all ABC files
    abc_files = list(Path(args.nottingham_path).glob("*.abc"))
    if args.max_files:
        abc_files = abc_files[: args.max_files]

    print(f"Found {len(abc_files)} ABC files to process")

    # Process all ABC files
    all_tunes = []

    for abc_file in tqdm(abc_files, desc="Processing ABC files"):
        try:
            tunes = parse_abc_file(str(abc_file))
            all_tunes.extend(tunes)
            print(f"Processed {abc_file.name}: {len(tunes)} tunes")
        except Exception as e:
            print(f"Error processing {abc_file}: {e}")
            continue

    print(f"Total processed tunes: {len(all_tunes)}")

    if not all_tunes:
        print("No tunes were successfully processed!")
        return

    # Split dataset
    splits = split_dataset(all_tunes)

    print(f"Dataset splits:")
    print(f"  Train: {len(splits['train'])} tunes")
    print(f"  Valid: {len(splits['valid'])} tunes")
    print(f"  Test: {len(splits['test'])} tunes")

    # Collect chord names (before augmentation)
    chord_names = collect_chord_names(all_tunes)
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
    for split_name, split_tunes in splits.items():
        cache_path = output_dir / f"{split_name}.jsonl"
        save_jsonl(split_tunes, cache_path)
        print(f"Saved {split_name} split to {cache_path}")

    # Save regular chord names
    chord_names_path = output_dir / "chord_names.json"
    with open(chord_names_path, "w") as f:
        json.dump(chord_names, f, indent=2)
    print(f"Saved chord names to {chord_names_path}")

    # Update global chord names
    print(f"\nUpdating global chord_names in {cache_dir}...")
    update_global_chord_names(chord_names, cache_dir, augmented=False)

    print("Nottingham dataset conversion completed!")


if __name__ == "__main__":
    main()
