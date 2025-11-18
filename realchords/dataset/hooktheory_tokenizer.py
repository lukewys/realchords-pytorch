"""Convert Hooktheory dataset to framewise sequences."""

from typing import Any, Dict, List, Optional, Tuple

import note_seq
import note_seq.chord_symbols_lib as chord_symbols_lib
import numpy as np
import pretty_midi
import torch

from realchords.constants import (
    BASS_OCTAVE,
    BASS_VELOCITY,
    BOS_TOKEN,
    CHORD_OCTAVE,
    CHORD_VELOCITY,
    EOS_TOKEN,
    MELODY_VELOCITY,
    PAD_TOKEN,
    ZERO_OCTAVE,
)
from realchords.utils.data_utils import to_chord_name


def to_midi_pitch(octave: int, pitch_class: int) -> int:
    """
    Convert octave and pitch class to MIDI pitch.

    Args:
        octave (int): The octave number.
        pitch_class (int): The pitch class.

    Returns:
        int: The MIDI pitch.
    """
    return octave * 12 + pitch_class + ZERO_OCTAVE


class HooktheoryTokenizer:
    def __init__(
        self,
        frame_per_beat: int = 4,
        num_pitches: int = 128,
        chord_names: Optional[List[str]] = None,
    ):
        """
        Initialize the HooktheoryTokenizer.

        Args:
            frame_per_beat (int): Number of frames per beat.
            num_pitches (int): Number of pitches.
            chord_names (Optional[List[str]]): List of chord names.
        """
        self.frame_per_beat = frame_per_beat
        self.num_pitches = num_pitches
        self.chord_names = chord_names
        self.init_dict()

    def init_dict(self) -> None:
        """
        Initialize the token dictionary.
        """
        self.name_to_id = {}
        special_tokens = {
            "PAD": PAD_TOKEN,
            "BOS": BOS_TOKEN,
            "EOS": EOS_TOKEN,
        }
        self.name_to_id.update(special_tokens)
        self.special_token_range = (0, self.eos_token)

        # Silence tokens
        self.name_to_id.update({"SILENCE": len(self.name_to_id)})

        pitch_to_token_id = {
            f"NOTE_{pitch}": i + len(self.name_to_id)
            for i, pitch in enumerate(range(self.num_pitches))
        }
        self.name_to_id.update(pitch_to_token_id)
        self.note_hold_token_range = (
            len(special_tokens) + 1,  # SILENCE
            len(self.name_to_id) - 1,
        )

        pitch_onset_to_token_id = {
            f"NOTE_ON_{pitch}": i + len(self.name_to_id)
            for i, pitch in enumerate(range(self.num_pitches))
        }
        self.name_to_id.update(pitch_onset_to_token_id)
        self.note_on_token_range = (
            self.note_hold_token_range[1] + 1,
            len(self.name_to_id) - 1,
        )

        self.note_token_range = (
            len(special_tokens) + 1,  # SILENCE
            len(self.name_to_id) - 1,
        )

        chord_to_token_id = {
            f"CHORD_{chord}": i + len(self.name_to_id)
            for i, chord in enumerate(self.chord_names)
        }
        self.name_to_id.update(chord_to_token_id)
        self.chord_hold_token_range = (
            self.note_token_range[1] + 1,
            len(self.name_to_id) - 1,
        )

        chord_onset_to_token_id = {
            f"CHORD_ON_{chord}": i + len(self.name_to_id)
            for i, chord in enumerate(self.chord_names)
        }
        self.name_to_id.update(chord_onset_to_token_id)
        self.chord_on_token_range = (
            self.chord_hold_token_range[1] + 1,
            len(self.name_to_id) - 1,
        )

        self.chord_token_range = (
            self.note_token_range[1] + 1,
            len(self.name_to_id) - 1,
        )

        self.id_to_name = {v: k for k, v in self.name_to_id.items()}

        self.onset_tokens = [
            self.name_to_id[f"CHORD_ON_{chord}"] for chord in self.chord_names
        ] + [
            self.name_to_id[f"NOTE_ON_{pitch}"]
            for pitch in range(self.num_pitches)
        ]

        self.hold_tokens = [
            self.name_to_id[f"CHORD_{chord}"] for chord in self.chord_names
        ] + [
            self.name_to_id[f"NOTE_{pitch}"]
            for pitch in range(self.num_pitches)
        ]

    @property
    def bos_token(self) -> int:
        """
        Get the BOS token ID.

        Returns:
            int: The BOS token ID.
        """
        return self.name_to_id["BOS"]

    @property
    def eos_token(self) -> int:
        """
        Get the EOS token ID.

        Returns:
            int: The EOS token ID.
        """
        return self.name_to_id["EOS"]

    @property
    def pad_token(self) -> int:
        """
        Get the PAD token ID.

        Returns:
            int: The PAD token ID.
        """
        return self.name_to_id["PAD"]

    @property
    def silence_token(self) -> int:
        """
        Get the SILENCE token ID.

        Returns:
            int: The SILENCE token ID.
        """
        return self.name_to_id["SILENCE"]

    @property
    def num_tokens(self) -> int:
        """
        Get the number of tokens.

        Returns:
            int: The number of tokens.
        """
        return len(self.name_to_id)

    def chord_on_id_to_chord_id(self, token_id: int) -> int:
        """
        Convert a chord onset token ID to the same chord ID.
        """
        chord_on_name = self.id_to_name[token_id]
        if "CHORD_ON_" in chord_on_name:
            chord_name = chord_on_name.replace("CHORD_ON_", "")
            return self.name_to_id[f"CHORD_{chord_name}"]
        else:
            raise ValueError(f"Invalid chord onset token ID: {token_id}")

    def chord_hold_id_to_chord_id(self, token_id: int) -> int:
        """
        Convert a chord hold token ID to the same chord ID.
        """
        chord_hold_name = self.id_to_name[token_id]
        if "CHORD_" in chord_hold_name:
            chord_name = chord_hold_name.replace("CHORD_", "")
            return self.name_to_id[f"CHORD_{chord_name}"]

    def note_on_id_to_note_id(self, token_id: int) -> int:
        """
        Convert a pitch onset token ID to the same pitch ID.
        """
        note_on_name = self.id_to_name[token_id]
        if "NOTE_ON_" in note_on_name:
            note_name = note_on_name.replace("NOTE_ON_", "")
            return self.name_to_id[f"NOTE_{note_name}"]
        else:
            raise ValueError(f"Invalid pitch onset token ID: {token_id}")

    def note_hold_id_to_note_id(self, token_id: int) -> int:
        """
        Convert a pitch hold token ID to the same pitch ID.
        """
        note_hold_name = self.id_to_name[token_id]
        if "NOTE_" in note_hold_name:
            note_name = note_hold_name.replace("NOTE_", "")
            return self.name_to_id[f"NOTE_{note_name}"]

    def is_chord_on(self, token_id: int) -> bool:
        """
        Check if a token ID is a chord onset token.
        """
        return "CHORD_ON_" in self.id_to_name[token_id]

    def is_chord_hold(self, token_id: int) -> bool:
        """
        Check if a token ID is a chord hold token.
        """
        return "CHORD_" in self.id_to_name[token_id] and not self.is_chord_on(
            token_id
        )

    def is_note_on(self, token_id: int) -> bool:
        """
        Check if a token ID is a pitch onset token.
        """
        return "NOTE_ON_" in self.id_to_name[token_id]

    def is_note_hold(self, token_id: int) -> bool:
        """
        Check if a token ID is a pitch hold token.
        """
        return "NOTE_" in self.id_to_name[token_id] and not self.is_note_on(
            token_id
        )

    def is_silence(self, token_id: int) -> bool:
        """
        Check if a token ID is a silence token.
        """
        return token_id == self.silence_token

    def is_onset(self, token_id: int) -> bool:
        """
        Check if a token ID is an onset token.
        """
        return token_id in self.onset_tokens

    def is_hold(self, token_id: int) -> bool:
        """
        Check if a token ID is a hold token.
        """
        return token_id in self.hold_tokens

    def melody_to_frames(
        self, melody: List[Dict[str, Any]], duration: int
    ) -> np.ndarray:
        """
        Convert melody to framewise sequences.

        Args:
            melody (List[Dict[str, Any]]): The melody annotations.
            duration (int): The duration in frames.

        Returns:
            np.ndarray: The framewise sequences.
        """
        frames = np.full(duration, self.name_to_id["SILENCE"], dtype=np.int32)
        for note in melody:
            start, end = note["onset"], note["offset"]
            pitch = to_midi_pitch(note["octave"], note["pitch_class"])
            start_frame = int(start * self.frame_per_beat)
            end_frame = int(end * self.frame_per_beat)
            if start_frame == end_frame:
                # If the note is shorter than a frame, extend it to the next frame
                end_frame += 1
            frames[start_frame:end_frame] = self.name_to_id[f"NOTE_{pitch}"]
            frames[start_frame] = self.name_to_id[f"NOTE_ON_{pitch}"]
        return frames

    def chord_to_frames(
        self, chord: List[Dict[str, Any]], duration: int
    ) -> np.ndarray:
        """
        Convert chord to framewise sequences.

        Args:
            chord (List[Dict[str, Any]]): The chord annotations
                (root_pitch_class + root_position_intervals).
            duration (int): The duration in frames.

        Returns:
            np.ndarray: The framewise sequences.
        """
        frames = np.full(duration, self.name_to_id["SILENCE"], dtype=np.int32)
        for c in chord:
            start, end = c["onset"], c["offset"]
            chord_name = to_chord_name(
                c["root_pitch_class"], c["root_position_intervals"]
            )
            start_frame = int(start * self.frame_per_beat)
            end_frame = int(end * self.frame_per_beat)
            if start_frame == end_frame:
                # If the note is shorter than a frame, extend it to the next frame
                end_frame += 1
            frames[start_frame:end_frame] = self.name_to_id[
                f"CHORD_{chord_name}"
            ]
            frames[start_frame] = self.name_to_id[f"CHORD_ON_{chord_name}"]
        return frames

    def chord_names_to_frames(
        self, chord: List[Dict[str, Any]], duration: int
    ) -> np.ndarray:
        """
        Convert chord names (from decode_chord_frames) to framewise sequences.

        Args:
            chord (List[Dict[str, Any]]): The chord names.
            duration (int): The duration in frames.

        Returns:
            np.ndarray: The framewise sequences.
        """
        frames = np.full(duration, self.name_to_id["SILENCE"], dtype=np.int32)
        for c in chord:
            start, end = c["onset"], c["offset"]
            chord_name = c["chord_name"]
            start_frame = int(start * self.frame_per_beat)
            end_frame = int(end * self.frame_per_beat)
            if start_frame == end_frame:
                # If the note is shorter than a frame, extend it to the next frame
                end_frame += 1
            frames[start_frame:end_frame] = self.name_to_id[
                f"CHORD_{chord_name}"
            ]
            frames[start_frame] = self.name_to_id[f"CHORD_ON_{chord_name}"]
        return frames

    def chord_to_names(self, chord: List[Dict[str, Any]]) -> List[str]:
        """
        Convert chord annotations to chord names.

        Args:
            chord (List[Dict[str, Any]]): The chord annotations.

        Returns:
            List[str]: The chord names.
        """
        chord_names = []
        for c in chord:
            pitch = to_chord_name(
                c["root_pitch_class"], c["root_position_intervals"]
            )
            chord_names.append(pitch)
        return chord_names

    def encode(self, example: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Convert a JSON to framewise sequences.

        Args:
            example (Dict[str, Any]): The example JSON.

        Returns:
            Dict[str, np.ndarray]: The framewise sequences.
        """
        melody = example["annotations"]["melody"]
        chord = example["annotations"]["harmony"]

        # Sort by onset time to ensure consistent processing
        melody = sorted(melody, key=lambda x: x["onset"])
        chord = sorted(chord, key=lambda x: x["onset"])

        melody_offset_last = melody[-1]["offset"]
        chord_offset_last = chord[-1]["offset"]
        melody_onset_last = melody[-1]["onset"]
        chord_onset_last = chord[-1]["onset"]
        duration = int(
            max(melody_offset_last, chord_offset_last) * self.frame_per_beat
        )
        if int(melody_offset_last * self.frame_per_beat) == int(
            melody_onset_last * self.frame_per_beat
        ):
            duration += 1
        elif int(chord_offset_last * self.frame_per_beat) == int(
            chord_onset_last * self.frame_per_beat
        ):
            duration += 1

        melody_frames = self.melody_to_frames(melody, duration)
        chord_frames = self.chord_to_frames(chord, duration)
        return {
            "melody": melody_frames,
            "chord": chord_frames,
        }

    def exclude_special_tokens(self, frames: np.ndarray) -> np.ndarray:
        """
        Exclude special tokens from frames.

        Args:
            frames (np.ndarray): The framewise sequences.

        Returns:
            np.ndarray: The framewise sequences without special tokens.
        """
        mask = (
            (frames != self.name_to_id["PAD"])
            & (frames != self.name_to_id["BOS"])
            & (frames != self.name_to_id["EOS"])
        )
        return frames[mask]

    def decode_chord_frames(
        self, chord_frames: np.ndarray
    ) -> List[Dict[str, Any]]:
        if isinstance(chord_frames, torch.Tensor):
            chord_frames = chord_frames.cpu().numpy()
        chord = []
        chord_frames = self.exclude_special_tokens(chord_frames)

        # Decode chord frames
        ongoing_chord = None
        ongoing_chord_name = None
        for i, frame in enumerate(chord_frames):
            frame_name = self.id_to_name[frame]
            if frame_name == "SILENCE":
                # if there's any ongoing chord, close it
                if ongoing_chord is not None:
                    ongoing_chord["offset"] = i / self.frame_per_beat
                    chord.append(ongoing_chord)
                    ongoing_chord = None
                continue

            # Check if it's a CHORD_ON event
            if "CHORD_ON_" in frame_name:
                if ongoing_chord is not None:
                    # Close the ongoing chord
                    ongoing_chord["offset"] = i / self.frame_per_beat
                    chord.append(ongoing_chord)
                # Start a new chord
                chord_name = frame_name.split("_")[2]
                ongoing_chord = {
                    "chord_name": chord_name,
                    "onset": i / self.frame_per_beat,
                }
                ongoing_chord_name = chord_name
            elif "CHORD_" in frame_name:
                chord_name = frame_name.split("_")[1]
                if ongoing_chord is None:
                    # We allow the first chord to be CHORD without CHORD_ON
                    ongoing_chord = {
                        "chord_name": chord_name,
                        "onset": i / self.frame_per_beat,
                    }
                    ongoing_chord_name = chord_name
                elif chord_name != ongoing_chord_name:
                    raise ValueError(
                        f"Chord off without chord on: {ongoing_chord_name} -> {chord_name}"
                    )
            else:
                raise ValueError(f"Invalid chord frame: {frame_name}")

        # If there's any ongoing chord at the end of the sequence, close it
        if ongoing_chord is not None:
            ongoing_chord["offset"] = len(chord_frames) / self.frame_per_beat
            chord.append(ongoing_chord)

        return chord

    def decode_melody_frames(
        self, melody_frames: np.ndarray
    ) -> List[Dict[str, Any]]:
        if isinstance(melody_frames, torch.Tensor):
            melody_frames = melody_frames.cpu().numpy()

        melody = []
        melody_frames = self.exclude_special_tokens(melody_frames)
        # Decode melody frames
        ongoing_note = None
        ongoing_pitch = None
        for i, frame in enumerate(melody_frames):
            frame_name = self.id_to_name[frame]
            if frame_name == "SILENCE":
                # if there's any ongoing note, close it
                if ongoing_note is not None:
                    ongoing_note["offset"] = i / self.frame_per_beat
                    melody.append(ongoing_note)
                    ongoing_note = None
                continue

            # Check if it's a NOTE_ON event
            if "NOTE_ON_" in frame_name:
                if ongoing_note is not None:
                    # Close the ongoing note
                    ongoing_note["offset"] = i / self.frame_per_beat
                    melody.append(ongoing_note)
                # Start a new note
                pitch = int(frame_name.split("_")[2])
                ongoing_note = {
                    "pitch_class": pitch % 12,
                    "octave": pitch // 12 - (ZERO_OCTAVE // 12),
                    "onset": i / self.frame_per_beat,
                }
                ongoing_pitch = pitch
            elif "NOTE_" in frame_name:
                pitch = int(frame_name.split("_")[1])
                if ongoing_note is None:
                    # We allow the first note to be NOTE without NOTE_ON
                    ongoing_note = {
                        "pitch_class": pitch % 12,
                        "octave": pitch // 12 - (ZERO_OCTAVE // 12),
                        "onset": i / self.frame_per_beat,
                    }
                    ongoing_pitch = pitch
                elif pitch != ongoing_pitch:
                    raise ValueError(
                        f"Note off without note on: {ongoing_pitch} -> {pitch}"
                    )
            else:
                raise ValueError(f"Invalid melody frame: {frame_name}")

        # If there's any ongoing note at the end of the sequence, close it
        if ongoing_note is not None:
            ongoing_note["offset"] = len(melody_frames) / self.frame_per_beat
            melody.append(ongoing_note)

        return melody

    def decode(
        self, chord_frames: np.ndarray, melody_frames: np.ndarray
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Decode framewise sequences into a JSON.

        Args:
            chord_frames (np.ndarray): The chord frames.
            melody_frames (np.ndarray): The melody frames.

        Returns:
            Dict[str, List[Dict[str, Any]]]: The decoded JSON.
        """

        melody = self.decode_melody_frames(melody_frames)
        chord = self.decode_chord_frames(chord_frames)
        return {"melody": melody, "harmony": chord}

    def decode_to_midi(
        self,
        chord_frames: Optional[np.ndarray] = None,
        melody_frames: Optional[np.ndarray] = None,
        bpm: int = 120,
    ) -> pretty_midi.PrettyMIDI:
        """
        Decode framewise sequences into a MIDI file.

        Args:
            chord_frames (np.ndarray): The chord frames.
            melody_frames (np.ndarray): The melody frames.
            bpm (int): The beats per minute.

        Returns:
            pretty_midi.PrettyMIDI: The generated MIDI file.
        """

        if chord_frames is None and melody_frames is None:
            raise ValueError(
                "At least one of chord_frames or melody_frames must be provided."
            )

        # Decode the frames into JSON format
        if chord_frames is not None:
            chord_annotations = self.decode_chord_frames(chord_frames)
        else:
            chord_annotations = []
        if melody_frames is not None:
            melody_annotations = self.decode_melody_frames(melody_frames)
        else:
            melody_annotations = []

        # Initialize PrettyMIDI object and instruments
        midi = pretty_midi.PrettyMIDI()
        # Default instrument for melody (Acoustic Grand Piano)
        melody_instr = pretty_midi.Instrument(program=0)
        # Default instrument for chords
        chord_instr = pretty_midi.Instrument(program=0)

        # Add melody notes to the MIDI
        for note in melody_annotations:
            # Convert note onset and offset to time in seconds
            onset_time = note["onset"] * 60 / bpm
            offset_time = note["offset"] * 60 / bpm
            pitch = to_midi_pitch(note["octave"], note["pitch_class"])

            # Create a PrettyMIDI Note object for the melody
            midi_note = pretty_midi.Note(
                velocity=MELODY_VELOCITY,
                pitch=pitch,
                start=onset_time,
                end=offset_time,
            )
            melody_instr.notes.append(midi_note)

        # Add chords to the MIDI
        for chord in chord_annotations:
            # Convert chord onset and offset to time in seconds
            onset_time = chord["onset"] * 60 / bpm
            offset_time = chord["offset"] * 60 / bpm
            chord_name = chord["chord_name"]

            # Get the chord pitches and bass using note_seq.chord_symbols_lib
            chord_pitches = chord_symbols_lib.chord_symbol_pitches(chord_name)
            chord_bass = chord_symbols_lib.chord_symbol_bass(chord_name)

            # Add chord notes to the chord instrument
            for pitch in chord_pitches:
                # # Avoid duplicate bass if already in chord_pitches
                # if pitch == chord_bass:
                #     continue
                midi_chord_note = pretty_midi.Note(
                    velocity=CHORD_VELOCITY,
                    pitch=pitch % 12 + CHORD_OCTAVE * 12,
                    start=onset_time,
                    end=offset_time,
                )
                chord_instr.notes.append(midi_chord_note)

            # Add the bass note
            # Bass note is one octave lower than the chord
            midi_bass_note = pretty_midi.Note(
                velocity=BASS_VELOCITY,
                pitch=chord_bass % 12 + BASS_OCTAVE * 12,
                start=onset_time,
                end=offset_time,
            )
            chord_instr.notes.append(midi_bass_note)

        # Add instruments to the PrettyMIDI object
        midi.instruments.append(melody_instr)
        midi.instruments.append(chord_instr)

        return midi

    def midi_to_json(self, midi: pretty_midi.PrettyMIDI) -> Dict[str, Any]:
        """
        Convert a MIDI file to a JSON.

        Args:
            midi (pretty_midi.PrettyMIDI): The MIDI file.

        Returns:
            Dict[str, Any]: The JSON representation.
        """
        pass
