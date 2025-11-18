"""Evaluation utilities for ReaLchords."""

import numpy as np
import torch
import note_seq.chord_symbols_lib as chord_symbols_lib

from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer


def evaluate_note_in_chord_ratio(
    sequences: torch.Tensor,
    tokenizer: HooktheoryTokenizer,
    model_part: str,
    return_count: bool = False,
    sequence_order: str = "chord_first",
) -> torch.Tensor:
    """
    Compute the note in chord ratio for each sequence in decoder_preds.

    The function assumes that decoder_preds has tokens in an alternating order:
    even indices contain note tokens and odd indices contain chord tokens (or reverse).
    For each note-chord pair, frames where either token is a special token (PAD, BOS, EOS, SILENCE)
    are ignored. For valid pairs, the function checks if the note (modulo 12) is one of the chord's pitches.
    The ratio for each sequence is the average of valid time steps with a note-in-chord match.

    Args:
        decoder_preds (torch.Tensor): Tensor of shape [batch, seq_len] with predicted token ids.
        tokenizer: An instance of HooktheoryTokenizer providing the id_to_name mapping.
        model_part (str): Specifies which part of the model to evaluate ("melody" or "chord").
        return_count (bool): If True, returns a tuple with the ratio, valid counts, and correct counts.

        torch.Tensor: A list of note in chord ratios, one for each sequence in the batch.
    """
    if sequence_order not in {"chord_first", "melody_first"}:
        raise ValueError(
            f"Unsupported sequence_order '{sequence_order}'. Expected 'chord_first' or 'melody_first'."
        )

    special_tokens = {"PAD", "BOS", "EOS", "SILENCE"}

    ratios = []
    valid_counts = []
    correct_counts = []
    # Assume even indices are note tokens and odd indices are chord tokens.
    for seq in sequences:
        # Convert tensor tokens to Python ints.
        seq = seq.cpu().tolist()
        if sequence_order == "chord_first":
            chord_tokens = seq[::2]
            melody_tokens = seq[1::2]
        else:
            melody_tokens = seq[::2]
            chord_tokens = seq[1::2]

        if model_part not in {"melody", "chord"}:
            raise ValueError(f"Invalid model_part: {model_part}")

        # Regardless of which model generated the tokens, melody tokens correspond to notes
        # and chord tokens provide the harmonic context.
        note_tokens = melody_tokens

        valid_count = 0
        correct_count = 0

        for note_token, chord_token in zip(note_tokens, chord_tokens):
            # Get token names using tokenizer's id_to_name mapping.
            note_name = tokenizer.id_to_name.get(note_token, "")
            chord_name_full = tokenizer.id_to_name.get(chord_token, "")

            # Skip this pair if either token is a special token.
            if note_name in special_tokens or chord_name_full in special_tokens:
                continue

            # Only process note tokens that start with "NOTE" or "NOTE_ON"
            if not (
                note_name.startswith("NOTE_")
                or note_name.startswith("NOTE_ON_")
            ):
                continue

            # Only process chords that start with "CHORD_" or "CHORD_ON_"
            if chord_name_full.startswith("CHORD_ON_"):
                chord_str = chord_name_full[len("CHORD_ON_") :]
            elif chord_name_full.startswith("CHORD_"):
                chord_str = chord_name_full[len("CHORD_") :]
            else:
                continue

            # Get chord pitches using chord_symbols_lib.
            chord_pitches = chord_symbols_lib.chord_symbol_pitches(chord_str)
            # Extract note pitch from note token.
            if note_name.startswith("NOTE_ON_"):
                try:
                    note_pitch = int(note_name[len("NOTE_ON_") :])
                except ValueError:
                    continue
            elif note_name.startswith("NOTE_"):
                try:
                    note_pitch = int(note_name[len("NOTE_") :])
                except ValueError:
                    continue
            else:
                continue

            # Check if note (mod 12) is in the chord: compare reducing to pitch-class.
            if note_pitch % 12 in [p % 12 for p in chord_pitches]:
                correct_count += 1
            valid_count += 1

        ratio = correct_count / valid_count if valid_count > 0 else np.nan
        ratios.append(ratio)

        valid_counts.append(valid_count)
        correct_counts.append(correct_count)
    if return_count:
        return torch.tensor(ratios), torch.tensor(valid_counts), torch.tensor(correct_counts)
    else:
        return torch.tensor(ratios)


def evaluate_initial_silence(
    sequences: torch.Tensor, tokenizer: HooktheoryTokenizer
) -> torch.Tensor:
    """
    Compute the number initial silence tokens in each sequence in decoder_preds.

    The function assumes that decoder_preds has tokens in an alternating order:
    even indices contain model tokens and odd indices contain context tokens.
    For each sequence, the function counts the number of initial silence tokens.
    The count is the number of consecutive silence tokens at the beginning of the sequence.

    Args:
        decoder_preds (torch.Tensor): Tensor of shape [batch, seq_len] with predicted token ids.
        tokenizer: An instance of HooktheoryTokenizer providing the id_to_name mapping.

    Returns:
        torch.Tensor: A list of initial silence, one for each sequence in the batch.
    """
    silence_count_all = []
    # Assume even indices are note tokens and odd indices are chord tokens.
    for seq in sequences:
        # Convert tensor tokens to Python ints.
        seq = seq.cpu().tolist()
        model_part_tokens = seq[::2]

        silence_count = 0

        # Get token names using tokenizer's id_to_name mapping.
        for token in model_part_tokens:
            token_name = tokenizer.id_to_name.get(token, "")
            if token_name == "SILENCE":
                silence_count += 1
            else:
                break

        silence_count_all.append(silence_count)

    return torch.tensor(silence_count_all).float()


def evaluate_average_duration(
    sequences: torch.Tensor, tokenizer: HooktheoryTokenizer
) -> torch.Tensor:
    """
    Compute the average duration of the model part in decoder_preds.

    The function assumes that decoder_preds has tokens in an alternating order:
    even indices contain model tokens and odd indices contain context tokens.
    For each sequence, the function computes the average duration of the model part.

    Args:
        decoder_preds (torch.Tensor): Tensor of shape [batch, seq_len] with predicted token ids.
        tokenizer: An instance of HooktheoryTokenizer providing the id_to_name mapping.

    Returns:
        torch.Tensor: A list of average durations, one for each sequence in the batch.
    """
    special_tokens = {"PAD", "BOS", "EOS", "SILENCE"}

    durations = []
    # Assume even indices are note tokens and odd indices are chord tokens.
    for seq in sequences:
        # Convert tensor tokens to Python ints.
        seq = seq.cpu().tolist()
        model_part_tokens = seq[::2]

        num_onset = 0
        num_tokens = 0
        num_silence = 0

        # Get token names using tokenizer's id_to_name mapping.
        for token in model_part_tokens:
            token_name = tokenizer.id_to_name.get(token, "")
            if token_name in special_tokens:
                continue
            else:
                num_tokens += 1
                if token_name.startswith("NOTE_ON_") or token_name.startswith(
                    "CHORD_ON_"
                ):
                    num_onset += 1
                elif token_name == "SILENCE":
                    num_silence += 1

        # Compute the average duration of the model part.
        if num_onset > 0:
            duration = (num_tokens - num_silence) / num_onset
        else:
            duration = 0.0

        durations.append(duration)

    return torch.tensor(durations)
