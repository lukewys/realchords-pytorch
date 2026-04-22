"""Data perturbation utilities for sequence generation experiments."""

import torch
from note_seq import chord_symbols_lib
from rapidfuzz import process, fuzz

from realchords.dataset.hooktheory_tokenizer import HooktheoryTokenizer


def postprocess_chord_name(chord_name: str) -> str:
    """Post-process chord name to handle note_seq.chord_symbols_lib bugs."""
    if "Cb" in chord_name:
        chord_name = chord_name.replace("Cb", "B")
    elif "B#" in chord_name:
        chord_name = chord_name.replace("B#", "C")
    elif "Fb" in chord_name:
        chord_name = chord_name.replace("Fb", "E")
    elif "E#" in chord_name:
        chord_name = chord_name.replace("E#", "F")
    return chord_name


def transpose_chord_name(
    original_chord_name: str, transpose_amount: int, chord_names: list
) -> str:
    """Transpose a chord name by a given number of semitones."""
    transposed_chord_name = chord_symbols_lib.transpose_chord_symbol(
        original_chord_name, transpose_amount
    )
    transposed_chord_name = postprocess_chord_name(transposed_chord_name)
    original_transposed_chord_name = transposed_chord_name

    if transposed_chord_name not in chord_names:
        transposed_chord_pitches = chord_symbols_lib.chord_symbol_pitches(
            transposed_chord_name
        )
        transposed_chord_name = chord_symbols_lib.pitches_to_chord_symbol(
            transposed_chord_pitches
        )
        if transposed_chord_name not in chord_names:
            best_match, _, _ = process.extractOne(
                transposed_chord_name,
                chord_names,
                scorer=fuzz.WRatio,
            )
            transposed_chord_name = best_match
            print(
                f"Warning: {original_transposed_chord_name} is not in the chord_names list, so we use {transposed_chord_name} instead."
            )
            if transposed_chord_name not in chord_names:
                raise ValueError(
                    f"No match found for {original_chord_name} and {transposed_chord_name}"
                )

    return transposed_chord_name


def transpose_chord(
    sequence: torch.Tensor,
    transpose_amount: int,
    tokenizer: HooktheoryTokenizer,
) -> torch.Tensor:
    """Transpose chord tokens in a sequence by a given number of semitones."""
    if transpose_amount == 0:
        return sequence

    sequence_transposed = sequence.clone()
    chord_names = tokenizer.chord_names

    for i in range(sequence.shape[0]):
        for j in range(sequence.shape[1]):
            token_name = tokenizer.id_to_name[sequence[i, j].item()]
            if token_name in ["BOS", "EOS", "PAD", "SILENCE"]:
                pass
            elif "NOTE" in token_name:
                pass
            elif "CHORD" in token_name:
                if "CHORD_ON" in token_name:
                    original_chord_name = token_name.replace("CHORD_ON_", "")
                    transposed_chord_name = transpose_chord_name(
                        original_chord_name, transpose_amount, chord_names
                    )
                    new_token_name = f"CHORD_ON_{transposed_chord_name}"
                    sequence_transposed[i, j] = tokenizer.name_to_id[
                        new_token_name
                    ]
                else:
                    original_chord_name = token_name.replace("CHORD_", "")
                    transposed_chord_name = transpose_chord_name(
                        original_chord_name, transpose_amount, chord_names
                    )
                    new_token_name = f"CHORD_{transposed_chord_name}"
                    sequence_transposed[i, j] = tokenizer.name_to_id[
                        new_token_name
                    ]
            else:
                raise ValueError(f"Unknown token name: {token_name}")

    return sequence_transposed


def transpose_note(
    sequence: torch.Tensor,
    transpose_amount: int,
    tokenizer: HooktheoryTokenizer,
) -> torch.Tensor:
    """Transpose note tokens in a sequence by a given number of semitones."""
    if transpose_amount == 0:
        return sequence

    sequence_transposed = sequence.clone()

    for i in range(sequence.shape[0]):
        for j in range(sequence.shape[1]):
            token_name = tokenizer.id_to_name[sequence[i, j].item()]
            if token_name in ["BOS", "EOS", "PAD", "SILENCE"]:
                pass
            elif "CHORD" in token_name:
                pass
            elif "NOTE" in token_name:
                if "NOTE_ON" in token_name:
                    pitch = int(token_name.replace("NOTE_ON_", ""))
                    new_note_name = f"NOTE_ON_{pitch + transpose_amount}"
                    sequence_transposed[i, j] = tokenizer.name_to_id[
                        new_note_name
                    ]
                else:
                    pitch = int(token_name.replace("NOTE_", ""))
                    new_note_name = f"NOTE_{pitch + transpose_amount}"
                    sequence_transposed[i, j] = tokenizer.name_to_id[
                        new_note_name
                    ]
            else:
                raise ValueError(f"Unknown token name: {token_name}")

    return sequence_transposed


def apply_multiple_transpose_chord(
    sequences: torch.Tensor, tokenizer: HooktheoryTokenizer
) -> torch.Tensor:
    """Apply the multi-segment transposition used in the perturbation experiments."""
    targets_transposed = sequences.clone()

    if sequences.shape[1] > 64:
        end_64_128 = min(128, sequences.shape[1])
        targets_transposed[:, 64:end_64_128] = transpose_chord(
            sequences[:, 64:end_64_128], 6, tokenizer
        )

    if sequences.shape[1] > 128:
        end_128_192 = min(192, sequences.shape[1])
        targets_transposed[:, 128:end_128_192] = transpose_chord(
            sequences[:, 128:end_128_192], 4, tokenizer
        )

    if sequences.shape[1] > 192:
        end_192_256 = min(256, sequences.shape[1])
        targets_transposed[:, 192:end_192_256] = transpose_chord(
            sequences[:, 192:end_192_256], -10, tokenizer
        )

    return targets_transposed


def apply_multiple_transpose_note(
    sequences: torch.Tensor, tokenizer: HooktheoryTokenizer
) -> torch.Tensor:
    """Apply the multi-segment transposition used in the perturbation experiments."""
    targets_transposed = sequences.clone()

    if sequences.shape[1] > 64:
        end_64_128 = min(128, sequences.shape[1])
        targets_transposed[:, 64:end_64_128] = transpose_note(
            sequences[:, 64:end_64_128], 6, tokenizer
        )

    if sequences.shape[1] > 128:
        end_128_192 = min(192, sequences.shape[1])
        targets_transposed[:, 128:end_128_192] = transpose_note(
            sequences[:, 128:end_128_192], 4, tokenizer
        )

    if sequences.shape[1] > 192:
        end_192_256 = min(256, sequences.shape[1])
        targets_transposed[:, 192:end_192_256] = transpose_note(
            sequences[:, 192:end_192_256], -10, tokenizer
        )

    return targets_transposed


def apply_single_transpose_6_chord(
    sequences: torch.Tensor, tokenizer: HooktheoryTokenizer
) -> torch.Tensor:
    """Transpose the suffix of chord sequences by +6 semitones."""
    targets_transposed = sequences.clone()

    if sequences.shape[1] > 128:
        targets_transposed[:, 128:] = transpose_chord(
            sequences[:, 128:], 6, tokenizer
        )

    return targets_transposed


def apply_single_transpose_6_note(
    sequences: torch.Tensor, tokenizer: HooktheoryTokenizer
) -> torch.Tensor:
    """Transpose the suffix of note sequences by +6 semitones."""
    targets_transposed = sequences.clone()

    if sequences.shape[1] > 128:
        targets_transposed[:, 128:] = transpose_note(
            sequences[:, 128:], 6, tokenizer
        )

    return targets_transposed


def apply_data_perturbation(
    sequences: torch.Tensor,
    perturbation_type: str,
    data_type: str,
    tokenizer: HooktheoryTokenizer,
) -> torch.Tensor:
    """Apply a supported perturbation to melody or chord sequences."""
    if perturbation_type == "none":
        return sequences

    if data_type not in ["melody", "chord"]:
        raise ValueError(
            f"data_type must be 'melody' or 'chord', got {data_type}"
        )

    if perturbation_type == "multiple_transpose":
        if data_type == "chord":
            return apply_multiple_transpose_chord(sequences, tokenizer)
        return apply_multiple_transpose_note(sequences, tokenizer)

    if perturbation_type == "single_transpose_6":
        if data_type == "chord":
            return apply_single_transpose_6_chord(sequences, tokenizer)
        return apply_single_transpose_6_note(sequences, tokenizer)

    raise ValueError(f"Invalid perturbation_type: {perturbation_type}")


def validate_perturbation_args(mode: str, data_perturbation: str) -> None:
    """Validate that perturbation arguments are compatible with the mode."""
    mode_parts = mode.split("_vs_")
    if len(mode_parts) != 2:
        raise ValueError(f"Invalid mode format: {mode}")

    if data_perturbation != "none":
        is_model_vs_data = "data" in mode_parts[0] or "data" in mode_parts[1]
        is_data_vs_data = "data" in mode_parts[0] and "data" in mode_parts[1]

        if not is_model_vs_data:
            raise ValueError(
                f"Data perturbation can only be used with modes containing 'data', but got mode: {mode}"
            )

        if is_data_vs_data and mode != "melody_data_vs_chord_data":
            raise ValueError(
                "Data perturbation is only supported for 'melody_data_vs_chord_data' in data vs data modes, "
                f"but got mode: {mode}"
            )
