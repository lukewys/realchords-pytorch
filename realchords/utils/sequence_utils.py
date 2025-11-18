"""Utility functions for sequences."""

import torch
import torch.nn.functional as F
from typing import Tuple


def pad_and_get_mask(
    sequence: torch.Tensor, max_len: int, pad_value: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad the sequence and get the mask.

    Args:
        sequence (torch.Tensor): Sequence (shape: [seq_len]).
        max_len (int): Maximum length.
        pad_value (int, optional): Value to pad with. Defaults to 0.

    Returns:
        tuple: Padded sequence and mask.
    """
    mask = torch.ones((max_len,), dtype=torch.bool)
    if sequence.size(0) < max_len:
        mask[-(max_len - sequence.size(0)) :] = 0
    sequence = F.pad(sequence, (0, max_len - sequence.size(0)), value=pad_value)
    return sequence, mask


def add_bos_to_sequence(
    sequence: torch.Tensor, bos_token_id: int
) -> torch.Tensor:
    return torch.cat(
        [
            torch.full(
                (sequence.size(0), 1),
                bos_token_id,
                device=sequence.device,
            ),
            sequence,
        ],
        dim=1,
    )


def add_eos_to_sequence(
    sequence: torch.Tensor,
    pad_token_id: int,
    eos_token_id: int,
    pad_end: bool = True,
) -> torch.Tensor:
    if pad_end:
        # First, add a padding token to the end of the sequence.
        sequence = torch.cat(
            [
                sequence,
                torch.full(
                    (sequence.size(0), 1),
                    pad_token_id,
                    device=sequence.device,
                ),
            ],
            dim=1,
        )

    # Identify the position of the first padding token and replace it with the EOS token.
    pad_pos = torch.zeros_like(sequence)
    pad_pos[sequence == pad_token_id] = 1
    first_pad_idx = pad_pos.argmax(dim=1)
    sequence[torch.arange(sequence.size(0)), first_pad_idx] = eos_token_id

    return sequence


def create_table_from_mapping(
    mapping: dict[int | tuple[int], int],
    default: int = -1,
    dtype: torch.dtype = torch.int32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create a table from a mapping."""
    max_val = max(v if isinstance(v, int) else max(v) for v in mapping)
    table = torch.full((max_val + 1,), default, dtype=dtype, device=device)
    for keys, value in mapping.items():
        keys = (keys,) if isinstance(keys, int) else keys
        table[torch.tensor(keys, device=device)] = value
    return table


def remap(
    x: torch.Tensor, mapping: dict[int | tuple[int], int], default: int = -1
) -> torch.Tensor:
    """Remap the values in the tensor using a mapping.

    Args:
        x (torch.Tensor): Tensor to remap.
        mapping (dict[int | tuple[int], int]): Mapping of values to remap.
        default (int, optional): Default value. Defaults to -1.

    Returns:
        torch.Tensor: Remapped tensor.

    Example:
        >>> x = torch.tensor([1, 2, 3])
        >>> mapping = {1: 10, 2: 20, 3: 30}
        >>> remap(x, mapping)
        tensor([10, 20, 30])

        >>> x = torch.tensor([1, 2, 3])
        >>> mapping = {(1, 2): 10, 3: 20}
        >>> remap(x, mapping)
        tensor([10, 10, 20])
    """

    table = create_table_from_mapping(mapping, default, x.dtype, x.device)
    return table[x]


def remap_from_table(x: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """Remap the values in the tensor using a table."""
    table = table.to(x.device)
    return table[x]


@torch.no_grad()
def sequences_order_to_counterpart(
    sequences: torch.Tensor, pad_token_id: int = 0
) -> torch.Tensor:
    # Assume first token is always the BOS token
    if sequences.shape[1] == 1:
        return sequences

    sequences_counterpart = torch.zeros_like(sequences)
    sequences_counterpart[:, 1:][:, ::2] = sequences[:, 1:][:, 1::2]
    sequences_counterpart[:, 1:][:, 1::2] = sequences[:, 1:][:, ::2]
    sequences_counterpart[:, 0] = sequences[:, 0]
    sequences_counterpart[sequences_counterpart == 2] = pad_token_id
    return sequences_counterpart


def log_probs_from_logits(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    # https://github.com/OpenRLHF/OpenRLHF/pull/718#issuecomment-2641081881
    if logits.dtype in [torch.float32, torch.float64]:
        logits_labels = torch.gather(
            logits, dim=-1, index=labels.unsqueeze(-1)
        ).squeeze(-1)
        logsumexp_values = torch.stack(
            [
                torch.logsumexp(l, dim=-1) for l in logits
            ]  # loop to reduce peak mem consumption
        )
        log_probs_labels = (
            logits_labels - logsumexp_values
        )  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        log_probs_labels = []
        for row_logits, row_labels in zip(
            logits, labels
        ):  # loop to reduce peak mem consumption
            row_log_probs = F.log_softmax(row_logits, dim=-1)
            row_log_probs_labels = row_log_probs.gather(
                dim=-1, index=row_labels.unsqueeze(-1)
            ).squeeze(-1)
            log_probs_labels.append(row_log_probs_labels)
        log_probs_labels = torch.stack(log_probs_labels)
    return log_probs_labels


@torch.no_grad()
def log_probs_from_online_model(
    model: torch.nn.Module,
    sequences: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = model(sequences, mask=None)
    log_probs = log_probs_from_logits(logits[:, :-1, :], sequences[:, 1:])
    logits = logits[:, :-1, :]  # remove the last token logits
    return log_probs, logits


@torch.no_grad()
def get_seperated_parts_from_sequence(
    sequence: torch.Tensor,
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # remove bos
    sequence = sequence[:, 1:].clone()
    model_tokens = sequence[:, ::2]
    context_tokens = sequence[:, 1::2]

    # add eos to the first padding token
    model_tokens = add_eos_to_sequence(model_tokens, pad_token_id, eos_token_id)
    context_tokens = add_eos_to_sequence(
        context_tokens, pad_token_id, eos_token_id
    )

    # add bos to the beginning of the sequence
    model_tokens = add_bos_to_sequence(model_tokens, bos_token_id)
    context_tokens = add_bos_to_sequence(context_tokens, bos_token_id)

    model_mask = model_tokens != pad_token_id
    context_mask = context_tokens != pad_token_id
    return context_tokens, model_tokens, context_mask, model_mask
