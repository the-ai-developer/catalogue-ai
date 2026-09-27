"""Padding and mask helpers for variable-length seq2seq batches.

``torch.tensor([[3, 7, 9], [4, 5]])` raises ``ValueError: expected sequence of
length 3 at dim 1`` — encoders and decoders consume fixed-width tensors, so a
batch of unequal token sequences must be padded and masked before it reaches
the model.  This module builds those batches and the boolean masks that
``MultiHeadAttention`` consumes (``masked_fill(~mask, -inf)``).

Every mask returned here keeps at least one visible key per query row, so a
softmax over a fully-masked row — which would produce ``NaN`` — cannot occur.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

try:
    import torch
except ImportError:  # pragma: no cover - torch optional at import time
    torch = None


def _require_torch():
    if torch is None:
        raise ImportError("torch is required for app.from_scratch.padding")


def pad_batch(sequences: Sequence[Sequence[int]], pad_id: int = 0,
              max_len: Optional[int] = None) -> "torch.Tensor":
    """Right-pad token ids into a ``[B, T]`` LongTensor plus a mask.

    Args:
        sequences: one list of token ids per row.
        pad_id: id written into the padding positions.
        max_len: truncate/pad to this width.  Defaults to the longest row, so
            nothing is added beyond what the batch needs.

    Returns:
        ``(padded, mask)`` where ``padded`` is ``[B, T]`` and ``mask`` is a
        ``[B, T]`` bool tensor that is ``True`` on real tokens.
    """
    _require_torch()
    rows = [list(s) for s in sequences]
    if not rows:
        raise ValueError("pad_batch requires at least one sequence")
    width = max_len or max((len(r) for r in rows), default=0)
    if width == 0:
        raise ValueError("cannot pad a batch of empty sequences")

    padded = torch.full((len(rows), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.bool)
    for i, row in enumerate(rows):
        keep = min(len(row), width)
        if keep:
            padded[i, :keep] = torch.tensor(row[:keep], dtype=torch.long)
            mask[i, :keep] = True
    return padded, mask


def causal_key_mask(key_mask: "torch.Tensor", size: Optional[int] = None
                    ) -> "torch.Tensor":
    """Combine a padding mask with a causal mask for decoder self-attention.

    Args:
        key_mask: ``[B, T]`` bool mask that is ``True`` on real tokens.
        size: query width; defaults to the key width.

    Returns:
        ``[B, Tq, Tk]`` bool tensor that is ``True`` where attention is allowed.
        The mask is **per row**: a short row never attends to padding that
        belongs to a longer row in the same batch, so a padded batch produces
        exactly the same output as running each row on its own.
    """
    _require_torch()
    _, tk = key_mask.shape
    tq = size or tk
    causal = torch.tril(torch.ones(tq, tk, dtype=torch.bool,
                                   device=key_mask.device))
    visible = causal[None, :, :] & key_mask[:, None, :]
    # A row that is entirely padding would softmax over all -inf and yield NaN,
    # which then propagates through the backward pass even when the loss ignores
    # those targets. Reveal the first key for such rows to keep it finite.
    empty = ~visible.any(dim=-1, keepdim=True)
    first_key = torch.zeros_like(visible)
    first_key[..., 0] = True
    return visible | (empty & first_key)


def encoder_mask(key_mask: "torch.Tensor") -> "torch.Tensor":
    """Broadcast a ``[B, S]`` padding mask to a ``[B, 1, 1, S]`` key mask."""
    _require_torch()
    return key_mask[:, None, None, :]


def batch_size(batch: "torch.Tensor") -> int:
    return int(batch.shape[0])


def to_device(batch: "torch.Tensor", device) -> "torch.Tensor":
    """Move a padded batch (and any mask tuple) to ``device``."""
    if torch is None:
        raise ImportError("torch is required for app.from_scratch.padding")
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, (tuple, list)):
        return type(batch)(to_device(b, device) for b in batch)
    return batch


__all__: List[str] = ["pad_batch", "causal_key_mask", "encoder_mask",
                      "to_device", "batch_size"]
