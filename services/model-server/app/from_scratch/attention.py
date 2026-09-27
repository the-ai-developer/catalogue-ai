"""From-scratch multi-head attention + encoder/decoder blocks (torch).

Mirrors the T5-style architecture the production model uses, small enough to
train on a notebook in minutes.  Torch is imported lazily-friendly so importing
this module never requires GPU.
"""

from __future__ import annotations

import math
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - torch optional at import time
    torch = nn = F = None


def _require_torch():
    if torch is None:
        raise ImportError("torch is required for app.from_scratch.attention")


class MultiHeadAttention(nn.Module):
    """Scaled dot-product multi-head attention with optional causal masking."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        _require_torch()
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_q, x_kv=None, mask: Optional[torch.Tensor] = None,
                return_weights: bool = False):
        x_kv = x_q if x_kv is None else x_kv
        B, Tq, _ = x_q.shape
        Tk = x_kv.shape[1]

        def split(t):  # [B,T,D] -> [B,H,T,Dh]
            return t.view(B, -1, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = split(self.q(x_q)), split(self.k(x_kv)), split(self.v(x_kv))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            # scores is [B, H, Tq, Tk]; a mask may be given as [Tk], [Tq, Tk],
            # [B, Tq, Tk] or [B, 1, Tk] — left-pad with singleton dims so the
            # broadcast aligns against the head axis rather than the batch.
            m = mask
            if m.dim() == 2:
                m = m[None, None, :, :]
            elif m.dim() == 3:
                m = m.unsqueeze(1)
            scores = scores.masked_fill(~m, float("-inf"))
        weights = self.dropout(torch.softmax(scores, dim=-1))
        out = (weights @ v).transpose(1, 2).contiguous().view(B, Tq, -1)
        out = self.out(out)
        return (out, weights) if return_weights else out


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        _require_torch()
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(d_ff, d_model))

    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        _require_torch()
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.self_attn(x, mask=mask))
        return self.norm2(x + self.ff(x))


class DecoderBlock(nn.Module):
    """Masked self-attention + cross-attention over encoder states."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        _require_torch()
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None,
                return_weights: bool = False):
        """Decode one step (or a full teacher-forced prefix).

        ``tgt_mask`` masks decoder self-attention (causal and/or padding);
        ``memory_mask`` masks cross-attention over ``memory`` so padded encoder
        positions are not attended to.
        """
        x = self.norm1(x + self.self_attn(x, mask=tgt_mask))
        attn_out, weights = self.cross_attn(x_kv=memory, x_q=x, mask=memory_mask,
                                            return_weights=True)
        x = self.norm2(x + attn_out)
        x = self.norm3(x + self.ff(x))
        return (x, weights) if return_weights else x


def causal_mask(size: int, device=None):
    """Lower-triangular mask so position t only attends to <= t."""
    _require_torch()
    return torch.tril(torch.ones(size, size, dtype=torch.bool, device=device))
