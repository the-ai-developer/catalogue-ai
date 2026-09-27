"""Hand-written beam search (pure Python — no ML libraries).

Given a step function ``step(prefix_tokens) -> {token_id: logprob}``, return the
top-``k`` sequences by length-penalised score, exactly like a seq2seq decoder's
beam search.  Used to validate the production T5 beam search and in notebooks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


@dataclass
class BeamResult:
    tokens: Tuple[int, ...]
    score: float            # length-penalised
    raw_logprob: float      # sum of token logprobs
    length: int


@dataclass(order=True)
class _Beam:
    score: float
    raw: float = field(compare=False)
    tokens: Tuple[int, ...] = field(compare=False)
    done: bool = field(compare=False, default=False)


def _length_penalty(length: int, alpha: float) -> float:
    # Google NMT length penalty (alpha=1.0 keeps it mild, as in the T5 serving config)
    return ((5.0 + length) / 6.0) ** alpha


def beam_search(step: Callable[[Tuple[int, ...]], Dict[int, float]],
                *, bos: int, eos: int, beam_width: int = 4,
                max_len: int = 32, length_penalty_alpha: float = 1.0,
                num_return: Optional[int] = None) -> List[BeamResult]:
    """Decode with beam search.

    ``step`` maps a partial token tuple to ``{token_id: logprob}`` for the next
    token (it must be deterministic for tests).  Finished beams (ended with
    ``eos``) are collected and returned best-first, up to ``num_return``.
    """
    beams = [_Beam(score=0.0, raw=0.0, tokens=(bos,), done=False)]
    finished: List[_Beam] = []

    for _ in range(max_len):
        candidates: List[_Beam] = []
        for beam in beams:
            if beam.done:
                continue
            for token_id, logp in step(beam.tokens).items():
                raw = beam.raw + float(logp)
                tokens = beam.tokens + (token_id,)
                length = len(tokens) - 1  # exclude bos
                score = raw / _length_penalty(max(1, length), length_penalty_alpha)
                candidates.append(_Beam(score=score, raw=raw, tokens=tokens,
                                        done=token_id == eos))
        if not candidates:
            break
        candidates.sort(reverse=True)
        beams = []
        for cand in candidates:
            (finished if cand.done else beams).append(cand)
            if len(beams) >= beam_width and len(finished) >= beam_width:
                break
        if not beams:
            break

    finished.sort(reverse=True)
    if not finished:  # never emitted eos: return best live beams
        finished = sorted(beams, reverse=True)
    out = [BeamResult(tokens=b.tokens, score=b.score, raw_logprob=b.raw,
                      length=max(0, len(b.tokens) - 1))
           for b in finished[: num_return or beam_width]]
    return out
