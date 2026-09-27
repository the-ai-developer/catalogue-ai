"""From-scratch ML building blocks (used by ml/notebooks/ and tests).

* ``beam_search``  — dependency-free beam search over a step-logprob function
* ``attention``    — multi-head attention + encoder/decoder blocks (torch)
* ``contrastive``  — InfoNCE loss + shared-space training step (torch)

These mirror the production components mathematically so the notebooks can
demonstrate the algorithms without library magic.
"""

from .beam_search import beam_search, BeamResult  # noqa: F401
