"""Shared-space projection (contract: "Shared-space mapping").

SBERT and CLIP raw outputs live in different spaces.  Both are L2-normalised and
then mapped by a bias-free linear map into one ``EMBED_DIM``-dimensional space
and re-normalised, so a single inner product compares text and image evidence.
Weights are trained with InfoNCE on item-level (text, image) pairs — see
``ml/training/train_projection.py``.  When no weights are present the mapper is
the identity (dimensions are padded/truncated to ``embed_dim``) and a warning is
logged, so the pipeline still runs before the projection is trained.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

from .embed import l2_normalise

log = logging.getLogger(__name__)


class ProjectionMapper:
    """Bias-free linear map + re-normalise into the shared space."""

    def __init__(self, embed_dim: int, weight: Optional[np.ndarray] = None,
                 path: str = "", trained: bool = False):
        self.embed_dim = embed_dim
        self.path = path
        self.trained = trained
        self.weight = np.asarray(weight, dtype=np.float32) if weight is not None else None

    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, path: str, embed_dim: int) -> "ProjectionMapper":
        """Load ``projection.pt`` (torch) or ``projection.npz`` (numpy).

        Torch checkpoints are optional at runtime: the matrix is converted to
        numpy immediately so serving needs no torch afterwards.
        """
        npz = path.rsplit(".", 1)[0] + ".npz"
        if os.path.exists(npz):
            data = np.load(npz)
            return cls(embed_dim, data["weight"], path=path, trained=True)
        if os.path.exists(path):
            try:
                import torch  # lazy

                state = torch.load(path, map_location="cpu")
                weight = state["weight"] if isinstance(state, dict) else state
                return cls(embed_dim, weight.detach().cpu().numpy(), path=path,
                           trained=True)
            except Exception as exc:
                log.warning("could not load projection %s: %s", path, exc)
        log.warning("projection weights missing (%s); using identity mapping — "
                    "run ml/training/train_projection.py for a trained shared space",
                    path)
        return cls(embed_dim, None, path=path, trained=False)

    def save(self, path: str) -> None:
        """Persist weights as ``.npz`` (and ``.pt`` when torch is available)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(path.rsplit(".", 1)[0] + ".npz", weight=self.weight)
        try:
            import torch  # lazy

            torch.save({"weight": torch.from_numpy(self.weight)}, path)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    def _adapt(self, x: np.ndarray) -> np.ndarray:
        """Pad/truncate raw vectors to the mapper's input dimension.

        Target width is the learned matrix's input dim when weights exist,
        otherwise ``embed_dim`` (identity mapping).
        """
        target = self.weight.shape[1] if self.weight is not None else self.embed_dim
        if x.shape[-1] == target:
            return x
        out = np.zeros(x.shape[:-1] + (target,), np.float32)
        n = min(x.shape[-1], target)
        out[..., :n] = x[..., :n]
        return out

    def __call__(self, vectors: np.ndarray) -> np.ndarray:
        x = np.asarray(vectors, dtype=np.float32)
        if x.ndim == 1:
            return self.__call__(x[None, :])[0]
        x = self._adapt(x)
        if self.weight is not None:
            x = x @ self.weight.T
        return l2_normalise(x)

    def info(self) -> dict:
        shape = None if self.weight is None else list(self.weight.shape)
        return {"trained": self.trained, "embed_dim": self.embed_dim,
                "weight_shape": shape, "path": self.path}
