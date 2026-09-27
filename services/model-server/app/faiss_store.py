"""FAISS vector store: two per-modality indexes over one shared space.

* ``text``  — SBERT-side chunk vectors
* ``image`` — CLIP-side product-photo vectors

Both are ``IndexIDMap2(IndexFlatIP)`` (inner product == cosine on the
L2-normalised shared space).  ``faiss_id`` is allocated by the Go API and equals
``embeddings.id``, so upserts are idempotent across re-ingests.  A sidecar JSON
maps ids back to catalogue metadata; FAISS files are disposable — Postgres plus
assets can rebuild everything (see docs/operations.md).
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .embed import l2_normalise

try:
    import faiss  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError("faiss is required: pip install faiss-cpu") from exc

META_FILE = "meta.json"
_INDEX_FILES = {"text": "text.index", "image": "image.index"}


class FaissStore:
    """Thread-safe dual FAISS index with metadata sidecar."""

    def __init__(self, dim: int, index_dir: str = ""):
        self.dim = dim
        self.index_dir = index_dir
        self._lock = threading.RLock()
        self._index = {m: faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
                       for m in ("text", "image")}
        self._ids: Dict[str, set] = {"text": set(), "image": set()}
        self._meta: Dict[int, Dict[str, Any]] = {}
        self._item_index: Dict[str, set] = {}

    @staticmethod
    def _remove_from(index, ids: List[int]) -> None:
        """Remove ids, tolerating faiss python API differences."""
        if not ids:
            return
        arr = np.asarray(ids, dtype=np.int64)
        try:
            index.remove_ids(arr)
        except Exception:
            index.remove_ids(faiss.IDSelectorBatch(arr))

    # ------------------------------- writes --------------------------- #
    def upsert(self, entries: Iterable[Dict[str, Any]]) -> int:
        """Insert or replace vectors.  Entry: faiss_id, item_id, modality,
        chunk_id?, asset_id?, text?, vector."""
        by_mod: Dict[str, List[tuple]] = {"text": [], "image": []}
        with self._lock:
            for e in entries:
                mod = e["modality"]
                fid = int(e["faiss_id"])
                vec = l2_normalise(np.asarray(e["vector"], np.float32)[None, :])
                if fid in self._ids[mod]:
                    self._remove_from(self._index[mod], [fid])
                by_mod[mod].append((fid, vec, e))
            n = 0
            for mod, rows in by_mod.items():
                if not rows:
                    continue
                ids = np.asarray([r[0] for r in rows], np.int64)
                mat = np.concatenate([r[1] for r in rows], axis=0)
                self._index[mod].add_with_ids(mat, ids)
                self._ids[mod].update(int(r[0]) for r in rows)
                for fid, _, e in rows:
                    item = str(e["item_id"])
                    self._meta[fid] = {
                        "faiss_id": fid, "item_id": item, "modality": mod,
                        "chunk_id": e.get("chunk_id"), "asset_id": e.get("asset_id"),
                        "text": e.get("text", ""),
                    }
                    self._item_index.setdefault(item, set()).add(fid)
                    n += 1
            return n

    def remove(self, faiss_ids: Optional[List[int]] = None,
               item_ids: Optional[List[str]] = None) -> int:
        with self._lock:
            ids: List[int] = list(faiss_ids or [])
            for item in item_ids or []:
                ids.extend(self._item_index.get(str(item), set()))
            ids = sorted(set(int(i) for i in ids))
            for fid in ids:
                mod = self._meta.get(fid, {}).get("modality")
                if mod and fid in self._ids[mod]:
                    self._remove_from(self._index[mod], [fid])
                    self._ids[mod].discard(fid)
                item = self._meta.pop(fid, {}).get("item_id")
                if item and item in self._item_index:
                    self._item_index[item].discard(fid)
                    if not self._item_index[item]:
                        del self._item_index[item]
            return len(ids)

    # ------------------------------- reads ---------------------------- #
    def _search_one(self, mod: str, query: np.ndarray, k: int,
                    item_ids: Optional[List[str]]):
        index = self._index[mod]
        if index.ntotal == 0:
            return []
        fetch = k if not item_ids else min(index.ntotal, max(k * 20, 200))
        scores, ids = index.search(np.asarray(query, np.float32)[None, :], fetch)
        hits = []
        for score, fid in zip(scores[0], ids[0]):
            if fid == -1:
                continue
            meta = self._meta.get(int(fid))
            if meta is None:
                continue
            if item_ids and meta["item_id"] not in {str(i) for i in item_ids}:
                continue
            hits.append({**meta, "score": float(score)})
            if len(hits) >= k:
                break
        return hits

    def search(self, *, text_vector: Optional[np.ndarray] = None,
               image_vector: Optional[np.ndarray] = None, top_k: int = 6,
               item_ids: Optional[List[str]] = None,
               modality: str = "any", fused: bool = True,
               text_weight: float = 0.6, image_weight: float = 0.4) -> List[dict]:
        """Top-k over one or both indexes.

        Fusion: when both query vectors are present and ``fused`` is true, the
        union of per-modality top-k is rescored as
        ``text_weight*s_text + image_weight*s_image`` (missing side scores 0).
        A single-modality query uses its raw cosine scores.
        """
        with self._lock:
            per_mod: Dict[str, List[dict]] = {}
            want = {"text": text_vector is not None and modality in ("any", "text"),
                    "image": image_vector is not None and modality in ("any", "image")}
            if want["text"]:
                per_mod["text"] = self._search_one("text", text_vector, top_k, item_ids)
            if want["image"]:
                per_mod["image"] = self._search_one("image", image_vector, top_k, item_ids)

            if fused and len(per_mod) == 2:
                union: Dict[int, dict] = {}
                for rows in per_mod.values():
                    for r in rows:
                        union.setdefault(r["faiss_id"], {**r, "score": 0.0})
                t = {r["faiss_id"]: r["score"] for r in per_mod.get("text", [])}
                i = {r["faiss_id"]: r["score"] for r in per_mod.get("image", [])}
                for fid, row in union.items():
                    row["score"] = text_weight * t.get(fid, 0.0) + image_weight * i.get(fid, 0.0)
                hits = sorted(union.values(), key=lambda r: -r["score"])
            else:
                hits = [r for rows in per_mod.values() for r in rows]
                hits.sort(key=lambda r: -r["score"])
            return hits[:top_k]

    # ---------------------------- persistence ------------------------- #
    def save(self, index_dir: str = "") -> str:
        directory = index_dir or self.index_dir
        os.makedirs(directory, exist_ok=True)
        with self._lock:
            for mod, name in _INDEX_FILES.items():
                faiss.write_index(self._index[mod], os.path.join(directory, name))
            with open(os.path.join(directory, META_FILE), "w") as fh:
                json.dump({str(k): v for k, v in self._meta.items()}, fh)
        return directory

    def load(self, index_dir: str = "") -> None:
        directory = index_dir or self.index_dir
        with self._lock:
            for mod, name in _INDEX_FILES.items():
                path = os.path.join(directory, name)
                if os.path.exists(path):
                    self._index[mod] = faiss.read_index(path)
            meta_path = os.path.join(directory, META_FILE)
            if os.path.exists(meta_path):
                with open(meta_path) as fh:
                    raw = json.load(fh)
                self._meta = {int(k): v for k, v in raw.items()}
                self._item_index = {}
                self._ids = {"text": set(), "image": set()}
                for fid, meta in self._meta.items():
                    self._item_index.setdefault(meta["item_id"], set()).add(fid)
                    self._ids.get(meta.get("modality", "text"), set()).add(fid)

    def counts(self) -> dict:
        with self._lock:
            return {"text": self._index["text"].ntotal,
                    "image": self._index["image"].ntotal,
                    "meta": len(self._meta)}
