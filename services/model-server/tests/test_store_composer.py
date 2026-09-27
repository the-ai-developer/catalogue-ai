"""Tests: FAISS store behaviour + grounded answer composer (CPU, no downloads)."""

import numpy as np
import pytest

from app.answer_composer import NOT_GROUNDED, AnswerComposer
from app.embed import HashEmbedder, l2_normalise
from app.faiss_store import FaissStore


def _vec(seed: int, dim: int = 32) -> list:
    rng = np.random.default_rng(seed)
    return l2_normalise(rng.standard_normal(dim)).tolist()


@pytest.fixture()
def store(tmp_path):
    return FaissStore(dim=32, index_dir=str(tmp_path))


class TestFaissStore:
    def test_upsert_search_remove(self, store):
        store.upsert([
            {"faiss_id": 1, "item_id": "a", "modality": "text",
             "chunk_id": "c1", "text": "dishwasher safe", "vector": _vec(1)},
            {"faiss_id": 2, "item_id": "b", "modality": "image",
             "asset_id": "i1", "text": "", "vector": _vec(2)},
        ])
        assert store.counts() == {"text": 1, "image": 1, "meta": 2}

        hits = store.search(text_vector=np.asarray(_vec(1)), top_k=2)
        assert hits and hits[0]["faiss_id"] == 1
        assert hits[0]["score"] > 0.99  # identical vector -> cosine ~ 1

        # upsert same id replaces
        store.upsert([{"faiss_id": 1, "item_id": "a", "modality": "text",
                       "chunk_id": "c1", "text": "updated", "vector": _vec(7)}])
        assert store.counts()["text"] == 1

        store.remove(item_ids=["a"])
        assert store.counts() == {"text": 0, "image": 1, "meta": 1}

    def test_fusion_prefers_both_matches(self, store):
        store.upsert([
            {"faiss_id": 1, "item_id": "a", "modality": "text",
             "text": "x", "vector": _vec(1)},
            {"faiss_id": 2, "item_id": "b", "modality": "text",
             "text": "y", "vector": _vec(2)},
            {"faiss_id": 3, "item_id": "a", "modality": "image",
             "text": "", "vector": _vec(1)},  # same direction as text hit
        ])
        hits = store.search(text_vector=np.asarray(_vec(1)),
                            image_vector=np.asarray(_vec(1)), top_k=3, fused=True)
        assert hits[0]["item_id"] == "a"  # matched in both modalities

    def test_save_load_roundtrip(self, store):
        store.upsert([{"faiss_id": 9, "item_id": "z", "modality": "text",
                       "text": "hello", "vector": _vec(3)}])
        store.save()
        fresh = FaissStore(32, store.index_dir)
        fresh.load()
        hits = fresh.search(text_vector=np.asarray(_vec(3)), top_k=1)
        assert hits[0]["faiss_id"] == 9 and hits[0]["text"] == "hello"

    def test_item_filter(self, store):
        store.upsert([
            {"faiss_id": 1, "item_id": "a", "modality": "text",
             "text": "x", "vector": _vec(1)},
            {"faiss_id": 2, "item_id": "b", "modality": "text",
             "text": "y", "vector": _vec(1)},
        ])
        hits = store.search(text_vector=np.asarray(_vec(1)), top_k=5,
                            item_ids=["b"])
        assert [h["item_id"] for h in hits] == ["b"]


class TestAnswerComposer:
    def _ctxs(self):
        return [
            {"item_id": "a", "sku": "KX-1042", "modality": "text", "score": 0.9,
             "text": "The Kestrel bottle holds 12 oz and is dishwasher safe."},
            {"item_id": "a", "sku": "KX-1042", "modality": "image", "score": 0.7,
             "text": ""},
        ]

    def test_extractive_is_grounded_with_citations(self):
        comp = AnswerComposer(0.35)
        out = comp.compose("Is the Kestrel bottle dishwasher safe?", self._ctxs())
        assert out["citation_check"]["passed"] is True
        assert out["answer"] != NOT_GROUNDED
        assert out["sentences"] and all(s["citations"] for s in out["sentences"])
        assert out["sentences"][0]["citations"][0]["context_index"] == 0

    def test_no_evidence_refuses_to_answer(self):
        comp = AnswerComposer(0.35)
        out = comp.compose("What is the warranty on the lunar rover?",
                           [{"item_id": "a", "modality": "text", "score": 0.2,
                             "text": "The bottle is made of stainless steel."}])
        assert out["citation_check"]["passed"] is False
        assert out["answer"] == NOT_GROUNDED

    def test_abstractive_strips_unsupported_sentences(self):
        comp = AnswerComposer(0.35)
        out = comp.compose(
            "Is it dishwasher safe?", self._ctxs(), mode="abstractive",
            generator=lambda q, c: "The Kestrel bottle is dishwasher safe. "
                                   "It is also made of solid gold.")
        assert out["citation_check"]["passed"] is False  # gold claim stripped
        assert "gold" not in out["answer"]
        assert "dishwasher" in out["answer"]
        stripped = [d for d in out["citation_check"]["details"] if d.get("stripped")]
        assert len(stripped) == 1

    def test_empty_contexts(self):
        out = AnswerComposer(0.35).compose("anything?", [])
        assert out["answer"] == NOT_GROUNDED and not out["citation_check"]["passed"]

    def test_visual_question_cites_images(self):
        comp = AnswerComposer(0.35)
        out = comp.compose("What does the bottle look like?", self._ctxs())
        image_cited = any(c["context_index"] == 1
                          for s in out["sentences"] for c in s["citations"])
        assert image_cited
