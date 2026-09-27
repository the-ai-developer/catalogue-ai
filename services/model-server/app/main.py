"""Catalogue AI model-server (FastAPI) — contract: docs/api-contract.md §2.

Serves chunking, dual embeddings (SBERT + CLIP → shared space), FAISS search,
grounded answer composition with citation checks, and T5 beam-search drafts.
Heavy models load lazily; ``EMBEDDER_BACKEND=hash`` gives a deterministic
CPU-only mode for tests and smoke runs.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .answer_composer import AnswerComposer
from .chunker import Tokenizer, chunk_text
from .config import get_settings
from .embed import build_encoders
from .faiss_store import FaissStore
from .generator import DescriptionGenerator
from .projection import ProjectionMapper

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# request / response models (shapes == api-contract.md §2)
# --------------------------------------------------------------------------- #


class ChunkReq(BaseModel):
    item_id: Optional[str] = None
    kind: str = "copy"
    text: str
    max_tokens: int = 220
    overlap_tokens: int = 40


class EmbedTextReq(BaseModel):
    texts: List[str]
    model: Optional[str] = None


class ImagePayload(BaseModel):
    asset_id: Optional[str] = None
    path: Optional[str] = None
    b64: Optional[str] = None


class EmbedImageReq(BaseModel):
    images: List[ImagePayload]


class IndexEntry(BaseModel):
    faiss_id: int
    item_id: str
    modality: str
    chunk_id: Optional[str] = None
    asset_id: Optional[str] = None
    text: str = ""
    vector: List[float]


class UpsertReq(BaseModel):
    entries: List[IndexEntry]


class RemoveReq(BaseModel):
    faiss_ids: Optional[List[int]] = None
    item_ids: Optional[List[str]] = None


class SearchReq(BaseModel):
    query_text: Optional[str] = None
    query_image: Optional[ImagePayload] = None
    top_k: int = 6
    item_ids: Optional[List[str]] = None
    modality: str = "any"
    fused: bool = True


class ComposeContext(BaseModel):
    faiss_id: Optional[int] = None
    item_id: str
    sku: Optional[str] = None
    chunk_id: Optional[str] = None
    asset_id: Optional[str] = None
    modality: str = "text"
    score: float = 0.0
    text: str = ""


class ComposeReq(BaseModel):
    question: str
    mode: str = "extractive"
    contexts: List[ComposeContext]


class GenerateReq(BaseModel):
    spec: Dict[str, Any]
    beam_width: int = Field(4, ge=1, le=32)
    num_return_sequences: int = Field(3, ge=1, le=16)
    max_len: int = Field(192, ge=16, le=512)


# --------------------------------------------------------------------------- #
# runtime state (lazy)
# --------------------------------------------------------------------------- #


class Runtime:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.encoders = None
        self.projection = None
        self.store = None
        self.generator = None
        self.tokenizer = None

    def init(self) -> None:
        s = self.settings
        self.encoders = build_encoders(s)
        self.projection = ProjectionMapper.load(s.projection_path, s.embed_dim)
        self.store = FaissStore(s.embed_dim, s.faiss_index_dir)
        self.store.load()  # no-op when nothing persisted yet
        self.generator = DescriptionGenerator(s.desc_model_dir, s.model_device, s.model_seed)
        self.tokenizer = Tokenizer(None)  # fast local tokenizer; chunking is count-based
        log.info("model-server ready: %s | projection=%s | index=%s",
                 self.encoders.models(), self.projection.info(), self.store.counts())


RT = Runtime()
app = FastAPI(title="catalogue-ai model-server", version="1.0.0")


@app.on_event("startup")
def _startup() -> None:
    RT.init()


def _img_bytes(img: ImagePayload) -> bytes:
    if img.b64:
        return base64.b64decode(img.b64.split(",", 1)[-1])
    if img.path:
        path = img.path
        if not os.path.isabs(path):
            path = os.path.join(RT.settings.asset_root, path)
        with open(path, "rb") as fh:
            return fh.read()
    raise HTTPException(400, "image payload needs 'path' or 'b64'")


# --------------------------------------------------------------------------- #
# platform
# --------------------------------------------------------------------------- #


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/readyz")
def readyz():
    ok = RT.store is not None and RT.encoders is not None
    return {"ok": ok, "index": RT.store.counts() if RT.store else None}


@app.get("/v1/models")
def models():
    s = RT.settings
    return {
        "models": RT.encoders.models() if RT.encoders else {},
        "text_dim": RT.encoders.text.dim if RT.encoders else None,
        "image_dim": RT.encoders.image.dim if RT.encoders else None,
        "embed_dim": s.embed_dim,
        "projection": RT.projection.info() if RT.projection else None,
        "generator": {"model": RT.generator.model_name,
                      "version": RT.generator.model_version} if RT.generator else None,
    }


# --------------------------------------------------------------------------- #
# chunk + embed
# --------------------------------------------------------------------------- #


@app.post("/v1/chunk")
def chunk(req: ChunkReq):
    chunks = chunk_text(req.text, kind=req.kind, max_tokens=req.max_tokens,
                        overlap_tokens=req.overlap_tokens, tokenizer=RT.tokenizer)
    return {"chunks": [c.to_dict() for c in chunks]}


def _embed_text(texts: List[str]) -> np.ndarray:
    raw = RT.encoders.text.encode(texts)
    return RT.projection(raw)


def _embed_image(images: List[bytes]) -> np.ndarray:
    raw = RT.encoders.image.encode(images)
    return RT.projection(raw)


@app.post("/v1/embed/text")
def embed_text(req: EmbedTextReq):
    if not req.texts:
        raise HTTPException(400, "texts must not be empty")
    vecs = _embed_text(req.texts)
    return {"model": RT.encoders.text.model_name,
            "model_version": RT.encoders.text.model_version,
            "dim": int(vecs.shape[1]), "vectors": vecs.tolist()}


@app.post("/v1/embed/image")
def embed_image(req: EmbedImageReq):
    if not req.images:
        raise HTTPException(400, "images must not be empty")
    vecs = _embed_image([_img_bytes(i) for i in req.images])
    return {"model": RT.encoders.image.model_name,
            "model_version": RT.encoders.image.model_version,
            "dim": int(vecs.shape[1]), "vectors": vecs.tolist()}


# --------------------------------------------------------------------------- #
# index
# --------------------------------------------------------------------------- #


@app.post("/v1/index/upsert")
def index_upsert(req: UpsertReq):
    n = RT.store.upsert([e.model_dump() for e in req.entries])
    return {"upserted": n}


@app.post("/v1/index/remove")
def index_remove(req: RemoveReq):
    n = RT.store.remove(req.faiss_ids, req.item_ids)
    return {"removed": n}


@app.post("/v1/index/save")
def index_save():
    return {"saved": RT.store.save()}


@app.post("/v1/index/load")
def index_load():
    RT.store.load()
    return {"index": RT.store.counts()}


# --------------------------------------------------------------------------- #
# search
# --------------------------------------------------------------------------- #


@app.post("/v1/search")
def search(req: SearchReq):
    if not req.query_text and not req.query_image:
        raise HTTPException(400, "provide query_text and/or query_image")
    s = RT.settings
    tvec = _embed_text([req.query_text])[0] if req.query_text else None
    ivec = (_embed_image([_img_bytes(req.query_image)])[0]
            if req.query_image else None)
    hits = RT.store.search(
        text_vector=tvec, image_vector=ivec, top_k=req.top_k,
        item_ids=req.item_ids, modality=req.modality, fused=req.fused,
        text_weight=s.text_index_weight, image_weight=s.image_index_weight)
    return {"hits": hits, "models": RT.encoders.models()}


# --------------------------------------------------------------------------- #
# grounded composition
# --------------------------------------------------------------------------- #


@app.post("/v1/answer/compose")
def answer_compose(req: ComposeReq):
    s = RT.settings
    composer = AnswerComposer(s.citation_support_threshold,
                              embedder=lambda texts: _embed_text(list(texts)))
    contexts = [c.model_dump() for c in req.contexts]
    generator = None
    if req.mode == "abstractive":

        def generator(question, ctxs):  # noqa: ANN001
            joined = " ".join(c.get("text", "") for c in ctxs if c.get("text"))[:800]
            spec = {"title": question, "category": "answer", "features": [joined]}
            drafts = RT.generator.generate(spec, beam_width=4,
                                           num_return_sequences=1, max_len=160)
            return drafts["drafts"][0]["text"] if drafts["drafts"] else question

    return composer.compose(req.question, contexts, req.mode, generator)


# --------------------------------------------------------------------------- #
# generation (Project 2)
# --------------------------------------------------------------------------- #


@app.post("/v1/generate/description")
def generate_description(req: GenerateReq):
    result = RT.generator.generate(req.spec, beam_width=req.beam_width,
                                   num_return_sequences=req.num_return_sequences,
                                   max_len=req.max_len)
    return result
