"""Spec → description generation with beam search (Project 2, contract §2).

Production path: fine-tuned T5 (``DESC_MODEL_DIR``) via ``model.generate`` beam
search.  If no checkpoint / transformers is available, a deterministic template
generator keeps the pipeline alive (clearly flagged in ``model_version``).
Scores are mean token log-probabilities, so drafts are comparable across beams.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Dict, List, Optional, Sequence

from .linearise import linearise_spec

log = logging.getLogger(__name__)


class DescriptionGenerator:
    """Beam-search spec→text generator with a safe template fallback."""

    def __init__(self, desc_model_dir: str = "", device: str = "cpu",
                 seed: Optional[int] = None):
        self.desc_model_dir = desc_model_dir
        self.device = device
        self.seed = seed
        self._model = None
        self._tok = None
        self.model_name = "t5-small-desc"
        self.model_version = "uninitialised"

    # ------------------------------------------------------------------ #
    def _load(self) -> bool:
        if self._model is not None:
            return True
        try:
            import torch  # lazy
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # lazy

            source = self.desc_model_dir if self.desc_model_dir and os.path.exists(
                self.desc_model_dir) else "t5-small"
            self._tok = AutoTokenizer.from_pretrained(source)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(source).to(
                self.device).eval()
            fine_tuned = source == self.desc_model_dir
            self.model_version = f"{source}:{'fine-tuned' if fine_tuned else 'base-not-fine-tuned'}"
            self._torch = torch
            return True
        except Exception as exc:
            log.warning("T5 unavailable (%s); template fallback in use", exc)
            self.model_version = "template-fallback"
            return False

    # ------------------------------------------------------------------ #
    @staticmethod
    def _template_drafts(spec: Dict, num: int) -> List[dict]:
        """Deterministic, spec-faithful drafts so the pipeline works anywhere."""
        name = spec.get("title") or f"{spec.get('category', 'item')}"
        material = spec.get("material") or ""
        feats: Sequence[str] = spec.get("features") or []
        dims = spec.get("dimensions") or {}
        dim_text = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in sorted(dims.items()))
        variants = [
            "{name}. {mat_clause}{dim_clause} {feat_clause}.",
            "Meet the {name}: {feat_clause_lower}{mat_clause}{dim_clause}.",
            "{name} — {feat_clause_lower}{dim_clause}{mat_clause}.",
        ]
        drafts = []
        for i in range(min(num, len(variants))):
            text = variants[i].format(
                name=name,
                mat_clause=f"Made from {material}. " if material else "",
                dim_clause=f"Dimensions: {dim_text}. " if dim_text else "",
                feat_clause="Features " + ", ".join(feats) if feats else
                "Built for everyday use",
                feat_clause_lower=("featuring " + ", ".join(feats) + ". ") if feats
                else "built for everyday use. ",
            )
            drafts.append({"rank": i + 1, "text": " ".join(text.split()),
                           "score": round(-0.4 - 0.1 * i, 4)})
        return drafts

    # ------------------------------------------------------------------ #
    def generate(self, spec: Dict, *, beam_width: int = 4,
                 num_return_sequences: int = 3, max_len: int = 192) -> dict:
        """Return ranked drafts ``[{rank, text, score}]`` + provenance."""
        num = max(1, min(num_return_sequences, beam_width))
        if not self._load():
            return {"model": self.model_name, "model_version": self.model_version,
                    "drafts": self._template_drafts(spec, num), "latency_ms": 0}

        import time

        if self.seed is not None:
            self._torch.manual_seed(self.seed)
        prompt = linearise_spec(spec)
        t0 = time.monotonic()
        inputs = self._tok(prompt, return_tensors="pt", truncation=True,
                           max_length=512).to(self.device)
        with self._torch.no_grad():
            out = self._model.generate(
                **inputs,
                num_beams=max(beam_width, num),
                num_return_sequences=num,
                max_new_tokens=max_len,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                return_dict_in_generate=True,
                output_scores=True,
            )
        seqs = out.sequences
        # mean token logprob per sequence (skip pad/eos)
        scores = self._torch.stack(out.scores, dim=1)  # [steps, vocab] -> [1, steps, vocab]
        logprobs = self._torch.log_softmax(scores, dim=-1)
        drafts = []
        for i in range(seqs.shape[0]):
            seq = seqs[i]
            tokens = seq.tolist()
            total, n = 0.0, 0
            for step, tok_id in enumerate(tokens):
                if step >= logprobs.shape[1]:
                    break
                if tok_id in (self._tok.pad_token_id, self._tok.eos_token_id):
                    continue
                total += float(logprobs[0, step, tok_id])
                n += 1
            text = self._tok.decode(seq, skip_special_tokens=True).strip()
            drafts.append({"rank": i + 1, "text": text,
                           "score": round(total / max(1, n), 4)})
        latency = int((time.monotonic() - t0) * 1000)
        return {"model": self.model_name, "model_version": self.model_version,
                "drafts": drafts, "latency_ms": latency}
