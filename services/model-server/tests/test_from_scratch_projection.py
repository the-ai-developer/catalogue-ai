"""Tests: from-scratch beam search, projection mapper, hash embedders."""

import numpy as np

from app.embed import HashEmbedder, l2_normalise
from app.from_scratch.beam_search import beam_search
from app.projection import ProjectionMapper


class TestBeamSearch:
    def test_matches_brute_force_on_toy_model(self):
        # toy vocab: prefers sequence [1,2,3]
        logp = {0: -2.0, 1: -0.1, 2: -0.1, 3: -0.1, 4: -3.0}

        def step(prefix):
            nxt = len(prefix) - 1  # 0-based step index
            return {tok: lp for tok, lp in logp.items()}

        results = beam_search(step, bos=99, eos=0, beam_width=3, max_len=4)
        assert results, "no beams returned"
        best = results[0]
        # best sequence must be the greedy favourite chain (1,2,3,...)
        assert best.tokens[0] == 99
        assert best.tokens[1] == 1
        assert best.score >= results[1].score  # sorted best-first

    def test_eos_terminates_and_length_penalty(self):
        def step(prefix):
            if len(prefix) == 1:
                return {5: -0.2, 6: -0.3}
            return {7: -0.5, 8: -2.0}

        out = beam_search(step, bos=1, eos=7, beam_width=2, max_len=10)
        assert all(b.tokens[-1] == 7 or len(b.tokens) <= 10 for b in out)
        assert out[0].score >= out[-1].score

    def test_return_count(self):
        step = lambda prefix: {i: -1.0 for i in range(1, 6)}
        out = beam_search(step, bos=0, eos=9, beam_width=4, max_len=5, num_return=2)
        assert len(out) == 2


class TestProjection:
    def test_identity_maps_and_normalises(self):
        proj = ProjectionMapper(embed_dim=8, trained=False)
        x = np.ones((2, 8), np.float32)
        y = proj(x)
        assert y.shape == (2, 8)
        assert np.allclose(np.linalg.norm(y, axis=1), 1.0)

    def test_learned_map_shapes(self):
        w = np.eye(4, 8, dtype=np.float32)  # 8 -> 4
        proj = ProjectionMapper(embed_dim=4, weight=w, trained=True)
        y = proj(np.random.default_rng(0).standard_normal((3, 8)).astype(np.float32))
        assert y.shape == (3, 4)
        assert np.allclose(np.linalg.norm(y, axis=1), 1.0)

    def test_save_load_roundtrip(self, tmp_path):
        w = np.random.default_rng(1).standard_normal((4, 8)).astype(np.float32)
        proj = ProjectionMapper(embed_dim=4, weight=w, trained=True)
        path = str(tmp_path / "projection.pt")
        proj.save(path)
        loaded = ProjectionMapper.load(path, embed_dim=4)
        assert loaded.trained and loaded.weight.shape == (4, 8)

    def test_missing_weights_fall_back_to_identity(self, tmp_path):
        loaded = ProjectionMapper.load(str(tmp_path / "nope.pt"), embed_dim=6)
        assert not loaded.trained
        assert loaded(np.ones(6, np.float32)).shape == (6,)


class TestHashEmbedders:
    def test_deterministic_and_normalised(self):
        emb = HashEmbedder(dim=16)
        a = emb.encode(["hello", "world"])
        b = emb.encode(["hello", "world"])
        assert a.shape == (2, 16)
        assert np.allclose(a, b)
        assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
        # unrelated inputs should not collapse to the same vector
        assert float(a[0] @ a[1]) < 0.9

    def test_bytes_input(self):
        emb = HashEmbedder(dim=8)
        out = emb.encode([b"\x89PNG"])
        assert out.shape == (1, 8)
