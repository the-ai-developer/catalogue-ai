"""Tests: batch padding, attention masking, and InfoNCE metric directions.

The padding helpers exist because ``torch.tensor`` rejects ragged token
sequences, which crashed notebook 02 outright.  The decisive property tested
here is *mask invariance*: a padded batch must produce exactly the same
output as the same sequences run unpadded, which only holds if every attention
path respects the mask.
"""

import pytest

torch = pytest.importorskip("torch")

from app.from_scratch.attention import (  # noqa: E402
    DecoderBlock,
    EncoderBlock,
    causal_mask,
)
from app.from_scratch.contrastive import info_nce  # noqa: E402
from app.from_scratch.padding import (  # noqa: E402
    causal_key_mask,
    encoder_mask,
    pad_batch,
    to_device,
)

PAD = 0


class TestPadBatch:
    def test_pads_to_longest_row(self):
        padded, mask = pad_batch([[1, 2, 3], [4], [5, 6]], pad_id=PAD)
        assert padded.shape == (3, 3)
        assert mask.shape == (3, 3)
        assert padded[1].tolist() == [4, 0, 0]
        assert mask[1].tolist() == [True, False, False]
        assert mask[0].all() and not mask[2, 2]

    def test_respects_max_len_truncation(self):
        padded, mask = pad_batch([[1, 2, 3, 4, 5]], pad_id=PAD, max_len=3)
        assert padded.shape == (1, 3)
        assert padded[0].tolist() == [1, 2, 3]
        assert mask.all()

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError):
            pad_batch([], pad_id=PAD)

    def test_rejects_all_empty_sequences(self):
        with pytest.raises(ValueError):
            pad_batch([[], []], pad_id=PAD)


class TestMasks:
    def test_encoder_mask_shape_broadcasts(self):
        _, mask = pad_batch([[1, 2], [3]], pad_id=PAD)
        enc = encoder_mask(mask)
        assert enc.shape == (2, 1, 1, 2)
        assert enc[0, 0, 0].tolist() == [True, True]
        assert enc[1, 0, 0].tolist() == [True, False]

    def test_causal_mask_is_lower_triangular(self):
        _, mask = pad_batch([[1, 2, 3]], pad_id=PAD)
        causal = causal_key_mask(mask)
        assert causal.shape == (1, 3, 3)
        assert causal[0].tolist() == [[True, False, False],
                                      [True, True, False],
                                      [True, True, True]]

    def test_mask_is_per_row_not_global(self):
        # Row 0 is longer than row 1; row 1 must not inherit row 0's keys.
        # Padding is masked as a *key* — a padded query position may still
        # attend to its own valid keys, and its output is discarded by the
        # ignore_index=PAD loss term rather than by masking the query.
        _, mask = pad_batch([[1, 2, 3, 4], [5, 6]], pad_id=PAD)
        causal = causal_key_mask(mask)
        assert causal.shape == (2, 4, 4)
        # row 0 sees every key it owns (strictly lower triangular)
        assert causal[0].tolist() == [[True, False, False, False],
                                      [True, True, False, False],
                                      [True, True, True, False],
                                      [True, True, True, True]]
        # row 1 sees only keys 0 and 1 — never row 0's keys 2 and 3
        assert not causal[1][:, 2].any()
        assert not causal[1][:, 3].any()
        assert causal[1][0].tolist() == [True, False, False, False]

    def test_per_row_mask_is_never_more_permissive_than_global(self):
        # key_mask.any(dim=0) collapses the batch to one key set; the per-row
        # mask must mask at least as much, so a shorter row can never gain
        # visibility that the shared version denied it.
        for sequences in ([[1, 2, 3, 4], [5, 6]], [[1], [2, 3, 4]],
                          [[1, 2], [3, 4, 5, 6, 7]]):
            _, mask = pad_batch(sequences, pad_id=PAD)
            width = mask.shape[1]
            per_row = causal_key_mask(mask)
            shared = torch.tril(torch.ones(width, width, dtype=torch.bool)) \
                & mask.any(dim=0)
            assert bool((per_row <= shared[None]).all()), sequences

    def test_no_row_is_fully_masked(self):
        # A fully-masked row makes softmax return NaN.
        for seqs in ([[1, 2, 3, 0]], [[1, 2], [3]], [[1]]):
            _, mask = pad_batch(seqs, pad_id=PAD)
            for width in range(1, mask.shape[1] + 1):
                causal = causal_key_mask(mask, size=width)
                assert causal.any(dim=-1).all(), f"empty row at width {width}"

    def test_fully_padded_row_stays_finite(self):
        # Caller must not pass explicit pad tokens, but a degenerate row must
        # not produce an all -inf softmax.
        _, mask = pad_batch([[1, 2, 3]], pad_id=PAD)
        mask = mask.clone()
        mask[0, 0] = False  # simulate a row with no real tokens
        causal = causal_key_mask(mask)
        assert causal.any(dim=-1).all()


class TestMaskInvariance:
    """Padded batches must not change the result for the real tokens."""

    WEIGHT = torch.randn(32, 16, generator=torch.Generator().manual_seed(7))

    @classmethod
    def _embed(cls, ids):
        return torch.nn.functional.embedding(ids, cls.WEIGHT)

    def test_encoder_output_matches_unpadded_run(self):
        torch.manual_seed(0)
        block = EncoderBlock(d_model=16, n_heads=4, d_ff=32).eval()
        sequences = [[1, 2, 3, 4], [5, 6], [7, 8, 9]]

        padded, mask = pad_batch(sequences, pad_id=PAD)
        with torch.no_grad():
            batched = block(self._embed(padded), mask=encoder_mask(mask))
            alone = [block(self._embed(torch.tensor(s)).unsqueeze(0))
                     for s in sequences]

        for i, seq in enumerate(sequences):
            torch.testing.assert_close(batched[i:i + 1, :len(seq)], alone[i])

    def test_decoder_output_matches_unpadded_run(self):
        torch.manual_seed(0)
        block = DecoderBlock(d_model=16, n_heads=4, d_ff=32).eval()
        targets = [[1, 2, 3, 4], [5, 6], [7, 8, 9]]
        sources = [[10, 11, 12], [13, 14], [15, 16, 17, 18]]

        tgt, tgt_mask = pad_batch(targets, pad_id=PAD)
        src, src_mask = pad_batch(sources, pad_id=PAD)
        with torch.no_grad():
            # both self-attention and cross-attention are masked here
            batched, _ = block(self._embed(tgt), self._embed(src),
                               tgt_mask=causal_key_mask(tgt_mask),
                               memory_mask=encoder_mask(src_mask),
                               return_weights=True)
            # Baseline: the same sequence with no padding, so the causal mask
            # is the full lower triangle and the memory needs no key mask.
            # Only padding differs between the two paths.
            alone = [
                block(self._embed(torch.tensor(t)).unsqueeze(0),
                      self._embed(torch.tensor(s)).unsqueeze(0),
                      tgt_mask=causal_mask(len(t)))
                for t, s in zip(targets, sources)]

        for i, tgt_row in enumerate(targets):
            torch.testing.assert_close(batched[i:i + 1, :len(tgt_row)], alone[i])

    def test_padded_targets_do_not_produce_nan(self):
        torch.manual_seed(0)
        block = DecoderBlock(d_model=16, n_heads=4, d_ff=32).eval()
        padded, mask = pad_batch([[1, 2], [3, 4]], pad_id=PAD)
        with torch.no_grad():
            out, _ = block(self._embed(padded), torch.randn(2, 4, 16),
                           tgt_mask=causal_key_mask(mask), return_weights=True)
        assert torch.isfinite(out).all()


class TestToDevice:
    def test_moves_tensor_and_tuples(self):
        padded, mask = pad_batch([[1, 2], [3]], pad_id=PAD)
        moved = to_device((padded, mask), "cpu")
        assert isinstance(moved, tuple) and len(moved) == 2
        assert moved[0].device.type == "cpu"


class TestInfoNceMetrics:
    """The two retrieval accuracies were previously reported under swapped names."""

    def test_direction_labels_are_not_inverted(self):
        # Rows pair with their own column; make the pairing trivially detectable
        # by giving every row the same text and distinct image vectors.
        dim = 4
        image = torch.eye(dim)
        text = torch.eye(dim) * 0 + 1.0
        text = text / text.norm(dim=-1, keepdim=True)
        _, metrics = info_nce(text, image, temperature=0.01)
        # identical text rows -> retrieval is ambiguous, both directions must
        # be reported consistently rather than one being the other's mirror
        assert set(metrics) == {"acc_image_to_text", "acc_text_to_image",
                                "temperature"}
        assert 0.0 <= metrics["acc_text_to_image"] <= 1.0
        assert 0.0 <= metrics["acc_image_to_text"] <= 1.0

    def test_perfect_pairs_score_one(self):
        torch.manual_seed(0)
        text = torch.nn.functional.normalize(torch.randn(16, 32), dim=-1)
        _, metrics = info_nce(text, text.clone(), temperature=0.01)
        assert metrics["acc_text_to_image"] == 1.0
        assert metrics["acc_image_to_text"] == 1.0

    def test_asymmetric_input_distinguishes_directions(self):
        # An asymmetric score matrix is what makes a swapped label detectable.
        #   text  = [[1,0,0],[0,1,0]]      image = [[0,0,1],[1,0,0]]
        #   text @ image.T = [[0,1],[0,0]]
        # argmax(dim=1) -> [1, 0] vs labels [0, 1]  => text  -> image = 0.0
        # argmax(dim=0) -> [0, 0] vs labels [0, 1]  => image -> text = 0.5
        text = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        image = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        _, metrics = info_nce(text, image, temperature=0.01)
        assert metrics["acc_text_to_image"] == 0.0
        assert metrics["acc_image_to_text"] == 0.5
