r"""Comprehensive CPU unit tests for ViDiCRF v2 & v1 MBR decoder.

Run: .\.venv\Scripts\python.exe -X utf8 -m unittest test_vidicrf_v2 -v
"""
from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "charexp"))

import sylmodel_punct as sm_v1
from vidicrf_v2.orthography import (
    compact_inventory,
    is_valid_vn_or_clean_candidate,
    normalize_tone_placement,
    parse_syllable_phonology,
    strip_diac,
)
from vidicrf_v2.v1_mbr_decoder import (
    constrained_viterbi,
    forward_backward_marginals,
    mbr_decode,
)
from vidicrf_v2.model import (
    ViDiCRFv2,
    compute_loss,
    trigram_gold_score,
    trigram_log_partition,
    trigram_marginals,
    trigram_viterbi,
)
from vidicrf_v2.data_pipeline import (
    build_v2_inventory,
    encode_corpus_v2,
    make_batch_v2,
)
from vidicrf_v2.restorer import FastRestorerV2


def _brute_force_linear_crf(emit_1, trans_1, mask_1, real_len):
    K = emit_1.shape[1]
    valid_slots = [
        [k for k in range(K) if bool(mask_1[t, k])]
        for t in range(real_len)
    ]
    scores = []
    paths = list(itertools.product(*valid_slots))
    for path in paths:
        s = sum(float(emit_1[t, path[t]]) for t in range(real_len))
        s += sum(float(trans_1[t, path[t], path[t + 1]]) for t in range(real_len - 1))
        scores.append(s)
    t_scores = torch.tensor(scores, dtype=torch.float64)
    log_z = float(torch.logsumexp(t_scores, dim=0))
    probs = torch.exp(t_scores - log_z)
    marg = torch.zeros(real_len, K, dtype=torch.float64)
    for path, p in zip(paths, probs.tolist()):
        for t, slot in enumerate(path):
            marg[t, slot] += p
    return log_z, marg


def _brute_force_trigram_crf(emit_1, bigram_1, trigram_1, mask_1, real_len):
    M = emit_1.shape[1]
    valid_slots = [
        [m for m in range(M) if bool(mask_1[t, m])]
        for t in range(real_len)
    ]
    paths = list(itertools.product(*valid_slots))
    scores = []
    for path in paths:
        s = sum(float(emit_1[t, path[t]]) for t in range(real_len))
        s += sum(float(bigram_1[t, path[t], path[t + 1]]) for t in range(real_len - 1))
        s += sum(
            float(trigram_1[t, path[t], path[t + 1], path[t + 2]])
            for t in range(real_len - 2)
        )
        scores.append(s)
    t_scores = torch.tensor(scores, dtype=torch.float64)
    log_z = float(torch.logsumexp(t_scores, dim=0))
    best_idx = int(t_scores.argmax())
    best_score = float(t_scores[best_idx])
    probs = torch.exp(t_scores - log_z)
    marg = torch.zeros(real_len, M, dtype=torch.float64)
    for path, p in zip(paths, probs.tolist()):
        for t, slot in enumerate(path):
            marg[t, slot] += p
    return log_z, best_score, marg, paths, scores


class TestOrthographyAndInventory(unittest.TestCase):
    def test_tone_placement_classic_and_modern(self):
        self.assertEqual(normalize_tone_placement("hoà", "classic"), "hòa")
        self.assertEqual(normalize_tone_placement("hòa", "modern"), "hoà")
        self.assertEqual(normalize_tone_placement("thuỷ", "classic"), "thủy")
        self.assertEqual(normalize_tone_placement("thủy", "modern"), "thuỷ")
        self.assertEqual(normalize_tone_placement("khoẻ", "classic"), "khỏe")
        self.assertEqual(normalize_tone_placement("khỏe", "modern"), "khoẻ")
        self.assertEqual(normalize_tone_placement("Hoà", "classic"), "Hòa")
        self.assertEqual(normalize_tone_placement("HOÀ", "classic"), "HÒA")

    def test_misplaced_tones_and_foreign_safety(self):
        self.assertEqual(normalize_tone_placement("qúy", "classic"), "quý")
        self.assertEqual(normalize_tone_placement("qủa", "classic"), "quả")
        self.assertEqual(normalize_tone_placement("vịêt", "classic"), "việt")
        self.assertEqual(normalize_tone_placement("tòan", "classic"), "toàn")
        self.assertEqual(normalize_tone_placement("hùynh", "classic"), "huỳnh")
        self.assertEqual(normalize_tone_placement("gìa", "classic"), "già")
        self.assertEqual(normalize_tone_placement("gì", "classic"), "gì")
        self.assertEqual(normalize_tone_placement("gìn", "classic"), "gìn")
        # Foreign words (including multi-syllable or pinyin) must remain untouched
        for foreign in ("café", "José", "Müller", "résumé", "iPhone2026", "Napoléon", "guó"):
            self.assertEqual(normalize_tone_placement(foreign, "classic"), foreign)

    def test_invalid_multitone_filtered(self):
        self.assertFalse(is_valid_vn_or_clean_candidate("vỉết", "viet"))
        self.assertFalse(is_valid_vn_or_clean_candidate("i\u0307", "i"))
        self.assertTrue(is_valid_vn_or_clean_candidate("việt", "viet"))
        self.assertTrue(is_valid_vn_or_clean_candidate("quý", "quy"))

    def test_phonology_decomposition(self):
        p_ban1 = parse_syllable_phonology("bàn")
        p_ban2 = parse_syllable_phonology("bán")
        self.assertEqual(p_ban1[:5], p_ban2[:5])
        self.assertNotEqual(p_ban1[5], p_ban2[5])
        p_duoc = parse_syllable_phonology("được")
        self.assertEqual(p_duoc[4], 6)  # double_horn
        self.assertEqual(p_duoc[5], 5)  # nang
        # Foreign fallback must never raise
        p_for = parse_syllable_phonology("café123")
        self.assertEqual(len(p_for), 6)

    def test_lossless_inventory_compaction(self):
        import collections
        counts = {
            "2023": collections.Counter({"2023": 500}),
            "internet": collections.Counter({"internet": 300}),
            "ban": collections.Counter({"bàn": 50, "bán": 40, "ban": 10}),
            "hoa": collections.Counter({"hòa": 30, "hoà": 20, "hoa": 15}),
            "viet": collections.Counter({"việt": 100, "vỉết": 2}),
            "doc": collections.Counter({"đọc": 80}),  # accented singleton MUST be kept
        }
        bases, vocab, rows, _, stats = compact_inventory(
            sorted(counts), [], ctab_or_counts=counts, style="classic"
        )
        self.assertNotIn("2023", bases)
        self.assertNotIn("internet", bases)
        self.assertIn("doc", bases)
        self.assertIn("ban", bases)
        self.assertIn("hoa", bases)
        self.assertEqual(stats["dropped_identity_singletons"], 2)
        # Merged hoà + hòa into single entry hòa
        hoa_idx = bases.index("hoa")
        hoa_forms = [vocab[vi] for vi in rows[hoa_idx] if vi >= 0]
        self.assertEqual(hoa_forms, ["hòa", "hoa"])
        self.assertNotIn("vỉết", vocab)

        # Frequent pure-VN identity singletons can be retained when min_identity_vn_count > 0
        counts["theo"] = collections.Counter({"theo": 25})
        counts["rarevn"] = collections.Counter({"rarevn": 2})
        bases2, _, _, _, _ = compact_inventory(
            sorted(counts), [], ctab_or_counts=counts, style="classic", min_identity_vn_count=5
        )
        self.assertIn("theo", bases2)
        self.assertNotIn("2023", bases2)
        self.assertNotIn("internet", bases2)  # > 7 chars
        self.assertNotIn("rarevn", bases2)    # count < 5


class TestExactCRFs(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_v1_forward_backward_matches_bruteforce_and_v1_logz(self):
        B, L, K = 3, 4, 3
        emit = torch.randn(B, L, K)
        trans = torch.randn(B, L - 1, K, K)
        mask = torch.tensor(
            [
                [[True, True, False], [True, True, True], [True, False, False], [True, True, False]],
                [[True, True, True], [True, True, False], [False, False, False], [False, False, False]],
                [[True, True, False], [False, False, False], [False, False, False], [False, False, False]],
            ]
        )
        lengths = [4, 2, 1]
        emit_m = emit.masked_fill(~mask, -1e9)
        log_z_v1 = sm_v1.log_partition(emit_m, trans, mask)
        log_marg, log_z_fb = forward_backward_marginals(emit, trans, mask)
        torch.testing.assert_close(log_z_fb, log_z_v1, atol=1e-4, rtol=1e-4)

        for b, rlen in enumerate(lengths):
            bf_z, bf_marg = _brute_force_linear_crf(emit[b], trans[b], mask[b], rlen)
            self.assertAlmostEqual(float(log_z_fb[b]), bf_z, places=4)
            fb_probs = torch.exp(log_marg[b, :rlen]) * mask[b, :rlen].float()
            torch.testing.assert_close(fb_probs, bf_marg.float(), atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(fb_probs.sum(dim=-1), torch.ones(rlen), atol=1e-4, rtol=1e-4)

    def test_v1_constrained_viterbi_and_mbr_lock(self):
        emit = torch.tensor([[[10.0, 1.0], [1.0, 10.0]]])
        trans = torch.tensor([[[0.0, 0.0], [20.0, 0.0]]])
        mask = torch.ones(1, 2, 2, dtype=torch.bool)
        locked = torch.tensor([[[False, True], [False, False]]])
        pred_v = constrained_viterbi(emit, trans, mask, locked_mask=locked)
        self.assertEqual(pred_v[0, 0].item(), 1)
        pred_m, _ = mbr_decode(emit, trans, mask, locked_mask=locked)
        self.assertEqual(pred_m[0, 0].item(), 1)

    def test_trigram_crf_matches_bruteforce_all_lengths(self):
        B, L, M = 3, 4, 3
        emit_p = torch.randn(B, L, M)
        bigram = torch.randn(B, L - 1, M, M)
        trigram = torch.randn(B, L - 2, M, M, M)
        mask_p = torch.tensor(
            [
                [[True, True, False], [True, True, True], [True, False, True], [True, True, True]],
                [[True, True, True], [True, True, False], [False, False, False], [False, False, False]],
                [[True, True, False], [False, False, False], [False, False, False], [False, False, False]],
            ]
        )
        lengths = [4, 2, 1]

        lz = trigram_log_partition(emit_p, bigram, trigram, mask_p)
        vit = trigram_viterbi(emit_p, bigram, trigram, mask_p)
        log_marg, lz_m = trigram_marginals(emit_p, bigram, trigram, mask_p)
        torch.testing.assert_close(lz, lz_m, atol=1e-4, rtol=1e-4)

        gold_p = torch.tensor([[0, 2, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0]])
        gs = trigram_gold_score(emit_p, bigram, trigram, mask_p, gold_p)

        for b, rlen in enumerate(lengths):
            bf_z, bf_best, bf_marg, paths, scores = _brute_force_trigram_crf(
                emit_p[b], bigram[b], trigram[b], mask_p[b], rlen
            )
            self.assertAlmostEqual(float(lz[b]), bf_z, places=4)
            # Check gold_score
            g_tuple = tuple(gold_p[b, :rlen].tolist())
            self.assertAlmostEqual(float(gs[b]), scores[paths.index(g_tuple)], places=4)
            # Check Viterbi optimality
            v_tuple = tuple(vit[b, :rlen].tolist())
            self.assertAlmostEqual(scores[paths.index(v_tuple)], bf_best, places=4)
            # Check marginals
            probs = torch.exp(log_marg[b, :rlen]) * mask_p[b, :rlen].float()
            torch.testing.assert_close(probs, bf_marg.float(), atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(probs.sum(dim=-1), torch.ones(rlen), atol=1e-4, rtol=1e-4)

        # Also explicitly check L=1 shape handling
        e1 = torch.randn(2, 1, 3)
        m1 = torch.ones(2, 1, 3, dtype=torch.bool)
        b1 = torch.empty(2, 0, 3, 3)
        t1 = torch.empty(2, 0, 3, 3, 3)
        torch.testing.assert_close(trigram_log_partition(e1, b1, t1, m1), torch.logsumexp(e1[:, 0], dim=-1))
        self.assertEqual(trigram_viterbi(e1, b1, t1, m1).shape, (2, 1))
        lm1, _ = trigram_marginals(e1, b1, t1, m1)
        torch.testing.assert_close(torch.exp(lm1[:, 0]).sum(-1), torch.ones(2), atol=1e-5, rtol=1e-5)


class TestViDiCRFv2ModelAndPipeline(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(11)
        self.train_pairs = [
            ("Hom nay toi di hoc o truong dai hoc quoc gia.", "Hôm nay tôi đi học ở trường đại học quốc gia."),
            ("Ban ay dang ban hoa qua o cho Ben Thanh.", "Bạn ấy đang bán hoa quả ở chợ Bến Thành."),
            ("Trong nam 2026 chung toi phat trien phan mem tieng Viet.", "Trong năm 2026 chúng tôi phát triển phần mềm tiếng Việt."),
            ("Hoa binh la uoc mo cua nhan dan.", "Hòa bình là ước mơ của nhân dân."),
            ("Co ay mac ao dai rat dep, cuoi rat tuoi.", "Cô ấy mặc áo dài rất đẹp, cười rất tươi."),
        ]
        self.inv = build_v2_inventory(
            self.train_pairs, max_lex_vocab=256, min_lex_count=1, kmax=8, chw=16, min_char_count=1
        )
        ct = self.inv.char_tables
        self.model = ViDiCRFv2(
            char_base_map=ct["char_base_map"],
            char_mod_map=ct["char_mod_map"],
            char_tone_map=ct["char_tone_map"],
            n_base_chars=ct["n_base_chars"],
            n_char_mods=ct["n_char_mods"],
            n_char_tones=ct["n_char_tones"],
            base_ch=self.inv.base_ch,
            base_phono=self.inv.base_phono,
            cand_ch=self.inv.cand_ch,
            cand_phono=self.inv.cand_phono,
            cand_lex=self.inv.cand_lex,
            cand_ids=self.inv.cand_ids,
            cand_priors=self.inv.cand_priors,
            npunct=len(self.inv.punct),
            n_lex_vocab=self.inv.n_lex_vocab,
            d=64,
            n_heads=4,
            stage1_layers=1,
            stage2_layers=1,
            ffn_dim=128,
            lex_dim=32,
            r_big=16,
            r_tri=8,
            top_m=4,
            drop=0.0,
        )

    def test_padding_invariance_and_cache_equivalence(self):
        enc = encode_corpus_v2(self.train_pairs[:2], self.inv, max_len=64)
        b_single = make_batch_v2(enc, self.inv, np.array([0]), "cpu")
        b_pair = make_batch_v2(enc, self.inv, np.array([0, 1]), "cpu")
        self.model.eval()
        with torch.no_grad():
            out_s = self.model(
                b_single["base_ids"], b_single["sent_len"], b_single["case_ids"],
                b_single["left_gaps"], b_single["right_gaps"],
                b_single["surface_chars"], b_single["surface_phono"],
                b_single["has_input_diac"], b_single["surface_lex"],
            )
            out_p = self.model(
                b_pair["base_ids"], b_pair["sent_len"], b_pair["case_ids"],
                b_pair["left_gaps"], b_pair["right_gaps"],
                b_pair["surface_chars"], b_pair["surface_phono"],
                b_pair["has_input_diac"], b_pair["surface_lex"],
            )
            n0 = int(b_single["sent_len"][0])
            torch.testing.assert_close(out_s["emit2"][0, :n0], out_p["emit2"][0, :n0], atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(out_s["bigram"][0, : n0 - 1], out_p["bigram"][0, : n0 - 1], atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(out_s["trigram"][0, : n0 - 2], out_p["trigram"][0, : n0 - 2], atol=1e-5, rtol=1e-5)

            # Cached vs uncached equivalence
            self.model.build_inference_cache()
            out_c = self.model(
                b_single["base_ids"], b_single["sent_len"], b_single["case_ids"],
                b_single["left_gaps"], b_single["right_gaps"],
                b_single["surface_chars"], b_single["surface_phono"],
                b_single["has_input_diac"], b_single["surface_lex"],
            )
            torch.testing.assert_close(out_s["emit2"], out_c["emit2"], atol=1e-5, rtol=1e-5)
            self.model.invalidate_cache()

    def test_training_loss_and_all_gradients_finite(self):
        enc = encode_corpus_v2(self.train_pairs, self.inv, max_len=64)
        rng = np.random.default_rng(123)
        batch = make_batch_v2(enc, self.inv, np.arange(len(enc.seg_lens)), "cpu", partial_diac_prob=0.5, rng=rng)
        self.model.train()
        out = self.model(
            batch["base_ids"], batch["sent_len"], batch["case_ids"],
            batch["left_gaps"], batch["right_gaps"],
            batch["surface_chars"], batch["surface_phono"],
            batch["has_input_diac"], batch["surface_lex"],
            gold=batch["gold"],
        )
        loss, stats = compute_loss(out, batch["gold"], batch["sent_len"], core_mask=batch["core_mask"])
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(stats["crf_loss"], 0.0)
        loss.backward()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Missing gradient on {name}")
                self.assertTrue(torch.isfinite(param.grad).all(), f"Non-finite gradient on {name}")

    def test_zero_copy_oov_diacritic_preserve_and_long_text(self):
        restorer = FastRestorerV2(self.model, self.inv, device="cpu", use_cache=True)
        ptrs_before = {
            name: (buf.data_ptr(), tuple(buf.shape))
            for name, buf in self.model.named_buffers()
        }
        # Input with completely unseen OOV words + numbers + user-preserved accented words
        text = "Hom nay, startup AntigravityX99 phat trien phần mềm moi trong nam 2099!"
        out_text, cov = restorer.predict(text, preserve_input_diacritics=True, decode_mode="mbr")
        ptrs_after = {
            name: (buf.data_ptr(), tuple(buf.shape))
            for name, buf in self.model.named_buffers()
        }
        self.assertEqual(ptrs_before, ptrs_after)
        self.assertGreater(cov["oov_copy_tokens"], 0)
        self.assertIn("AntigravityX99", out_text)
        self.assertIn("2099", out_text)
        self.assertIn("phần", out_text)
        self.assertIn("mềm", out_text)

        # Long text > 256 words sliding window
        long_text = " ".join(["Hom nay toi di hoc o truong dai hoc quoc gia."] * 40)
        out_long, cov_long = restorer.predict(long_text, decode_mode="viterbi", max_window=64, halo=12)
        self.assertEqual(cov_long["input_tokens"], 440)
        self.assertEqual(len(out_long.split()), len(long_text.split()))

        # Classic vs Modern tone placement toggle
        out_cls, _ = restorer.predict("Hoa binh", tone_style="classic")
        out_mod, _ = restorer.predict("Hoa binh", tone_style="modern")
        self.assertEqual(strip_diac(out_cls).lower(), "hoa binh")
        self.assertEqual(strip_diac(out_mod).lower(), "hoa binh")
        if out_cls.lower().startswith("hòa"):
            self.assertTrue(out_mod.lower().startswith("hoà"))


if __name__ == "__main__":
    unittest.main()
