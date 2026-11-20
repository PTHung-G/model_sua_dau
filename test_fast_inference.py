"""Equivalence + optimization tests for vidicrf_v2.fast_inference and pmi."""
from __future__ import annotations

import unittest

import torch

from vidicrf_v2.fast_inference import (
    CSRInventory,
    FastRestorerV2Opt,
    WordCache,
    scripted_viterbi,
)
from vidicrf_v2.model import trigram_viterbi
from vidicrf_v2.pmi import PMITable, pmi_bigram_addend


def _rand_lattice(B, L, M, seed):
    g = torch.Generator().manual_seed(seed)
    emit = torch.randn(B, L, M, generator=g)
    bigram = torch.randn(B, max(0, L - 1), M, M, generator=g) * 0.5
    trigram = torch.randn(B, max(0, L - 2), M, M, M, generator=g) * 0.3
    mask = torch.rand(B, L, M, generator=g) > 0.3
    # Guarantee >=1 valid candidate per position
    for b in range(B):
        for t in range(L):
            mask[b, t, torch.randint(0, M, (1,), generator=g)] = True
    return emit, bigram, trigram, mask


class TestScriptedViterbi(unittest.TestCase):
    def test_matches_eager_random(self):
        for i, (B, L, M) in enumerate([(3, 1, 4), (2, 2, 3), (3, 5, 6), (2, 7, 4), (1, 4, 2)]):
            emit, bigram, trigram, mask = _rand_lattice(B, L, M, seed=100 + i)
            eager = trigram_viterbi(emit, bigram, trigram, mask)
            script = scripted_viterbi(emit, bigram, trigram, mask)
            self.assertTrue(torch.equal(eager, script), f"case {(B, L, M)}")

    def test_matches_eager_locked(self):
        g = torch.Generator().manual_seed(7)
        B, L, M = 3, 6, 5
        emit, bigram, trigram, mask = _rand_lattice(B, L, M, seed=11)
        locked = torch.rand(B, L, M, generator=g) > 0.6
        # Ensure at least one locked slot per locked row and overlap with mask
        locked = locked & mask
        for b in range(B):
            t = torch.randint(0, L, (1,), generator=g)
            locked[b, t, mask[b, t].float().argmax()] = True
        eager = trigram_viterbi(emit, bigram, trigram, mask, locked_mask_p=locked)
        script = scripted_viterbi(emit, bigram, trigram, mask, locked_mask_p=locked)
        self.assertTrue(torch.equal(eager, script))

    def test_bruteforce_tiny(self):
        # Exhaustive enumeration, B=1, L=4, M=3, all-valid mask
        emit, bigram, trigram, mask = _rand_lattice(1, 4, 3, seed=42)
        mask[:] = True
        best_score, best_path = None, None
        for y0 in range(3):
            for y1 in range(3):
                for y2 in range(3):
                    for y3 in range(3):
                        path = [y0, y1, y2, y3]
                        s = emit[0, 0, y0] + emit[0, 1, y1] + emit[0, 2, y2] + emit[0, 3, y3]
                        s = s + bigram[0, 0, y0, y1] + bigram[0, 1, y1, y2] + bigram[0, 2, y2, y3]
                        s = s + trigram[0, 0, y0, y1, y2] + trigram[0, 1, y1, y2, y3]
                        if best_score is None or s > best_score:
                            best_score, best_path = s, path
        pred = scripted_viterbi(emit, bigram, trigram, mask)[0].tolist()
        self.assertEqual(pred, best_path)


class TestCSR(unittest.TestCase):
    def test_roundtrip_and_stats(self):
        dense = torch.tensor(
            [
                [5, -1, -1, -1],
                [-1, -1, -1, -1],
                [7, 8, 9, -1],
                [2, -1, -1, -1],
            ],
            dtype=torch.int32,
        )
        csr = CSRInventory(dense)
        self.assertEqual(csr.row(0).tolist(), [5])
        self.assertEqual(csr.row(1).tolist(), [])
        self.assertEqual(csr.row(2).tolist(), [7, 8, 9])
        self.assertTrue(torch.equal(csr.dense_like(), dense))
        self.assertAlmostEqual(csr.stats["dense_utilization_pct"], 5 / 16 * 100, places=5)
        self.assertGreater(csr.stats["compression_ratio"], 1.0)


class TestWordCache(unittest.TestCase):
    def test_features_match_manual(self):
        from vidicrf_v2.data_pipeline import case_id
        from vidicrf_v2.orthography import has_diacritic, normalize_tone_placement, strip_diac

        wc = WordCache({}, 12, style="classic")
        for tok in ["Ngân Hàng", "hòa", "hoà", "ĐÀ NẴNG", "việt", "ABC", "kinh tế"]:
            b, _s_low, s_enc, case, hd, _lex = wc.features(tok)[:6]
            self.assertEqual(b, strip_diac(tok).lower())
            self.assertEqual(case, case_id(tok))
            self.assertEqual(hd, 1 if has_diacritic(tok) else 0)
            if hd:
                s_manual = normalize_tone_placement(
                    __import__("unicodedata").normalize("NFC", tok).lower(), style="classic"
                )
            else:
                s_manual = b
            self.assertEqual(s_enc, s_manual)

    def test_char_tensors_consistent(self):
        from vidicrf_v2.data_pipeline import encode_strings_chars, encode_strings_phono

        wc = WordCache({"a": 2, "b": 3}, 8, style="classic")
        words = ["ab", "ba", "ab"]
        ch, ph = wc.char_tensors(words)
        ch_ref = encode_strings_chars(words, {"a": 2, "b": 3}, 8)
        ph_ref = encode_strings_phono(words)
        self.assertTrue(torch.equal(ch, ch_ref))
        self.assertTrue(torch.equal(ph, ph_ref))


class _TinyFixture:
    """Build a small but realistic v2 inventory + untrained model from a tiny corpus."""

    CORPUS = [
        ("ngan hang nha nuocViet Nam vua ha lai suat dieu hanh len them 25 diem phan tram",
         "Ngân hàng Nhà nước Việt Nam vừa hạ lãi suất điều hành thêm 25 điểm phần trăm"),
        ("kinh te Viet Nam nam nay tang truong du kien dat muc 7 phan tram tro len",
         "Kinh tế Việt Nam năm nay tăng trưởng dự kiến đạt mức 7 phần trăm trở lên"),
        ("chu nghia xa hoi khoa hoc la mot trong ba dong lau cua chu nghia Mac Lenin",
         "Chủ nghĩa xã hội khoa học là một trong ba bộ phận cấu thành của chủ nghĩa Mác Lênin"),
        ("ban ghe van phong gia re duoc ban nhieu nhat tai chuoi cua hang nay",
         "Bàn ghế văn phòng giá rẻ được bán nhiều nhất tại chuỗi cửa hàng này"),
        ("hoc sinh lop hai duoc hoc mien phi sach giao khoa theo nghi quyet moi",
         "Học sinh lớp hai được học miễn phí sách giáo khoa theo nghị quyết mới"),
        ("cong an xa da truy bat doi tuong khai thac khoang san trai phep o vung cao",
         "Công an xã đã truy bắt đối tượng khai thác khoáng sản trái phép ở vùng cao"),
        ("thi truong chung khoan hom nay phuc hoi nho dong von ngoai vao manh me",
         "Thị trường chứng khoán hôm nay phục hồi nhờ dòng vốn ngoại vào mạnh mẽ"),
        ("ngan hang trung uong nhat ban van duy chinh sach lai suat am hoan toan",
         "Ngân hàng trung ương Nhật Bản vẫn duy trì chính sách lãi suất âm hoàn toàn"),
    ]

    def __init__(self, seed: int = 0):
        torch.manual_seed(seed)
        pairs = self.CORPUS * 3  # small counts so compaction keeps multi-candidate bases
        from vidicrf_v2.data_pipeline import build_v2_inventory

        self.inv = build_v2_inventory(
            pairs, max_lex_vocab=256, min_lex_count=1, kmax=8, chw=14,
            min_char_count=1, style="classic", normalize_orthography=True,
            drop_identity_singletons=True, filter_invalid=True,
        )
        from vidicrf_v2.data_pipeline import V2Inventory  # noqa: F401  (typing only)
        from vidicrf_v2.model import ViDiCRFv2

        ct = self.inv.char_tables
        torch.manual_seed(seed)
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
            d=32, n_heads=2, stage1_layers=1, stage2_layers=1,
            ffn_dim=64, lex_dim=16, r_big=8, r_tri=4, top_m=4, drop=0.0,
        ).eval()


class TestOptRestorerEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fix = _TinyFixture()

    def _restorers(self):
        base = FastRestorerV2Opt(self.fix.model, self.fix.inv, device="cpu", use_cache=True)
        # Second restorer sharing the same model/inference cache
        opt = FastRestorerV2Opt(self.fix.model, self.fix.inv, device="cpu", use_cache=False)
        return base, opt

    SENTENCES = [
        "ngan hang nha nuoc vua ha lai suat",
        "Kinh tế việt nam tăng trưởng mạnh",
        "ban ghe van phong gia re ban nhieu nhat",
        "Thi truong chung khoan hom nay Phuc Hoi",
        "chu nghia xa hoi khoa hoc",
    ]

    def test_viterbi_matches_parent_semantics(self):
        fix = self.fix
        # Parent reference: build a plain FastRestorerV2
        from vidicrf_v2.restorer import FastRestorerV2

        parent = FastRestorerV2(fix.model, fix.inv, device="cpu", use_cache=False)
        opt = FastRestorerV2Opt(fix.model, fix.inv, device="cpu", use_cache=False)
        for s in self.SENTENCES:
            r1 = parent.predict(s, decode_mode="viterbi")
            r2 = opt.predict(s, decode_mode="viterbi")
            self.assertEqual(r1[0], r2[0], s)
            self.assertEqual(r1[1], r2[1])

    def test_mbr_and_options_match(self):
        from vidicrf_v2.restorer import FastRestorerV2

        fix = self.fix
        parent = FastRestorerV2(fix.model, fix.inv, device="cpu", use_cache=False)
        opt = FastRestorerV2Opt(fix.model, fix.inv, device="cpu", use_cache=False)
        s = "ngan hang vua ha lai suat dieu hanh"
        for kwargs in [
            dict(decode_mode="mbr"),
            dict(decode_mode="viterbi", preserve_input_diacritics=True),
            dict(decode_mode="viterbi", locked_word_indices=[0, 2]),
            dict(decode_mode="viterbi", tone_style="modern"),
        ]:
            a = parent.predict(s, **kwargs)[0]
            b = opt.predict(s, **kwargs)[0]
            self.assertEqual(a, b, str(kwargs))

    def test_windows_match(self):
        from vidicrf_v2.restorer import FastRestorerV2

        fix = self.fix
        parent = FastRestorerV2(fix.model, fix.inv, device="cpu", use_cache=False)
        opt = FastRestorerV2Opt(fix.model, fix.inv, device="cpu", use_cache=False)
        long_src = " ".join(s for s, _ in fix.__class__.CORPUS for _ in range(2))[:4000]
        a = parent.predict(long_src, decode_mode="viterbi", max_window=24, halo=6)[0]
        b = opt.predict(long_src, decode_mode="viterbi", max_window=24, halo=6)[0]
        self.assertEqual(a, b)

    def test_cache_clear_deterministic(self):
        _, opt = self._restorers()
        s = self.SENTENCES[0]
        r1 = opt.predict(s)[0]
        r2 = opt.predict(s)[0]
        self.assertEqual(r1, r2)
        opt.clear_caches()
        r3 = opt.predict(s)[0]
        self.assertEqual(r1, r3)


class TestPMI(unittest.TestCase):
    def test_table_values_and_injection(self):
        targets = [
            "ngân hàng nhà nước và ngân hàng thương mại",
            "kinh tế việt nam tăng mạnh",
            "ngân hàng thương mại tăng vốn",
        ]
        table = PMITable(targets, min_pair_count=1, max_pairs=1000)
        # ("ngân", "hàng") co-occurs 3x, strong PMI; random pair weak/absent
        x = table.wid["ngân"]
        y = table.wid["hàng"]
        n = len(table.words) + 1
        v = table.lookup_keys(torch.tensor([x * n + y]))
        self.assertGreater(float(v[0]), 0.0)
        ghost = table.lookup_keys(torch.tensor([999999]))
        self.assertEqual(float(ghost[0]), 0.0)
        # words appearing in no kept pair must have id 0/absent behavior
        self.assertIn("hàng", table.wid)

    def test_injection_shapes_and_gradient(self):
        table = PMITable(
            ["ngân hàng bán vàng", "ngân hàng bándollars không", "bàn bán hàng"],
            min_pair_count=1, max_pairs=100,
        )
        vocab = ["ngân", "hàng", "bàn", "bán"]
        table.attach_vocab(vocab)
        table.to("cpu")
        B, L, M, K = 2, 5, 3, 4
        base_ids = torch.randint(0, K, (B, L))
        top_idx = torch.randint(0, K, (B, L, M))
        cand_ids = torch.randint(0, len(vocab), (K + 1, K))
        bigram = torch.randn(B, L - 1, M, M, requires_grad=True)
        lam = torch.tensor(0.5, requires_grad=True)
        addend = pmi_bigram_addend(table, bigram, base_ids, top_idx, cand_ids, lam)
        self.assertEqual(addend.shape, bigram.shape)
        loss = addend.sum()
        loss.backward()
        self.assertTrue(torch.isfinite(lam.grad))
        # lambda gradient must be nonzero for at least one covered pair
        self.assertGreater(float(lam.grad.abs()), 0.0)

    def test_smoke_training_step(self):
        """One real training step with PMI injection on the tiny corpus: finite loss, λ grad flows."""
        fix = _TinyFixture(seed=3)
        targets = [t for _, t in _TinyFixture.CORPUS] * 3
        table = PMITable(targets, min_pair_count=2, max_pairs=5000)
        table.attach_vocab(fix.inv.vocab)
        table.to("cpu")
        lam = torch.tensor(0.5, requires_grad=True)
        from vidicrf_v2.model import compute_loss
        from vidicrf_v2.data_pipeline import encode_corpus_v2, make_batch_v2
        import numpy as np

        enc = encode_corpus_v2(_TinyFixture.CORPUS, fix.inv, max_len=32, halo=4)
        self.assertGreater(len(enc.seg_lens), 0)
        idx = np.array([0])
        batch = make_batch_v2(enc, fix.inv, idx, "cpu", partial_diac_prob=0.0)
        out = fix.model(
            batch["base_ids"], batch["sent_len"], batch["case_ids"], batch["left_gaps"],
            batch["right_gaps"], batch["surface_chars"], batch["surface_phono"],
            batch["has_input_diac"], batch["surface_lex"], gold=batch["gold"],
        )
        addend = pmi_bigram_addend(
            table, out["bigram"], batch["base_ids"],
            out["top_idx"], fix.inv.cand_ids, lam,
        )
        self.assertEqual(addend.shape, out["bigram"].shape)
        out["bigram"] = out["bigram"] + addend
        loss, stats = compute_loss(out, batch["gold"], batch["sent_len"], margin=0.10)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(lam.grad)
        self.assertGreater(float(lam.grad.abs()), 0.0)


if __name__ == "__main__":
    unittest.main()
