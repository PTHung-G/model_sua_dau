"""Local CPU benchmark for ViDiCRF v3 optimization targets (replicates Exp-A setup).

Measures, on the real compacted DauDemo inventory (1.38M bases -> compaction):
  1. Stage breakdown of predict() (tokenize / encode / assemble / forward / decode / reassemble)
  2. Baseline FastRestorerV2 vs FastRestorerV2Opt (word cache + JIT Viterbi + input memo)
  3. CSR compaction stats + true runtime memory accounting (incl. inference cache)

Untrained weights are intentional (latency/memory profiling, as in run_experiment_a).
Results -> lich_su_sua_doi/ket_qua_toi_uu_cpu_v3.json
"""
from __future__ import annotations

import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vidicrf_v2.data_pipeline import V2Inventory, build_char_tables, encode_strings_chars, encode_strings_phono
from vidicrf_v2.fast_inference import CSRInventory, FastRestorerV2Opt, profile_predict_breakdown
from vidicrf_v2.model import ViDiCRFv2
from vidicrf_v2.orthography import compact_inventory, has_diacritic
from vidicrf_v2.restorer import FastRestorerV2

DEMO_DIR = ROOT / "windows_demo_build" / "DauDemo"
RESULTS_OUT = ROOT / "lich_su_sua_doi" / "ket_qua_toi_uu_cpu_v3.json"

SAMPLES = [
    "ngan hang nha nuoc vua ha lai suat dieu hanh xuong them 25 diem phan tram tu ngay mai",
    "kinh te viet nam nam nay du kien tang truong tren 7 phan tram nho sang tao va dau tu nuoc ngoai",
    "chu nghia xa hoi khoa hoc la mot trong ba bo phan cau thanh cua chu nghia mac lenin",
    "ban ghe van phong gia re duoc ban nhieu nhat tai chuoi cua hang nay vi mau ma dep va ben",
    "thi truong chung khoan hom nay phuc hoi manh nho dong von ngoai vao nganh ngan hang va bat dong san",
    "hoc sinh lop hai duoc hoc mien phi sach giao khoa theo nghi quyet moi cua bo giao duc va dao tao",
    "cong an xa da truy bat doi tuong khai thac khoang san trai phep trong khu vuc nui cao gioi han",
    "ngan hang trung uong nhat ban van duy tri chinh sach lai suat am hoan toan de kich cau trong nuoc",
]
LONG_TEXT = " ".join(SAMPLES + SAMPLES)  # ~176 words, forces halo windows at max_window=96


def load_daudemo_inventory() -> tuple:
    bases = json.loads((DEMO_DIR / "bases.json").read_text(encoding="utf-8"))
    vocab = json.loads((DEMO_DIR / "vocab.json").read_text(encoding="utf-8"))
    chars_map = json.loads((DEMO_DIR / "chars.json").read_text(encoding="utf-8"))
    punct_list = json.loads((DEMO_DIR / "punct.json").read_text(encoding="utf-8"))
    return bases, vocab, chars_map, punct_list


def build_v2_from_daudemo(bases, vocab, punct_list, kmax=32, chw=16):
    from vidicrf_v2.orthography import strip_diac

    bid_v1 = {b: i + 1 for i, b in enumerate(bases)}
    ctab_np = np.full((len(bases) + 1, kmax), -1, dtype=np.int32)
    row_fill = np.zeros(len(bases) + 1, dtype=np.int32)
    for vi, form in enumerate(vocab):
        bi = bid_v1.get(strip_diac(form).lower())
        if bi is not None and row_fill[bi] < kmax:
            ctab_np[bi, row_fill[bi]] = vi
            row_fill[bi] += 1

    t0 = time.perf_counter()
    new_bases, new_vocab, new_rows, new_counts, comp_stats = compact_inventory(
        bases, vocab, ctab_or_counts=ctab_np, style="classic",
        drop_identity_singletons=True, filter_invalid=True, kmax=kmax,
    )
    build_s = time.perf_counter() - t0

    bid = {b: i + 1 for i, b in enumerate(new_bases)}
    vid = {v: i for i, v in enumerate(new_vocab)}
    cid = torch.full((len(new_bases) + 1, kmax), -1, dtype=torch.int32)
    priors = torch.zeros((len(new_bases) + 1, kmax, 3), dtype=torch.float16)
    for bi, (b, r_ids, r_cnts) in enumerate(zip(new_bases, new_rows, new_counts), 1):
        tot = sum(r_cnts)
        for k, (vi, cnt) in enumerate(zip(r_ids, r_cnts)):
            if vi < 0:
                break
            cid[bi, k] = vi
            priors[bi, k, 0] = math.log(cnt / max(1.0, tot) + 1e-6)
            priors[bi, k, 1] = math.log1p(cnt) / 15.0
            priors[bi, k, 2] = 1.0 if new_vocab[vi] == b else 0.0

    lex_n = min(24576, len(new_vocab)) + 1
    cand_lex = torch.zeros(len(new_vocab), dtype=torch.int32)
    cand_hd = torch.zeros(len(new_vocab), dtype=torch.bool)
    for vi in range(min(24576, len(new_vocab))):
        cand_lex[vi] = vi + 1
    for vi, w in enumerate(new_vocab):
        cand_hd[vi] = has_diacritic(w)

    ct = build_char_tables(new_bases + new_vocab, min_char_count=10)
    c2i = ct["c2i"]
    inv = V2Inventory(
        bases=new_bases, bid=bid, vocab=new_vocab, vid=vid,
        cand_ids=cid, cand_priors=priors,
        base_ch=encode_strings_chars([""] + new_bases, c2i, chw),
        base_phono=encode_strings_phono([""] + new_bases),
        cand_ch=encode_strings_chars(new_vocab, c2i, chw),
        cand_phono=encode_strings_phono(new_vocab),
        cand_lex=cand_lex, cand_has_diac=cand_hd,
        n_lex_vocab=lex_n, punct=punct_list,
        pid={g: i for i, g in enumerate(punct_list)},
        char_tables=ct, chw=chw, kmax=kmax,
        style="classic", normalize_orthography=True, stats=comp_stats,
    )
    return inv, build_s


def build_model(inv: V2Inventory):
    ct = inv.char_tables
    return ViDiCRFv2(
        char_base_map=ct["char_base_map"],
        char_mod_map=ct["char_mod_map"],
        char_tone_map=ct["char_tone_map"],
        n_base_chars=ct["n_base_chars"],
        n_char_mods=ct["n_char_mods"],
        n_char_tones=ct["n_char_tones"],
        base_ch=inv.base_ch,
        base_phono=inv.base_phono,
        cand_ch=inv.cand_ch,
        cand_phono=inv.cand_phono,
        cand_lex=inv.cand_lex,
        cand_ids=inv.cand_ids,
        cand_priors=inv.cand_priors,
        npunct=len(inv.punct),
        n_lex_vocab=inv.n_lex_vocab,
        d=128, n_heads=4, stage1_layers=2, stage2_layers=2,
        ffn_dim=256, lex_dim=64, r_big=32, r_tri=16, top_m=8, drop=0.0,
    ).eval()


def lat_ms(fn, warmup=5, reps=30):
    for _ in range(warmup):
        fn()
    xs = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t) * 1000.0)
    return round(float(np.median(xs)), 2), round(float(np.percentile(xs, 95)), 2)


def main():
    torch.set_num_threads(2)
    print("Loading DauDemo dictionary...", flush=True)
    bases, vocab, chars_map, punct_list = load_daudemo_inventory()
    inv, build_s = build_v2_from_daudemo(bases, vocab, punct_list)
    print(f"v2 inventory built in {build_s:.1f}s: {inv.stats}", flush=True)

    model = build_model(inv)
    print("Building inference cache (148k candidates)...", flush=True)
    t0 = time.perf_counter()
    base_restorer = FastRestorerV2(model, inv, device="cpu", use_cache=True)
    cache_build_s = time.perf_counter() - t0
    print(f"inference cache built in {cache_build_s:.1f}s", flush=True)

    opt_restorer = FastRestorerV2Opt(model, inv, device="cpu", use_cache=False)

    # ---- memory accounting (the Exp-A blind spot) ----
    param_mb = sum(t.numel() * t.element_size() for t in model.parameters()) / 1024**2
    buf_mb = sum(t.numel() * t.element_size() for t in model.buffers()) / 1024**2
    cand_cache = getattr(model, "_cached_cand_vecs", None)
    cand_cache_mb = (cand_cache.numel() * cand_cache.element_size() / 1024**2) if cand_cache is not None else 0.0
    csr = CSRInventory(inv.cand_ids)
    mb = lambda x: round(x, 2)
    memory = {
        "param_mb": mb(param_mb),
        "buffer_mb": mb(buf_mb),
        "checkpoint_total_mb": mb(param_mb + buf_mb),
        "inference_cache_mb": mb(cand_cache_mb),
        "runtime_total_mb": mb(param_mb + buf_mb + cand_cache_mb),
        "exp_a_reported_buffer_mb": 42.72,
        "exp_a_reported_total_mb": 61.14,
        "csr_stats": csr.stats,
        "cache_build_seconds": round(cache_build_s, 1),
        "inventory_build_seconds": round(build_s, 1),
    }
    print("memory:", json.dumps(memory, indent=2), flush=True)

    # ---- breakdown on baseline (no word cache) ----
    breakdown_base = profile_predict_breakdown(base_restorer, SAMPLES[0], repeats=15, use_word_cache=False)
    print("baseline breakdown:", breakdown_base, flush=True)
    breakdown_opt = profile_predict_breakdown(opt_restorer, SAMPLES[0], repeats=15, use_word_cache=True)
    print("opt breakdown:", breakdown_opt, flush=True)

    # ---- end-to-end latency ----
    b_p50, b_p95 = lat_ms(lambda: base_restorer.predict(SAMPLES[0], decode_mode="viterbi"))
    o_p50, o_p95 = lat_ms(lambda: opt_restorer.predict(SAMPLES[0], decode_mode="viterbi"))
    bm_p50, bm_p95 = lat_ms(lambda: base_restorer.predict(SAMPLES[0], decode_mode="mbr"), warmup=2, reps=10)
    om_p50, om_p95 = lat_ms(lambda: opt_restorer.predict(SAMPLES[0], decode_mode="mbr"), warmup=2, reps=10)
    b_long, _ = lat_ms(lambda: base_restorer.predict(LONG_TEXT, decode_mode="viterbi", max_window=96), warmup=2, reps=10)
    o_long, _ = lat_ms(lambda: opt_restorer.predict(LONG_TEXT, decode_mode="viterbi", max_window=96), warmup=2, reps=10)

    # cold-start: clear word caches, first call cost (novel sentence realistic typing)
    opt_restorer.clear_caches()
    novel = "thu do ha noi mua thu dep nhat nam vua don don khach du lich quoc te dam dac nhat"
    t = time.perf_counter()
    opt_restorer.predict(novel)
    o_cold_first = round((time.perf_counter() - t) * 1000, 2)
    o_novel_p50, _ = lat_ms(lambda: opt_restorer.predict(novel), warmup=0, reps=10)

    # fresh-input latency: every rep uses a DIFFERENT sentence (realistic traffic),
    # so the whole-input memo never hits; word-level cache warm-up is amortized.
    fresh_sents = [
        f"cau so {i} noi ve {SAMPLES[i % len(SAMPLES)]} voi them {i} chi tiet moi" for i in range(30)
    ]
    opt_restorer.clear_caches()
    for s in fresh_sents[:3]:  # warm word-level caches only
        opt_restorer.predict(s)
    t = time.perf_counter()
    for s in fresh_sents:
        opt_restorer.predict(s)
    o_fresh_avg = round((time.perf_counter() - t) * 1000 / len(fresh_sents), 2)
    t = time.perf_counter()
    for s in fresh_sents:
        base_restorer.predict(s)
    b_fresh_avg = round((time.perf_counter() - t) * 1000 / len(fresh_sents), 2)

    # equivalence spot check on the real inventory
    same = all(
        base_restorer.predict(s)[0] == opt_restorer.predict(s)[0] for s in SAMPLES + [LONG_TEXT]
    )

    results = {
        "device": "cpu",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "num_threads": 2,
        "weights": "untrained (latency/memory profiling only)",
        "equivalence_ok": bool(same),
        "memory_accounting": memory,
        "breakdown_baseline": breakdown_base,
        "breakdown_opt": breakdown_opt,
        "latency_ms": {
            "baseline_viterbi_p50": b_p50, "baseline_viterbi_p95": b_p95,
            "opt_viterbi_p50": o_p50, "opt_viterbi_p95": o_p95,
            "baseline_mbr_p50": bm_p50, "opt_mbr_p50": om_p50,
            "long176w_baseline": b_long, "long176w_opt": o_long,
            "opt_novel_sentence_first_call": o_cold_first,
            "opt_novel_sentence_p50": o_novel_p50,
            "fresh_input_baseline_avg": b_fresh_avg,
            "fresh_input_opt_avg": o_fresh_avg,
        },
    }
    RESULTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
