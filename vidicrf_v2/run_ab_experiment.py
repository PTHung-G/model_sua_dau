"""Controlled A/B Benchmark: ViDiCRF v1 vs v1+MBR vs ViDiCRF v2.

Experiment A: Memory footprint, OOV copy overhead, and CPU inference latency
              on the real 1.38M-base DauDemo dictionary.
Experiment B: Fair end-to-end training & RAW evaluation on real Vietnamese
              Wikipedia (20231101.vi) article-disjoint splits.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
import time
import tracemalloc
import unicodedata

import numpy as np
import pyarrow.parquet as pq
import requests
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "charexp"))

import build_punct_tensors_v1 as b_v1
import sylmodel_punct as sm_v1
import syltrain_punct as st_v1
from eval_punct_vs_best import replace_spans as v1_replace_spans, restore_case as v1_restore_case, score_pairs

from vidicrf_v2.data_pipeline import (
    V2Inventory,
    build_char_tables,
    build_v2_inventory,
    encode_corpus_v2,
    encode_strings_chars,
    encode_strings_phono,
    make_batch_v2,
)
from vidicrf_v2.model import (
    ViDiCRFv2,
    compute_loss,
    trigram_viterbi,
)
from vidicrf_v2.orthography import (
    compact_inventory,
    has_diacritic,
    normalize_tone_placement,
    strip_diac,
)
from vidicrf_v2.restorer import FastRestorerV2
from vidicrf_v2.v1_mbr_decoder import mbr_decode

TOK = re.compile(r"[^\W_]+", re.UNICODE)
WORK = ROOT / "ab_exp_work"
RESULTS_OUT = ROOT / "lich_su_sua_doi" / "ket_qua_benchmark_v1_vs_v2.json"


def _strip(text: str) -> str:
    return strip_diac(text)


def _lexical_key(text: str) -> bytes:
    return hashlib.blake2b(
        " ".join(TOK.findall(text.lower())).encode("utf-8"), digest_size=16
    ).digest()


def _split_for(doc_id: str) -> str:
    h = int(hashlib.sha256(("punctuation-v1:" + doc_id).encode("utf-8")).hexdigest()[:8], 16) % 100
    return "test" if h < 2 else "validation" if h < 4 else "train"


def _tensor_bytes(tensors) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


# =============================================================================
# Experiment A: Real 1.38M DauDemo Inventory Memory, OOV Overhead, CPU Speed
# =============================================================================

def run_experiment_a(sample_sentences: list[str]) -> dict:
    print("\n=== EXPERIMENT A: Real DauDemo 1.38M Dictionary Memory & CPU Latency ===", flush=True)
    demo_dir = ROOT / "windows_demo_build" / "DauDemo"
    bases = json.loads((demo_dir / "bases.json").read_text(encoding="utf-8"))
    vocab = json.loads((demo_dir / "vocab.json").read_text(encoding="utf-8"))
    chars_map = json.loads((demo_dir / "chars.json").read_text(encoding="utf-8"))
    punct_list = json.loads((demo_dir / "punct.json").read_text(encoding="utf-8"))

    # Reconstruct v1 ctab from DauDemo bases & vocab (same logic as build_punct_tensors_v1)
    kmax = 32
    chw = 16
    bid_v1 = {b: i + 1 for i, b in enumerate(bases)}
    ctab_np = np.full((len(bases) + 1, kmax), -1, dtype=np.int32)
    row_fill = np.zeros(len(bases) + 1, dtype=np.int32)
    for vi, form in enumerate(vocab):
        bi = bid_v1.get(_strip(form).lower())
        if bi is not None and row_fill[bi] < kmax:
            ctab_np[bi, row_fill[bi]] = vi
            row_fill[bi] += 1

    nchar_v1 = max(chars_map.values()) + 1
    npunct_v1 = len(punct_list)
    pid_v1 = {g: i for i, g in enumerate(punct_list)}

    def enc_v1(strings):
        arr = np.zeros((len(strings), chw), dtype=np.int16)
        for i, s in enumerate(strings):
            for j in range(min(len(s), chw)):
                arr[i, j] = chars_map.get(s[j], 1)
        return torch.from_numpy(arr)

    t0 = time.perf_counter()
    v1_base_ch = enc_v1([""] + bases)
    v1_cand_ch = enc_v1(vocab)
    v1_cid = torch.from_numpy(ctab_np)
    v1_model = sm_v1.PunctSylCRF(
        nchar_v1, v1_base_ch, v1_cand_ch, v1_cid, npunct_v1,
        d=128, r=32, hidden=136, layers=2, drop=0.0,
    ).eval()
    v1_build_s = time.perf_counter() - t0

    v1_param_bytes = _tensor_bytes(v1_model.parameters())
    v1_buf_bytes = _tensor_bytes(v1_model.buffers())
    v1_param_count = sum(p.numel() for p in v1_model.parameters())

    # Measure v1 OOV torch.cat allocation spike
    tracemalloc.start()
    s1 = tracemalloc.take_snapshot()
    _cat_b = torch.cat((v1_model.base_ch, enc_v1(["antigravityx99"])))
    _cat_c = torch.cat((v1_model.cand_ch, enc_v1(["antigravityx99"])))
    _cat_id = torch.cat((v1_model.cand_ids, torch.full((1, kmax), -1, dtype=torch.int32)))
    s2 = tracemalloc.take_snapshot()
    tracemalloc.stop()
    v1_oov_alloc_mb = (_cat_b.numel() * _cat_b.element_size() + _cat_c.numel() * _cat_c.element_size() + _cat_id.numel() * _cat_id.element_size()) / (1024 ** 2)
    del _cat_b, _cat_c, _cat_id

    # Measure v1 CPU latency (2 threads, identical to DauDemo/server.py)
    torch.set_num_threads(2)

    @torch.inference_mode()
    def v1_predict_one(text: str) -> str:
        matches = list(TOK.finditer(text))
        if not matches:
            return text
        forms = [_strip(m.group()).lower() for m in matches]
        m = v1_model
        saved = (m.base_ch, m.cand_ch, m.cand_ids)
        new_forms, new_rows = [], []
        local_bid = dict()
        local_vocab_extra = []
        try:
            for form in forms:
                if form not in bid_v1 and form not in local_bid:
                    local_bid[form] = len(bases) + len(new_forms) + 1
                    row = torch.full((kmax,), -1, dtype=m.cand_ids.dtype)
                    row[0] = len(vocab) + len(local_vocab_extra)
                    local_vocab_extra.append(form)
                    new_forms.append(form)
                    new_rows.append(row)
            if new_forms:
                m.base_ch = torch.cat((m.base_ch, enc_v1(new_forms)))
                m.cand_ch = torch.cat((m.cand_ch, enc_v1(new_forms)))
                m.cand_ids = torch.cat((m.cand_ids, torch.stack(new_rows)))
            x = torch.tensor([[bid_v1.get(f, local_bid.get(f, 0)) for f in forms]])
            ln = torch.tensor([len(forms)])
            _, gaps = b_v1.gaps(text)
            seq = [p for p, _ in gaps] + [gaps[-1][1]]
            gg = torch.tensor([[pid_v1.get(g, 1) for g in seq]])
            ff = torch.tensor([[b_v1.case_id(mm.group()) for mm in matches]])
            e, tr, mask = m(x, ln, ff, gg)
            pred = sm_v1.viterbi(e, tr, mask)[0]
            words = []
            for i, (form, mm) in enumerate(zip(forms, matches)):
                if form not in bid_v1:
                    words.append(mm.group())
                else:
                    vi = int(m.cand_ids[bid_v1[form], int(pred[i])])
                    cand_w = vocab[vi]
                    words.append(v1_restore_case(cand_w, mm.group()) if len(cand_w) == len(mm.group()) else cand_w)
            return v1_replace_spans(text, words)
        finally:
            m.base_ch, m.cand_ch, m.cand_ids = saved

    for sent in sample_sentences[:10]:
        v1_predict_one(sent)
    v1_lat = []
    for sent in sample_sentences:
        t_s = time.perf_counter()
        v1_predict_one(sent)
        v1_lat.append((time.perf_counter() - t_s) * 1000.0)

    # Also test v1 latency on a sentence WITH an unseen OOV word
    oov_text = sample_sentences[0] + " AntigravityX99"
    oov_lats_v1 = []
    for _ in range(10):
        t_s = time.perf_counter()
        v1_predict_one(oov_text)
        oov_lats_v1.append((time.perf_counter() - t_s) * 1000.0)

    # Now build compacted v2 inventory from the exact same DauDemo dictionary
    t0 = time.perf_counter()
    new_bases, new_vocab, new_rows, new_counts, comp_stats = compact_inventory(
        bases, vocab, ctab_or_counts=ctab_np, style="classic",
        drop_identity_singletons=True, filter_invalid=True, kmax=kmax,
    )
    bid_v2 = {b: i + 1 for i, b in enumerate(new_bases)}
    vid_v2 = {v: i for i, v in enumerate(new_vocab)}
    v2_cid = torch.full((len(new_bases) + 1, kmax), -1, dtype=torch.int32)
    v2_priors = torch.zeros((len(new_bases) + 1, kmax, 3), dtype=torch.float16)
    for bi, (b, r_ids, r_cnts) in enumerate(zip(new_bases, new_rows, new_counts), 1):
        tot = sum(r_cnts)
        for k, (vi, cnt) in enumerate(zip(r_ids, r_cnts)):
            if vi < 0:
                break
            v2_cid[bi, k] = vi
            v2_priors[bi, k, 0] = math.log(cnt / max(1.0, tot) + 1e-6)
            v2_priors[bi, k, 1] = math.log1p(cnt) / 15.0
            v2_priors[bi, k, 2] = 1.0 if new_vocab[vi] == b else 0.0

    # Top 24,576 frequent syllables in lexical table
    lex_vocab_n = min(24576, len(new_vocab)) + 1
    cand_lex = torch.zeros(len(new_vocab), dtype=torch.int32)
    cand_has_diac = torch.zeros(len(new_vocab), dtype=torch.bool)
    for vi in range(min(24576, len(new_vocab))):
        cand_lex[vi] = vi + 1
    for vi, w in enumerate(new_vocab):
        cand_has_diac[vi] = has_diacritic(w)

    char_tables = build_char_tables(new_bases + new_vocab, min_char_count=10)
    c2i_v2 = char_tables["c2i"]
    v2_base_ch = encode_strings_chars([""] + new_bases, c2i_v2, chw)
    v2_base_ph = encode_strings_phono([""] + new_bases)
    v2_cand_ch = encode_strings_chars(new_vocab, c2i_v2, chw)
    v2_cand_ph = encode_strings_phono(new_vocab)

    inv_v2 = V2Inventory(
        bases=new_bases, bid=bid_v2, vocab=new_vocab, vid=vid_v2,
        cand_ids=v2_cid, cand_priors=v2_priors,
        base_ch=v2_base_ch, base_phono=v2_base_ph,
        cand_ch=v2_cand_ch, cand_phono=v2_cand_ph,
        cand_lex=cand_lex, cand_has_diac=cand_has_diac,
        n_lex_vocab=lex_vocab_n, punct=punct_list, pid=pid_v1,
        char_tables=char_tables, chw=chw, kmax=kmax,
        style="classic", normalize_orthography=True, stats=comp_stats,
    )
    v2_model = ViDiCRFv2(
        char_base_map=char_tables["char_base_map"],
        char_mod_map=char_tables["char_mod_map"],
        char_tone_map=char_tables["char_tone_map"],
        n_base_chars=char_tables["n_base_chars"],
        n_char_mods=char_tables["n_char_mods"],
        n_char_tones=char_tables["n_char_tones"],
        base_ch=inv_v2.base_ch,
        base_phono=inv_v2.base_phono,
        cand_ch=inv_v2.cand_ch,
        cand_phono=inv_v2.cand_phono,
        cand_lex=inv_v2.cand_lex,
        cand_ids=inv_v2.cand_ids,
        cand_priors=inv_v2.cand_priors,
        npunct=len(inv_v2.punct),
        n_lex_vocab=inv_v2.n_lex_vocab,
        d=128, n_heads=4, stage1_layers=2, stage2_layers=2,
        ffn_dim=256, lex_dim=64, r_big=32, r_tri=16, top_m=8, drop=0.0,
    ).eval()

    restorer_v2 = FastRestorerV2(v2_model, inv_v2, device="cpu", use_cache=True)
    v2_build_s = time.perf_counter() - t0
    v2_param_bytes = _tensor_bytes(v2_model.parameters())
    v2_buf_bytes = _tensor_bytes(v2_model.buffers())
    v2_param_count = sum(p.numel() for p in v2_model.parameters())

    for sent in sample_sentences[:10]:
        restorer_v2.predict(sent, decode_mode="viterbi")
    v2_lat_vit = []
    for sent in sample_sentences:
        t_s = time.perf_counter()
        restorer_v2.predict(sent, decode_mode="viterbi")
        v2_lat_vit.append((time.perf_counter() - t_s) * 1000.0)

    oov_lats_v2 = []
    ptrs_before = {n: b.data_ptr() for n, b in v2_model.named_buffers()}
    for _ in range(10):
        t_s = time.perf_counter()
        restorer_v2.predict(oov_text, decode_mode="viterbi")
        oov_lats_v2.append((time.perf_counter() - t_s) * 1000.0)
    ptrs_after = {n: b.data_ptr() for n, b in v2_model.named_buffers()}
    assert ptrs_before == ptrs_after

    res_a = {
        "v1": {
            "bases": len(bases),
            "vocab": len(vocab),
            "trainable_params": v1_param_count,
            "param_mb": round(v1_param_bytes / 1024**2, 2),
            "buffer_mb": round(v1_buf_bytes / 1024**2, 2),
            "total_model_mb": round((v1_param_bytes + v1_buf_bytes) / 1024**2, 2),
            "oov_realloc_mb": round(v1_oov_alloc_mb, 2),
            "cpu_p50_ms": round(float(np.median(v1_lat)), 2),
            "cpu_p95_ms": round(float(np.percentile(v1_lat, 95)), 2),
            "cpu_oov_p50_ms": round(float(np.median(oov_lats_v1)), 2),
        },
        "v2": {
            "bases": len(new_bases),
            "vocab": len(new_vocab),
            "compaction_stats": comp_stats,
            "trainable_params": v2_param_count,
            "param_mb": round(v2_param_bytes / 1024**2, 2),
            "buffer_mb": round(v2_buf_bytes / 1024**2, 2),
            "total_model_mb": round((v2_param_bytes + v2_buf_bytes) / 1024**2, 2),
            "oov_realloc_mb": 0.0,
            "cpu_p50_ms": round(float(np.median(v2_lat_vit)), 2),
            "cpu_p95_ms": round(float(np.percentile(v2_lat_vit, 95)), 2),
            "cpu_oov_p50_ms": round(float(np.median(oov_lats_v2)), 2),
        },
    }
    print(json.dumps(res_a, indent=2, ensure_ascii=False), flush=True)
    return res_a


# =============================================================================
# Experiment B: End-to-End Training & RAW Evaluation on Real Wikipedia 20231101.vi
# =============================================================================

def prepare_real_wikipedia_splits(
    n_train: int = 12000,
    n_val: int = 1000,
    n_test: int = 1000,
):
    WORK.mkdir(parents=True, exist_ok=True)
    split_cache = WORK / f"wiki_splits_{n_train}_{n_val}_{n_test}.json"
    if split_cache.exists():
        print("Loading cached Wikipedia splits:", split_cache, flush=True)
        return json.loads(split_cache.read_text(encoding="utf-8"))

    raw_dir = WORK / "raw_parquet"
    raw_dir.mkdir(exist_ok=True)
    session = requests.Session()
    meta = session.get("https://huggingface.co/api/datasets/wikimedia/wikipedia", timeout=60).json()
    revision = meta["sha"]
    listing = session.get(
        f"https://huggingface.co/api/datasets/wikimedia/wikipedia/tree/{revision}/20231101.vi",
        timeout=60,
    ).json()
    parquets = sorted([e for e in listing if e["path"].endswith(".parquet")], key=lambda x: x["path"])

    train_rows, val_rows, test_rows = [], [], []
    seen_eval = set()
    seen_train = set()

    for entry in parquets:
        dest = raw_dir / Path(entry["path"]).name
        if not dest.exists():
            url = f"https://huggingface.co/datasets/wikimedia/wikipedia/resolve/{revision}/{entry['path']}"
            print(f"Downloading {dest.name} ({entry['size']/1024**2:.1f} MB)...", flush=True)
            with session.get(url, stream=True, timeout=(30, 180)) as resp:
                resp.raise_for_status()
                with dest.open("wb") as f:
                    for chunk in resp.iter_content(2**20):
                        if chunk:
                            f.write(chunk)

        pf = pq.ParquetFile(dest)
        # First pass eval splits, second pass train to enforce evaluation-priority dedup
        docs_list = []
        for batch in pf.iter_batches(batch_size=256, columns=["id", "text"]):
            docs_list.extend(batch.to_pylist())

        for doc in docs_list:
            sp = _split_for(str(doc["id"]))
            if sp == "train":
                continue
            if (sp == "test" and len(test_rows) >= n_test) or (sp == "validation" and len(val_rows) >= n_val):
                continue
            for para in doc["text"].splitlines():
                target = unicodedata.normalize("NFC", para.strip())
                n = len(TOK.findall(target))
                if not 5 <= n <= 256:
                    continue
                if "\ufffd" in target or any(unicodedata.category(c) == "Cc" and c != "\t" for c in target):
                    continue
                if not has_diacritic(target):
                    continue
                k = _lexical_key(target).hex()
                if k in seen_eval:
                    continue
                src = _strip(target)
                if len(src) != len(target) or TOK.sub("", src) != TOK.sub("", target):
                    continue
                seen_eval.add(k)
                pair = [src, target]
                if sp == "test" and len(test_rows) < n_test:
                    test_rows.append(pair)
                elif sp == "validation" and len(val_rows) < n_val:
                    val_rows.append(pair)

        for doc in docs_list:
            if len(train_rows) >= n_train:
                break
            if _split_for(str(doc["id"])) != "train":
                continue
            for para in doc["text"].splitlines():
                if len(train_rows) >= n_train:
                    break
                target = unicodedata.normalize("NFC", para.strip())
                n = len(TOK.findall(target))
                if not 5 <= n <= 256:
                    continue
                if "\ufffd" in target or any(unicodedata.category(c) == "Cc" and c != "\t" for c in target):
                    continue
                if not has_diacritic(target):
                    continue
                k = _lexical_key(target).hex()
                if k in seen_eval or k in seen_train:
                    continue
                src = _strip(target)
                if len(src) != len(target) or TOK.sub("", src) != TOK.sub("", target):
                    continue
                seen_train.add(k)
                train_rows.append([src, target])

        print(
            f"Scanned {dest.name}: train={len(train_rows)}/{n_train}, val={len(val_rows)}/{n_val}, test={len(test_rows)}/{n_test}",
            flush=True,
        )
        if len(train_rows) >= n_train and len(val_rows) >= n_val and len(test_rows) >= n_test:
            break

    data = {"revision": revision, "train": train_rows, "validation": val_rows, "test": test_rows}
    split_cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def score_predictions_detailed(
    pairs: list[tuple[str, str]],
    predictions: list[str],
    ambiguous_bases: set[str],
) -> dict:
    raw = score_pairs(pairs, predictions)
    amb_tot = amb_ok = 0
    norm_tok_tot = norm_tok_ok = norm_sent_ok = 0

    for (src, tgt), pred in zip(pairs, predictions):
        tgt_nfc = unicodedata.normalize("NFC", tgt)
        pred_nfc = unicodedata.normalize("NFC", pred)
        g_toks = TOK.findall(tgt_nfc)
        p_toks = TOK.findall(pred_nfc)
        s_toks = TOK.findall(src)
        for st, gt, pt in zip(s_toks, g_toks, p_toks):
            b = _strip(st).lower()
            if b in ambiguous_bases:
                amb_tot += 1
                amb_ok += int(gt == pt)
            gn = normalize_tone_placement(gt, "classic")
            pn = normalize_tone_placement(pt, "classic")
            norm_tok_tot += 1
            norm_tok_ok += int(gn == pn)

        g_norm_str = v1_replace_spans(tgt_nfc, [normalize_tone_placement(w, "classic") for w in g_toks])
        p_norm_str = v1_replace_spans(pred_nfc, [normalize_tone_placement(w, "classic") for w in p_toks])
        if g_norm_str == p_norm_str:
            norm_sent_ok += 1

    return {
        "token_accuracy": round(raw["token_accuracy"] * 100.0, 2),
        "ambiguous_token_accuracy": round(amb_ok / max(1, amb_tot) * 100.0, 2),
        "ambiguous_tokens": amb_tot,
        "sentence_exact": round(raw["sentence_exact"] * 100.0, 2),
        "ortho_norm_token_accuracy": round(norm_tok_ok / max(1, norm_tok_tot) * 100.0, 2),
        "ortho_norm_sentence_exact": round(norm_sent_ok / max(1, len(pairs)) * 100.0, 2),
        "punctuation_preserved_rows": raw["input_punctuation_preserved_rows"],
    }


def make_partial_diac_inputs(
    pairs: list[tuple[str, str]], keep_frac: float = 0.40, seed: int = 999
) -> tuple[list[tuple[str, str]], list[list[bool]]]:
    rng = random.Random(seed)
    partial_pairs = []
    kept_masks = []
    for _, tgt in pairs:
        tgt_nfc = unicodedata.normalize("NFC", tgt)
        src_full = _strip(tgt_nfc)
        t_matches = list(TOK.finditer(tgt_nfc))
        s_matches = list(TOK.finditer(src_full))
        words = []
        km = []
        for sm, tm in zip(s_matches, t_matches):
            has_d = has_diacritic(tm.group())
            if has_d and rng.random() < keep_frac:
                words.append(tm.group())
                km.append(True)
            else:
                words.append(sm.group())
                km.append(False)
        partial_pairs.append((v1_replace_spans(src_full, words), tgt_nfc))
        kept_masks.append(km)
    return partial_pairs, kept_masks


def score_partial_diac(
    partial_pairs: list[tuple[str, str]],
    predictions: list[str],
    kept_masks: list[list[bool]],
) -> dict:
    unacc_tot = unacc_ok = acc_tot = acc_preserved = 0
    for (_, tgt), pred, km in zip(partial_pairs, predictions, kept_masks):
        g_toks = TOK.findall(tgt)
        p_toks = TOK.findall(unicodedata.normalize("NFC", pred))
        for gt, pt, is_kept in zip(g_toks, p_toks, km):
            if is_kept:
                acc_tot += 1
                acc_preserved += int(gt == pt)
            else:
                unacc_tot += 1
                unacc_ok += int(gt == pt)
    return {
        "unaccented_60pct_token_accuracy": round(unacc_ok / max(1, unacc_tot) * 100.0, 2),
        "accented_40pct_preserved_rate": round(acc_preserved / max(1, acc_tot) * 100.0, 2),
        "accented_40pct_broken_tokens": acc_tot - acc_preserved,
    }


def train_and_eval_v1(
    train_pairs: list[tuple[str, str]],
    val_pairs: list[tuple[str, str]],
    test_pairs: list[tuple[str, str]],
    partial_test_pairs: list[tuple[str, str]],
    kept_masks: list[list[bool]],
    ambiguous_bases: set[str],
    device: str,
    epochs: int = 4,
    bs: int = 32,
    seed: int = 42,
) -> tuple[dict, dict, dict]:
    print("\n--- Training ViDiCRF v1 (PunctSylCRF Baseline) ---", flush=True)
    v1_dir = WORK / "v1_data"
    v1_dir.mkdir(parents=True, exist_ok=True)
    tr_jsonl = v1_dir / "train.jsonl"
    va_jsonl = v1_dir / "val.jsonl"
    tr_jsonl.write_text("".join(json.dumps({"input": s, "target": t}, ensure_ascii=False) + "\n" for s, t in train_pairs), encoding="utf-8")
    va_jsonl.write_text("".join(json.dumps({"input": s, "target": t}, ensure_ascii=False) + "\n" for s, t in val_pairs), encoding="utf-8")

    bases, vocab, ctab, pid, _ = b_v1.build_table(str(tr_jsonl))
    b_v1.encode(str(tr_jsonl), str(v1_dir / "train"), bases, vocab, ctab, pid)
    b_v1.encode(str(va_jsonl), str(v1_dir / "val"), bases, vocab, ctab, pid)
    (v1_dir / "manifest.json").write_text(json.dumps({"pipeline_schema": 2}), encoding="utf-8")

    # Note: min_char_count=10 to keep rare VN chars on 12k subset fairly
    nchar, base_ch, cand_ch, cid, _ = st_v1.build_tensors(str(v1_dir / "train"), min_char_count=10)
    chars_cnt = Counter()
    for w in bases + vocab:
        chars_cnt.update(w)
    c2i_v1 = {c: i + 2 for i, c in enumerate(sorted(c for c, n in chars_cnt.items() if n >= 10))}
    bid_v1 = {b: i + 1 for i, b in enumerate(bases)}

    train_data = st_v1.load_data(str(v1_dir / "train"))
    val_data = st_v1.load_data(str(v1_dir / "val"))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)

    model = sm_v1.PunctSylCRF(
        nchar, base_ch, cand_ch, cid, len(pid), d=128, hidden=136, r=32, layers=2
    ).to(device)
    starts, lengths = st_v1.segments(train_data[2], train_data[3], st_v1.MAXL)
    steps_per_epoch = math.ceil(len(starts) / bs)
    total_steps = max(1, steps_per_epoch * epochs)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: st_v1.lr_factor(s, total_steps))

    best_acc = -1.0
    best_state = None
    t_train_start = time.perf_counter()

    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        n_steps = 0
        for idx in st_v1.epoch_batches(lengths, bs, rng):
            x, y, ln, ff, gg = st_v1.make_batch(train_data, starts, lengths, idx, device)
            e, tr, mask = model(x, ln, ff, gg)
            loss = sm_v1.nll(e, tr, mask, y, ln)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            ep_loss += float(loss.detach())
            n_steps += 1
        val_acc, _ = st_v1.evaluate(model, val_data, bs)
        print(f"  [v1] epoch {ep}/{epochs} loss={ep_loss/max(1,n_steps):.4f} val_acc={val_acc*100:.2f}%", flush=True)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    train_secs = time.perf_counter() - t_train_start
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()

    # RAW test evaluation for v1 (Viterbi and FB-MBR)
    def encode_v1_strings(strings, width):
        return torch.tensor(
            [[c2i_v1.get(c, 1) for c in s[:width]] + [0] * max(0, width - len(s)) for s in strings],
            dtype=model.base_ch.dtype, device=device,
        )

    @torch.inference_mode()
    def predict_v1(source: str, mode: str = "viterbi") -> str:
        matches = list(TOK.finditer(source))
        if not matches:
            return source
        forms = [_strip(m.group()).lower() for m in matches]
        saved = (model.base_ch, model.cand_ch, model.cand_ids)
        local_bid = {}
        new_forms, new_rows = [], []
        extra_vocab = []
        try:
            for form in forms:
                if form not in bid_v1 and form not in local_bid:
                    local_bid[form] = len(bases) + len(new_forms) + 1
                    row = torch.full((model.cand_ids.shape[1],), -1, dtype=model.cand_ids.dtype, device=device)
                    row[0] = len(vocab) + len(extra_vocab)
                    extra_vocab.append(form)
                    new_forms.append(form)
                    new_rows.append(row)
            if new_forms:
                model.base_ch = torch.cat((model.base_ch, encode_v1_strings(new_forms, model.base_ch.shape[-1])))
                model.cand_ch = torch.cat((model.cand_ch, encode_v1_strings(new_forms, model.cand_ch.shape[-1])))
                model.cand_ids = torch.cat((model.cand_ids, torch.stack(new_rows)))
            x = torch.tensor([[bid_v1.get(f, local_bid.get(f, 0)) for f in forms]], device=device)
            ln = torch.tensor([len(forms)], device=device)
            _, gs = b_v1.gaps(source)
            seq = [p for p, _ in gs] + [gs[-1][1]]
            gg = torch.tensor([[pid.get(g, 1) for g in seq]], device=device)
            ff = torch.tensor([[b_v1.case_id(mm.group()) for mm in matches]], device=device)
            e, tr, mask = model(x, ln, ff, gg)
            if mode == "viterbi":
                pred = sm_v1.viterbi(e, tr, mask)[0]
            else:
                pred, _ = mbr_decode(e, tr, mask, marginal_weight=1.0)
                pred = pred[0]
            words = []
            for i, (form, mm) in enumerate(zip(forms, matches)):
                if form not in bid_v1:
                    words.append(mm.group())
                else:
                    vi = int(model.cand_ids[bid_v1[form], int(pred[i])])
                    cand_w = vocab[vi]
                    words.append(v1_restore_case(cand_w, mm.group()) if len(cand_w) == len(mm.group()) else cand_w)
            return v1_replace_spans(source, words)
        finally:
            model.base_ch, model.cand_ch, model.cand_ids = saved

    # Compute baselines on the exact same train inventory + RAW test
    tf_lookup = {}
    for i, b in enumerate(bases, 1):
        tf_lookup[b] = vocab[int(ctab[i, 0])]
    copy_preds = [s for s, _ in test_pairs]
    tf_preds = []
    for s, _ in test_pairs:
        ws = []
        for mm in TOK.finditer(s):
            f = _strip(mm.group()).lower()
            tf_w = tf_lookup.get(f, f)
            ws.append(v1_restore_case(tf_w, mm.group()) if len(tf_w) == len(mm.group()) else tf_w)
        tf_preds.append(v1_replace_spans(s, ws))

    v1_vit_preds = [predict_v1(s, "viterbi") for s, _ in test_pairs]
    v1_mbr_preds = [predict_v1(s, "mbr") for s, _ in test_pairs]
    v1_partial_preds = [predict_v1(s, "viterbi") for s, _ in partial_test_pairs]

    ckpt_path = WORK / "v1_best.pt"
    torch.save({"model": best_state}, ckpt_path)

    baselines = {
        "COPY": score_predictions_detailed(test_pairs, copy_preds, ambiguous_bases),
        "Train-frequency": score_predictions_detailed(test_pairs, tf_preds, ambiguous_bases),
    }
    v1_vit_res = {
        "trainable_params": sum(p.numel() for p in model.parameters()),
        "checkpoint_mb": round(ckpt_path.stat().st_size / 1024**2, 2),
        "train_seconds": round(train_secs, 1),
        "best_val_acc_pct": round(best_acc * 100.0, 2),
        **score_predictions_detailed(test_pairs, v1_vit_preds, ambiguous_bases),
        "partial_diacritic_40pct": score_partial_diac(partial_test_pairs, v1_partial_preds, kept_masks),
    }
    v1_mbr_res = {
        "note": "Zero-retrain upgrade: identical v1 weights, Forward-Backward MBR decoding",
        **score_predictions_detailed(test_pairs, v1_mbr_preds, ambiguous_bases),
    }
    return baselines, v1_vit_res, v1_mbr_res


def train_and_eval_v2(
    train_pairs: list[tuple[str, str]],
    val_pairs: list[tuple[str, str]],
    test_pairs: list[tuple[str, str]],
    partial_test_pairs: list[tuple[str, str]],
    kept_masks: list[list[bool]],
    ambiguous_bases: set[str],
    device: str,
    epochs: int = 4,
    bs: int = 32,
    seed: int = 42,
) -> tuple[dict, dict]:
    print("\n--- Training ViDiCRF v2 (Hybrid BiGRU + Conformer-RoPE + AnchorProp + Top-8 Trigram CRF) ---", flush=True)
    inv = build_v2_inventory(
        train_pairs, max_lex_vocab=16384, min_lex_count=2, kmax=32, chw=20,
        min_char_count=10, style="classic", normalize_orthography=True,
        drop_identity_singletons=True, filter_invalid=True,
    )
    print(f"  [v2] inventory stats: {inv.stats}", flush=True)

    train_enc = encode_corpus_v2(train_pairs, inv, max_len=128, halo=16)
    val_enc = encode_corpus_v2(val_pairs, inv, max_len=256, halo=16)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)

    ct = inv.char_tables
    model = ViDiCRFv2(
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
        ffn_dim=256, lex_dim=64, r_big=32, r_tri=16, top_m=8, drop=0.10,
    ).to(device)

    lengths = train_enc.seg_lens
    steps_per_epoch = math.ceil(len(lengths) / bs)
    total_steps = max(1, steps_per_epoch * epochs)
    no_decay_ids = set()
    for mod in model.modules():
        if isinstance(mod, (torch.nn.Embedding, torch.nn.LayerNorm)):
            for p in mod.parameters(recurse=False):
                no_decay_ids.add(id(p))
    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in no_decay_ids or name.endswith(".bias") or p.ndim < 2:
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    opt = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": 0.01},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=0.0020,
    )
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: st_v1.lr_factor(s, total_steps))

    @torch.no_grad()
    def eval_v2_val() -> float:
        model.eval()
        order = np.argsort(val_enc.seg_lens, kind="stable")
        ok = tot = 0
        for pos in range(0, len(order), bs):
            idx = order[pos : pos + bs]
            b = make_batch_v2(val_enc, inv, idx, device, partial_diac_prob=0.0)
            out = model(
                b["base_ids"], b["sent_len"], b["case_ids"], b["left_gaps"],
                b["right_gaps"], b["surface_chars"], b["surface_phono"],
                b["has_input_diac"], b["surface_lex"],
            )
            pred_p = trigram_viterbi(out["emit_p"], out["bigram"], out["trigram"], out["mask_p"])
            pred_orig = out["top_idx"].gather(-1, pred_p.unsqueeze(-1)).squeeze(-1)
            valid = out["valid_tok"]
            ok += int(((pred_orig == b["gold"]) & valid).sum())
            tot += int(b["sent_len"].sum())
        return ok / max(1, tot)

    best_acc = -1.0
    best_state = None
    t_train_start = time.perf_counter()

    for ep in range(1, epochs + 1):
        model.train()
        model.invalidate_cache()
        ep_loss = 0.0
        n_steps = 0
        for idx in st_v1.epoch_batches(lengths, bs, rng):
            b = make_batch_v2(train_enc, inv, idx, device, partial_diac_prob=0.15, rng=rng)
            out = model(
                b["base_ids"], b["sent_len"], b["case_ids"], b["left_gaps"],
                b["right_gaps"], b["surface_chars"], b["surface_phono"],
                b["has_input_diac"], b["surface_lex"], gold=b["gold"],
            )
            loss, _ = compute_loss(
                out, b["gold"], b["sent_len"], margin=0.10,
                lambda_emit2=0.35, lambda_emit1=0.25, core_mask=b["core_mask"],
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            ep_loss += float(loss.detach())
            n_steps += 1
        val_acc = eval_v2_val()
        print(f"  [v2] epoch {ep}/{epochs} loss={ep_loss/max(1,n_steps):.4f} val_acc={val_acc*100:.2f}%", flush=True)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    train_secs = time.perf_counter() - t_train_start
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    restorer = FastRestorerV2(model, inv, device=device, use_cache=True)

    ckpt_path = WORK / "v2_best.pt"
    torch.save({"model": best_state}, ckpt_path)

    v2_vit_preds = [restorer.predict(s, decode_mode="viterbi")[0] for s, _ in test_pairs]
    v2_mbr_preds = [restorer.predict(s, decode_mode="mbr")[0] for s, _ in test_pairs]
    v2_partial_preds = [
        restorer.predict(s, decode_mode="viterbi", preserve_input_diacritics=True)[0]
        for s, _ in partial_test_pairs
    ]

    v2_vit_res = {
        "trainable_params": sum(p.numel() for p in model.parameters()),
        "checkpoint_mb": round(ckpt_path.stat().st_size / 1024**2, 2),
        "train_seconds": round(train_secs, 1),
        "best_val_acc_pct": round(best_acc * 100.0, 2),
        "inventory_stats": inv.stats,
        **score_predictions_detailed(test_pairs, v2_vit_preds, ambiguous_bases),
        "partial_diacritic_40pct": score_partial_diac(partial_test_pairs, v2_partial_preds, kept_masks),
    }
    v2_mbr_res = {
        **score_predictions_detailed(test_pairs, v2_mbr_preds, ambiguous_bases),
    }
    return v2_vit_res, v2_mbr_res


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running A/B Experiment on device={device}", flush=True)
    splits = prepare_real_wikipedia_splits(n_train=12000, n_val=1000, n_test=1000)
    train_pairs = [(s, t) for s, t in splits["train"]]
    val_pairs = [(s, t) for s, t in splits["validation"]]
    test_pairs = [(s, t) for s, t in splits["test"]]

    # Identify bases that have >= 2 distinct spellings in train (for Ambiguous-Token Accuracy)
    form_sets = defaultdict(set)
    for _, tgt in train_pairs:
        for tok in TOK.findall(unicodedata.normalize("NFC", tgt)):
            form_sets[_strip(tok).lower()].add(tok.lower())
    ambiguous_bases = {b for b, forms in form_sets.items() if len(forms) >= 2}

    partial_test_pairs, kept_masks = make_partial_diac_inputs(test_pairs, keep_frac=0.40, seed=999)

    # Experiment A uses 100 real stripped Wikipedia test paragraphs (length <= 256 words)
    exp_a_sentences = [s for s, _ in test_pairs[:100]]
    res_a = run_experiment_a(exp_a_sentences)

    # Experiment B trains v1 and v2 on identical splits on GPU (or CPU)
    baselines, v1_vit_res, v1_mbr_res = train_and_eval_v1(
        train_pairs, val_pairs, test_pairs, partial_test_pairs, kept_masks,
        ambiguous_bases, device=device, epochs=4, bs=32, seed=42,
    )
    v2_vit_res, v2_mbr_res = train_and_eval_v2(
        train_pairs, val_pairs, test_pairs, partial_test_pairs, kept_masks,
        ambiguous_bases, device=device, epochs=4, bs=32, seed=42,
    )

    report = {
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "dataset": {
            "source": "wikimedia/wikipedia 20231101.vi",
            "revision": splits["revision"],
            "train_rows": len(train_pairs),
            "validation_rows": len(val_pairs),
            "test_rows": len(test_pairs),
            "test_tokens": sum(len(TOK.findall(t)) for _, t in test_pairs),
            "ambiguous_bases_in_train": len(ambiguous_bases),
        },
        "experiment_a_daudemo_1_38m_inventory": res_a,
        "experiment_b_end_to_end_ab": {
            "COPY": baselines["COPY"],
            "Train-frequency": baselines["Train-frequency"],
            "ViDiCRF_v1_Viterbi": v1_vit_res,
            "ViDiCRF_v1_FB_MBR_ZeroRetrain": v1_mbr_res,
            "ViDiCRF_v2_Viterbi": v2_vit_res,
            "ViDiCRF_v2_FB_MBR": v2_mbr_res,
        },
    }
    RESULTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== FINAL BENCHMARK REPORT ===", flush=True)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
