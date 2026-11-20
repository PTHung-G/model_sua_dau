# =============================================================================
# ViDiCRF v3 — Scale-up + (optional) train-only PMI ablation on Kaggle GPU
#
# Protocol identical to vidicrf_v2/run_ab_experiment.py Experiment B (seed 42,
# 4 epochs, bs 32, lr 0.0020 cosine+warmup, val-based best checkpoint, RAW test,
# eval-priority dedup, salt "punctuation-v1:"), with:
#   - N_TRAIN scaled up (default 200,000 paragraphs), pinned HF revision
#   - optional FP16/BF16 autocast on the encoder backbone (CRF losses stay FP32)
#   - optional train-only bigram PMI injection into CRF transitions (VARIANT="pmi")
#   - decode modes on test: Viterbi / FB-MBR / two-pass anchored (tau grid)
# Outputs: results_v3.json, v2_best.pt, inventory bundle
# =============================================================================
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

VARIANT = os.environ.get("VARIANT", "pmi")  # "base" | "pmi"  (sed per kernel copy)
N_TRAIN = int(os.environ.get("N_TRAIN", "12000"))
N_VAL = int(os.environ.get("N_VAL", "1000"))
N_TEST = int(os.environ.get("N_TEST", "1000"))
EPOCHS = int(os.environ.get("EPOCHS", "4"))
BS = int(os.environ.get("BS", "32"))
SEED = 42
SMOKE = os.environ.get("SMOKE", "") == "1"
HF_REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"  # pinned 20231101.vi snapshot
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(__file__).resolve().parent
SMOKE_WORK = Path(__file__).resolve().parent / "smoke_work"
if SMOKE:  # only for local smoke runs; /kaggle/src is read-only on Kaggle
    SMOKE_WORK.mkdir(parents=True, exist_ok=True)
OUT_JSON = (SMOKE_WORK if SMOKE else WORK) / "results_v3.json"

print(f"[env] torch={torch.__version__} cuda={torch.cuda.is_available()}"
      f" gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}"
      f" variant={VARIANT} n_train={N_TRAIN} smoke={SMOKE}", flush=True)

# --- import source: prefer attached dataset, fall back to repo layout (local smoke) ---
if Path("/kaggle/working").exists():
    # Kaggle mounts attached datasets at /kaggle/input/<slug> (legacy) or
    # /kaggle/input/datasets/<owner>/<slug> (new layout) — scan for both.
    SRC = None
    for cand_src in [
        Path("/kaggle/input/vidicrf-v3-src"),
        Path("/kaggle/input/datasets/tunhngphm/vidicrf-v3-src"),
    ]:
        if (cand_src / "model.py").exists() or (cand_src / "vidicrf_v2.zip").exists():
            SRC = cand_src
            break
    if SRC is None:
        hits = list(Path("/kaggle/input").glob("**/vidicrf-v3-src/model.py"))
        SRC = hits[0].parent if hits else None
    DST_PARENT = Path("/kaggle/working/src")
    if SRC is not None:
        if (SRC / "model.py").exists() and not (SRC / "vidicrf_v2" / "model.py").exists():
            # dataset uploaded with --dir-mode zip: files unpacked flat at dataset root
            pkg = DST_PARENT / "vidicrf_v2"
            pkg.mkdir(parents=True, exist_ok=True)
            for f in SRC.glob("*.py"):
                shutil.copy(f, pkg / f.name)
            print("[src] assembled flat dataset into package layout", flush=True)
        elif (SRC / "vidicrf_v2" / "model.py").exists():
            shutil.copytree(SRC / "vidicrf_v2", DST_PARENT / "vidicrf_v2", dirs_exist_ok=True)
            print("[src] copied dataset folder", flush=True)
        elif (SRC / "vidicrf_v2.zip").exists():
            DST_PARENT.mkdir(parents=True, exist_ok=True)
            shutil.unpack_archive(SRC / "vidicrf_v2.zip", DST_PARENT)
            print("[src] unpacked vidicrf_v2.zip", flush=True)
    if (DST_PARENT / "vidicrf_v2" / "model.py").exists():
        sys.path.insert(0, str(DST_PARENT))
        print("[src] using attached dataset source", flush=True)
for cand in [
    Path(__file__).resolve().parent,
    Path(__file__).resolve().parent.parent,
    Path(__file__).resolve().parent.parent.parent,
    Path("/kaggle/working/src"),
]:
    if (cand / "vidicrf_v2" / "model.py").exists():
        sys.path.insert(0, str(cand))
        print(f"[src] using {cand}", flush=True)
        break

from vidicrf_v2.data_pipeline import (  # noqa: E402
    TOK, case_id, encode_strings_chars, encode_strings_phono, extract_gaps,
    build_v2_inventory, encode_corpus_v2, make_batch_v2,
)
from vidicrf_v2.model import ViDiCRFv2, compute_loss, trigram_marginals, trigram_viterbi  # noqa: E402
from vidicrf_v2.orthography import has_diacritic, normalize_tone_placement, strip_diac  # noqa: E402
from vidicrf_v2.restorer import FastRestorerV2, replace_spans, restore_case  # noqa: E402
from vidicrf_v2.fast_inference import get_scripted_viterbi  # noqa: E402


def _replace_words(source: str, words) -> str:
    """v1-style replace_spans: tokenize source internally."""
    matches = list(TOK.finditer(source))
    if len(matches) != len(words):
        raise ValueError("Prediction/source token count mismatch")
    chunks, end = [], 0
    for match, word in zip(matches, words):
        chunks.extend((source[end:match.start()], word))
        end = match.end()
    return "".join(chunks) + source[end:]


def _strip(t):
    return strip_diac(t)


def _lexical_key(text: str) -> bytes:
    return hashlib.blake2b(
        " ".join(TOK.findall(text.lower())).encode("utf-8"), digest_size=16
    ).digest()


def _split_for(doc_id: str) -> str:
    h = int(hashlib.sha256(("punctuation-v1:" + doc_id).encode("utf-8")).hexdigest()[:8], 16) % 100
    return "test" if h < 2 else "validation" if h < 4 else "train"


def lr_factor(update, total):
    warm = max(1, total // 20)
    if update < warm:
        return (update + 1) / warm
    progress = min(1.0, (update - warm) / max(1, total - warm - 1))
    return 0.05 + 0.95 * (1 + math.cos(math.pi * progress)) / 2


def epoch_batches(lengths, bs, rng, block_batches=32):
    order = rng.permutation(len(lengths))
    block = bs * block_batches
    for pos in range(0, len(order), block):
        sub = order[pos:pos + block]
        sub = sub[np.argsort(lengths[sub], kind="stable")]
        for j in range(0, len(sub), bs):
            yield sub[j:j + bs]


# =============================================================================
# 1) Splits (verbatim split semantics from run_ab_experiment, eval-priority)
# =============================================================================
def prepare_splits(n_train, n_val, n_test):
    if SMOKE:
        return synthetic_splits(n_train, n_val, n_test)
    import pyarrow.parquet as pq
    import requests

    raw_dir = WORK / "raw_parquet"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache = WORK / f"wiki_splits_{n_train}_{n_val}_{n_test}.json"
    if cache.exists():
        print("Loading cached splits", flush=True)
        return json.loads(cache.read_text(encoding="utf-8"))

    session = requests.Session()
    listing = session.get(
        f"https://huggingface.co/api/datasets/wikimedia/wikipedia/tree/{HF_REVISION}/20231101.vi",
        timeout=60,
    ).json()
    files = sorted([e for e in listing if e["path"].endswith(".parquet")], key=lambda x: x["path"])
    print(f"[data] {len(files)} parquet files at revision {HF_REVISION[:8]}", flush=True)

    train_rows, val_rows, test_rows = [], [], []
    seen_eval, seen_train = set(), set()
    for entry in files:
        dest = raw_dir / Path(entry["path"]).name
        if not dest.exists():
            url = f"https://huggingface.co/datasets/wikimedia/wikipedia/resolve/{HF_REVISION}/{entry['path']}"
            print(f"Downloading {dest.name} ({entry['size']/1024**2:.1f} MB)...", flush=True)
            with session.get(url, stream=True, timeout=(30, 300)) as resp:
                resp.raise_for_status()
                with dest.open("wb") as f:
                    for chunk in resp.iter_content(2**20):
                        if chunk:
                            f.write(chunk)
        pf = pq.ParquetFile(dest)
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
                if sp == "test" and len(test_rows) < n_test:
                    test_rows.append([src, target])
                elif sp == "validation" and len(val_rows) < n_val:
                    val_rows.append([src, target])
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
        print(f"Scanned {dest.name}: train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)}", flush=True)
        if len(train_rows) >= n_train and len(val_rows) >= n_val and len(test_rows) >= n_test:
            break
    data = {"revision": HF_REVISION, "train": train_rows, "validation": val_rows, "test": test_rows}
    cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def synthetic_splits(n_train, n_val, n_test):
    """Deterministic tiny corpus for SMOKE mode (no network)."""
    words = [
        ("ngan hang", "ngân hàng"), ("kinh te", "kinh tế"), ("chu nghia", "chủ nghĩa"),
        ("ban ghe", "bàn ghế"), ("hoc sinh", "học sinh"), ("thi truong", "thị trường"),
        ("nguoi dan", "người dân"), ("chinh sach", "chính sách"), ("phat trien", "phát triển"),
        ("giao duc", "giáo dục"), ("y te", "y tế"), ("van hoa", "văn hóa"),
        ("xa hoi", "xã hội"), ("cong nghe", "công nghệ"), ("vu khi", "vũ khí"),
        ("dien tu", "điện tử"), ("hai san", "hải sản"), ("nong nghiep", "nông nghiệp"),
    ]
    rng = random.Random(7)
    rows = []
    for i in range(n_train + n_val + n_test + 50):
        ws = [words[rng.randrange(len(words))] for _ in range(rng.randint(8, 20))]
        tgt = " ".join(w[1] for w in ws)
        src = " ".join(w[0] for w in ws)
        rows.append([src, tgt])
    return {
        "revision": "synthetic-smoke",
        "train": rows[:n_train],
        "validation": rows[n_train:n_train + n_val],
        "test": rows[n_train + n_val:n_train + n_val + n_test],
    }


# =============================================================================
# 2) Scoring (mirrors run_ab_experiment, self-contained)
# =============================================================================
def score_pairs_detailed(pairs, predictions, ambiguous_bases):
    tot = ok = sent_ok = amb_tot = amb_ok = norm_ok = norm_tot = norm_sent = 0
    punct_rows = 0
    for (src, tgt), pred in zip(pairs, predictions):
        tgt_nfc = unicodedata.normalize("NFC", tgt)
        pred_nfc = unicodedata.normalize("NFC", pred)
        g, p, s = TOK.findall(tgt_nfc), TOK.findall(pred_nfc), TOK.findall(src)
        tot += len(g)
        ok += sum(a == b for a, b in zip(g, p))
        sent_ok += int(tgt_nfc == pred_nfc)
        punct_rows += int(TOK.sub("", src) == TOK.sub("", pred))
        for st, gt, pt in zip(s, g, p):
            if _strip(st).lower() in ambiguous_bases:
                amb_tot += 1
                amb_ok += int(gt == pt)
            norm_tot += 1
            norm_ok += int(
                normalize_tone_placement(gt, "classic") == normalize_tone_placement(pt, "classic")
            )
        gs = _replace_words(tgt_nfc, [normalize_tone_placement(w, "classic") for w in g])
        ps = _replace_words(pred_nfc, [normalize_tone_placement(w, "classic") for w in p])
        norm_sent += int(gs == ps)
    return {
        "token_accuracy": round(ok / max(1, tot) * 100.0, 2),
        "ambiguous_token_accuracy": round(amb_ok / max(1, amb_tot) * 100.0, 2),
        "ambiguous_tokens": amb_tot,
        "sentence_exact": round(sent_ok / max(1, len(pairs)) * 100.0, 2),
        "ortho_norm_token_accuracy": round(norm_ok / max(1, norm_tot) * 100.0, 2),
        "ortho_norm_sentence_exact": round(norm_sent / max(1, len(pairs)) * 100.0, 2),
        "punctuation_preserved_rows": punct_rows,
    }


def make_partial_diac_inputs(pairs, keep_frac=0.40, seed=999):
    rng = random.Random(seed)
    out, masks = [], []
    for _, tgt in pairs:
        tgt_nfc = unicodedata.normalize("NFC", tgt)
        src_full = _strip(tgt_nfc)
        tm, sm = TOK.findall(tgt_nfc), TOK.findall(src_full)
        words, km = [], []
        for st, tt in zip(sm, tm):
            if has_diacritic(tt) and rng.random() < keep_frac:
                words.append(tt)
                km.append(True)
            else:
                words.append(st)
                km.append(False)
        out.append((_replace_words(src_full, words), tgt_nfc))
        masks.append(km)
    return out, masks


def score_partial(pairs, predictions, kept_masks):
    ut = uo = at = ap = 0
    for (_, tgt), pred, km in zip(pairs, predictions, kept_masks):
        g = TOK.findall(tgt)
        p = TOK.findall(unicodedata.normalize("NFC", pred))
        for gt, pt, kept in zip(g, p, km):
            if kept:
                at += 1
                ap += int(gt == pt)
            else:
                ut += 1
                uo += int(gt == pt)
    return {
        "unaccented_60pct_token_accuracy": round(uo / max(1, ut) * 100.0, 2),
        "accented_40pct_preserved_rate": round(ap / max(1, at) * 100.0, 2),
        "accented_40pct_broken_tokens": at - ap,
    }


# =============================================================================
# 3) Main
# =============================================================================
def main():
    device = "cpu" if SMOKE else ("cuda" if torch.cuda.is_available() else "cpu")
    n_train, n_val, n_test = (60, 8, 8) if SMOKE else (N_TRAIN, N_VAL, N_TEST)
    epochs = 1 if SMOKE else EPOCHS

    t_data0 = time.perf_counter()
    splits = prepare_splits(n_train, n_val, n_test)
    train_pairs = [(s, t) for s, t in splits["train"]]
    val_pairs = [(s, t) for s, t in splits["validation"]]
    test_pairs = [(s, t) for s, t in splits["test"]]
    print(f"[data] splits ready in {time.perf_counter()-t_data0:.0f}s "
          f"(train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)})", flush=True)
    print(f"[data] revision={splits['revision']} test_tokens={sum(len(TOK.findall(t)) for _, t in test_pairs)}", flush=True)

    form_sets = defaultdict(set)
    for _, tgt in train_pairs:
        for tok in TOK.findall(unicodedata.normalize("NFC", tgt)):
            form_sets[_strip(tok).lower()].add(tok.lower())
    ambiguous_bases = {b for b, forms in form_sets.items() if len(forms) >= 2}

    partial_pairs, kept_masks = make_partial_diac_inputs(test_pairs, 0.40, 999)

    print("[inv] building inventory...", flush=True)
    t0 = time.perf_counter()
    inv = build_v2_inventory(
        train_pairs, max_lex_vocab=16384, min_lex_count=2, kmax=32, chw=20,
        min_char_count=10, style="classic", normalize_orthography=True,
        drop_identity_singletons=True, filter_invalid=True,
    )
    print(f"[inv] built in {time.perf_counter()-t0:.0f}s: {inv.stats}", flush=True)

    print("[enc] encoding corpora...", flush=True)
    t0 = time.perf_counter()
    train_enc = encode_corpus_v2(train_pairs, inv, max_len=128, halo=16)
    val_enc = encode_corpus_v2(val_pairs, inv, max_len=256, halo=16)
    print(f"[enc] encoded in {time.perf_counter()-t0:.0f}s: "
          f"train_segs={len(train_enc.seg_lens)} val_segs={len(val_enc.seg_lens)} "
          f"kept={train_enc.kept_rows} skipped={train_enc.skipped_rows}", flush=True)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device == "cuda":
        torch.cuda.manual_seed_all(SEED)
    rng = np.random.default_rng(SEED)

    ct = inv.char_tables
    model = ViDiCRFv2(
        char_base_map=ct["char_base_map"], char_mod_map=ct["char_mod_map"],
        char_tone_map=ct["char_tone_map"], n_base_chars=ct["n_base_chars"],
        n_char_mods=ct["n_char_mods"], n_char_tones=ct["n_char_tones"],
        base_ch=inv.base_ch, base_phono=inv.base_phono, cand_ch=inv.cand_ch,
        cand_phono=inv.cand_phono, cand_lex=inv.cand_lex, cand_ids=inv.cand_ids,
        cand_priors=inv.cand_priors, npunct=len(inv.punct), n_lex_vocab=inv.n_lex_vocab,
        d=128, n_heads=4, stage1_layers=2, stage2_layers=2,
        ffn_dim=256, lex_dim=64, r_big=32, r_tri=16, top_m=8, drop=0.10,
    ).to(device)

    # --- optional PMI (train-only) ---
    pmi_table, lam = None, None
    if VARIANT == "pmi":
        from vidicrf_v2.pmi import PMITable

        t0 = time.perf_counter()
        pmi_table = PMITable(
            [t for _, t in train_pairs],  # TRAIN PARTITION ONLY
            min_pair_count=3, max_pairs=400_000,
        )
        pmi_table.attach_vocab(inv.vocab)
        pmi_table.to(device)
        lam = torch.tensor(0.5, device=device, requires_grad=True)
        # inv.cand_ids is a plain CPU tensor (not a registered buffer, so
        # model.to(device) does not move it); PMI lookup indexes it with CUDA
        # top_idx, so keep a device-resident copy for the whole run.
        cand_ids_dev = inv.cand_ids.to(device)
        print(f"[pmi] table built in {time.perf_counter()-t0:.0f}s: {pmi_table.stats}", flush=True)
    else:
        cand_ids_dev = inv.cand_ids

    # --- optimizer (same grouping as A/B) ---
    lengths = train_enc.seg_lens
    steps_per_epoch = math.ceil(len(lengths) / BS)
    total_steps = max(1, steps_per_epoch * epochs)
    no_decay_ids = set()
    for mod in model.modules():
        if isinstance(mod, (torch.nn.Embedding, torch.nn.LayerNorm)):
            for p in mod.parameters(recurse=False):
                no_decay_ids.add(id(p))
    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if id(p) in no_decay_ids or name.endswith(".bias") or p.ndim < 2:
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    groups = [
        {"params": decay_params, "weight_decay": 0.01},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    if lam is not None:
        groups.append({"params": [lam], "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, lr=0.0020)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, total_steps))

    use_amp = device == "cuda" and not SMOKE
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    def pmi_addend(out, base_ids):
        from vidicrf_v2.pmi import pmi_bigram_addend

        return pmi_bigram_addend(
            pmi_table, out["bigram"], base_ids, out["top_idx"], cand_ids_dev, lam
        )

    @torch.no_grad()
    def eval_val():
        model.eval()
        order = np.argsort(val_enc.seg_lens, kind="stable")
        ok = tot = 0
        for pos in range(0, len(order), BS):
            idx = order[pos:pos + BS]
            b = make_batch_v2(val_enc, inv, idx, device, partial_diac_prob=0.0)
            out = model(
                b["base_ids"], b["sent_len"], b["case_ids"], b["left_gaps"],
                b["right_gaps"], b["surface_chars"], b["surface_phono"],
                b["has_input_diac"], b["surface_lex"],
            )
            if pmi_table is not None:
                out["bigram"] = out["bigram"] + pmi_addend(out, b["base_ids"])
            pred_p = trigram_viterbi(out["emit_p"], out["bigram"], out["trigram"], out["mask_p"])
            pred_orig = out["top_idx"].gather(-1, pred_p.unsqueeze(-1)).squeeze(-1)
            ok += int(((pred_orig == b["gold"]) & out["valid_tok"]).sum())
            tot += int(b["sent_len"].sum())
        model.train()
        return ok / max(1, tot)

    best_acc, best_state = -1.0, None
    t_train0 = time.perf_counter()
    n_nan = 0
    for ep in range(1, epochs + 1):
        model.train()
        model.invalidate_cache()
        ep_loss = n_steps = 0.0
        t_ep = time.perf_counter()
        for idx in epoch_batches(lengths, BS, rng):
            b = make_batch_v2(train_enc, inv, idx, device, partial_diac_prob=0.15, rng=rng)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                out = model(
                    b["base_ids"], b["sent_len"], b["case_ids"], b["left_gaps"],
                    b["right_gaps"], b["surface_chars"], b["surface_phono"],
                    b["has_input_diac"], b["surface_lex"], gold=b["gold"],
                )
            # CRF + CE losses always in FP32 (NEG=-1e9 overflows FP16 logsumexp)
            out = {
                k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
                for k, v in out.items()
            }
            if pmi_table is not None:
                with torch.autocast("cuda", enabled=False):
                    out["bigram"] = out["bigram"] + pmi_addend(out, b["base_ids"])
            with torch.autocast("cuda", enabled=False):
                loss, _ = compute_loss(
                    out, b["gold"], b["sent_len"], margin=0.10,
                    lambda_emit2=0.35, lambda_emit1=0.25, core_mask=b["core_mask"],
                )
            if not torch.isfinite(loss):
                n_nan += 1
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            plist = [p for g in groups for p in g["params"]]
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(plist, 5.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(plist, 5.0)
                opt.step()
            sched.step()
            ep_loss += float(loss.detach())
            n_steps += 1
        val_acc = eval_val()
        print(f"[ep {ep}/{epochs}] loss={ep_loss/max(1,n_steps):.4f} val_acc={val_acc*100:.2f}% "
              f"({time.perf_counter()-t_ep:.0f}s, nan_skips={n_nan})"
              + (f" lam={float(lam.detach().item()):.4f}" if lam is not None else ""), flush=True)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    train_secs = time.perf_counter() - t_train0
    print(f"[train] total {train_secs:.0f}s, best_val={best_acc*100:.2f}%", flush=True)

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    torch.save({"model": best_state}, WORK / "v2_best.pt")

    # ---------------------------------------------------------------------------
    # Unified inline evaluation path (works for base & PMI variants; the PMI
    # variant re-adds the (frozen-scalar) PMI addend to bigram before decoding).
    # ---------------------------------------------------------------------------
    model.build_inference_cache()

    def forward_source(source, use_pmi=True):
        matches = list(TOK.finditer(source))
        n = len(matches)
        raw = [m.group() for m in matches]
        sl = [_strip(t).lower() for t in raw]
        hd = [1 if has_diacritic(t) else 0 for t in raw]
        surf = [
            normalize_tone_placement(unicodedata.normalize("NFC", t).lower(), "classic") if h else b
            for t, b, h in zip(raw, sl, hd)
        ]
        lex = [restorer._lex_lookup.get(s, 0) if h else 0 for s, h in zip(surf, hd)]
        _, gp = extract_gaps(source)
        gseq = [p for p, _ in gp] + [gp[-1][1]]
        sch = encode_strings_chars(surf, inv.char_tables["c2i"], inv.chw).to(device)
        sph = encode_strings_phono(surf).to(device)
        b_t = torch.tensor([[inv.bid.get(b, 0) for b in sl]], device=device)
        ln = torch.tensor([n], device=device)
        c_t = torch.tensor([[case_id(t) for t in raw]], device=device)
        lg = torch.tensor([[inv.pid.get(g, 1) for g in gseq[:n]]], device=device)
        rg = torch.tensor([[inv.pid.get(g, 1) for g in gseq[1:n + 1]]], device=device)
        hd_t = torch.tensor([hd], device=device)
        lx = torch.tensor([lex], device=device)
        out = model(b_t, ln, c_t, lg, rg, sch.unsqueeze(0), sph.unsqueeze(0),
                    has_input_diac=hd_t, surface_lex=lx)
        if pmi_table is not None and use_pmi:
            out["bigram"] = out["bigram"] + pmi_addend(out, b_t)
        return raw, b_t, out

    def reassemble(source, matches, raw, b_t, out, pred):
        top_idx = out["top_idx"][0]
        words = []
        for i, t in enumerate(raw):
            bi = int(b_t[0, i])
            if bi == 0:
                words.append(t)
                continue
            slot = int(top_idx[i, int(pred[i])])
            vid = int(inv.cand_ids[bi, slot])
            w = inv.vocab[vid] if vid >= 0 else t
            words.append(restore_case(w, t) if len(w) == len(t) else w)
        return replace_spans(source, matches, words)

    def predict_any(source, mode="viterbi", tau=None, use_pmi=True):
        matches = list(TOK.finditer(source))
        n = len(matches)
        if n == 0:
            return source
        raw, b_t, out = forward_source(source, use_pmi=use_pmi)
        if tau is not None:
            # Two-pass anchored decoding on the SAME lattice:
            # pass 1 = FB-MBR marginals; anchor tokens with marginal prob >= tau.
            logm, _ = trigram_marginals(
                out["emit_p"], out["bigram"], out["trigram"], out["mask_p"]
            )
            probs = logm.exp()
            conf, slot1 = probs.max(dim=-1)  # (1, L)
            locked_mask = torch.zeros_like(out["mask_p"])
            anchor = (conf >= tau) & out["mask_p"].any(dim=-1)
            pos = torch.nonzero(anchor[0])
            for t in pos.tolist():
                s = int(slot1[0, t])
                if out["mask_p"][0, t, s]:
                    locked_mask[0, t, s] = True
            pred = get_scripted_viterbi()(
                out["emit_p"], out["bigram"], out["trigram"], out["mask_p"],
                locked_mask_p=locked_mask,
            )[0]
        elif mode == "mbr":
            logm, _ = trigram_marginals(
                out["emit_p"], out["bigram"], out["trigram"], out["mask_p"]
            )
            pred = logm.argmax(-1)[0]
        else:
            pred = get_scripted_viterbi()(
                out["emit_p"], out["bigram"], out["trigram"], out["mask_p"]
            )[0]
        return reassemble(source, matches, raw, b_t, out, pred)

    restorer = FastRestorerV2(model, inv, device=device, use_cache=True)

    # Sanity: inline path must reproduce restorer outputs exactly (base decoding,
    # no PMI — restorer has no PMI hook, so compare the plain inline decode).
    n_check = min(20, len(test_pairs))
    for s, _ in test_pairs[:n_check]:
        a = restorer.predict(s, decode_mode="viterbi")[0]
        b = predict_any(s, "viterbi", use_pmi=False)
        assert a == b, f"inline/restorer mismatch:\n{a}\n{b}"
    print(f"[eval] inline/restorer parity OK on {n_check} paragraphs", flush=True)

    print("[eval] scoring test...", flush=True)
    t_eval0 = time.perf_counter()
    res_by_mode = {}
    preds_cache = {}
    for mode in ("viterbi", "mbr"):
        preds = [predict_any(s, mode) for s, _ in test_pairs]
        preds_cache[mode] = preds
        res_by_mode[mode] = score_pairs_detailed(test_pairs, preds, ambiguous_bases)
        print(f"[res] {mode}:", json.dumps(res_by_mode[mode], ensure_ascii=False), flush=True)

    two_pass = {}
    for tau in (0.80, 0.90, 0.95):
        preds = [predict_any(s, "viterbi", tau=tau) for s, _ in test_pairs]
        two_pass[str(tau)] = score_pairs_detailed(test_pairs, preds, ambiguous_bases)
        print(f"[res] two_pass tau={tau}:", json.dumps(two_pass[str(tau)], ensure_ascii=False), flush=True)

    preds_partial = [predict_any(s, "viterbi") for s, _ in partial_pairs]
    res_partial = score_partial(partial_pairs, preds_partial, kept_masks)
    print(f"[res] partial:", json.dumps(res_partial, ensure_ascii=False), flush=True)
    print(f"[eval] total {time.perf_counter()-t_eval0:.0f}s", flush=True)

    results = {
        "variant": VARIANT,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "torch": torch.__version__,
        "smoke": SMOKE,
        "data": {
            "source": "wikimedia/wikipedia 20231101.vi",
            "revision": splits["revision"],
            "train_rows": len(train_pairs), "validation_rows": len(val_pairs),
            "test_rows": len(test_pairs),
            "test_tokens": sum(len(TOK.findall(t)) for _, t in test_pairs),
            "ambiguous_bases_in_train": len(ambiguous_bases),
        },
        "inventory_stats": inv.stats,
        "pmi_stats": pmi_table.stats if pmi_table is not None else None,
        "lambda_pmi_final": (float(lam.detach().item()) if lam is not None else None),
        "train_seconds": round(train_secs, 1),
        "epochs": epochs,
        "best_val_acc_pct": round(best_acc * 100.0, 2),
        "nan_skips": n_nan,
        "amp": {"enabled": use_amp, "dtype": str(amp_dtype) if use_amp else None},
        "test_viterbi": res_by_mode["viterbi"],
        "test_mbr": res_by_mode["mbr"],
        "test_two_pass": two_pass,
        "partial_diacritic_40pct": res_partial,
    }
    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[done] results ->", OUT_JSON, flush=True)
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
