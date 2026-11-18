"""Legacy-checkpoint diagnostic scorer, not an official seq2seq leaderboard.

Inference uses public GRID inputs (never targets), or stripped local text.
Tokens containing letters are scored case-insensitively on punctuation-free NFC
cores. Report stem coverage and exact-gold candidate coverage separately.
Historical checkpoints retain unpacked character encoding, singleton batches;
changing batching changes legacy predictions. New pilot models are separate.
"""
import argparse
import collections
import json
import unicodedata

import torch

import pilot_balance as pb
from attn_model import AttnDiac
from vn import strip_all, tone_of, decompose

PUNCT_STRIP = str.maketrans("", "", ".,;:!?\"'()[]{}-–—…«»/\\|*_`~@#$%^&+=<>")

# (tag, path, note). Captions are what appear in the result tables.
TESTSETS = [
    ("wiki",     "data/te.txt",      "Wikipedia, article-level holdout, 138,497 sentences"),
    ("news",     "data/te_news.txt", "News holdout, 59,997 sentences, cross-register"),
    ("business", "GRID:business_55", "public grid, contracts/news"),
    ("formal",   "GRID:formal_72",   "public grid, legal prose (UDHR)"),
    ("conv",     "GRID:conversational_300", "public grid, Tatoeba conversation"),
    ("literary", "GRID:literary_800", "public grid, classical literature"),
]

CAND = {}
FORMS_BY_STRIP = {}


def core(w):
    return "".join(c for c in unicodedata.normalize("NFC", w)
                   if not unicodedata.category(c).startswith("P")
                   and c.translate(PUNCT_STRIP))


def normalized_core(w):
    return core(w).lower()


def is_scoring_token(w):
    return any(c.isalpha() for c in core(w))


def load_pairs(path):
    """Return (inference input, gold); targets never serve as model input."""
    if path.startswith("GRID:"):
        from datasets import load_dataset
        ds = load_dataset("nrl-ai/vn-diacritic-eval", path.split(":", 1)[1], split="train")
        return [(ex["input"], ex["target"]) for ex in ds]
    with open(path, encoding="utf-8") as f:
        gold = [unicodedata.normalize("NFC", l.rstrip("\n")) for l in f]
    return [(strip_all(g), g) for g in gold]


def load_lines(path):
    """Compatibility helper returning input text only, never GRID targets."""
    return [inp for inp, _ in load_pairs(path)]


def restore_token(token, chosen):
    """Preserve input punctuation and simple case around a candidate core."""
    original = core(token)
    if original.isupper():
        chosen = chosen.upper()
    elif original[:1].isupper():
        chosen = chosen.capitalize()
    letters = iter(chosen)
    return "".join(c if not core(c) else next(letters, "")
                   for c in token) + "".join(letters)


@torch.inference_mode()
def predict(model, sents, char2id, nchar, max_c):
    """Return list-of-token predictions, aligned to the gold token positions.

    Input accents are stripped defensively, including on uncovered tokens.
    Legacy inference uses one sentence per batch and unpacked character states.
    """
    out = []
    for s in sents:
        toks = unicodedata.normalize("NFC", strip_all(s)).split()
        rows = []
        for w in toks:
            st = strip_all(normalized_core(w))
            cs = CAND.get(st)
            if cs:
                rows.append((st, len(cs)))
        if not rows:
            out.append(toks)
            continue
        T = len(rows)
        L = max(len(r[0]) for r in rows)
        xc = torch.full((1, T, L), nchar - 1, dtype=torch.long)
        cl = torch.ones((1, T), dtype=torch.long)
        f = torch.zeros((1, T), dtype=torch.long)
        ln = torch.ones((1, T), dtype=torch.long)
        for t, (st, k) in enumerate(rows):
            ids = [char2id.get(c, 0) for c in st[:64]]
            xc[0, t, :len(ids)] = torch.tensor(ids)
            cl[0, t] = max(len(ids), 1)
            f[0, t] = FORMS_BY_STRIP[st]
            ln[0, t] = k
        lg = pb.logits_of(model, xc.to(pb.DEV), cl.to(pb.DEV), ln.to(pb.DEV))
        pick = lg.argmax(-1)[0].tolist()
        words, j = [], 0
        for w in toks:
            st = strip_all(normalized_core(w))
            cs = CAND.get(st)
            if not cs:
                words.append(w)
                continue
            c = cs[pick[j]] if pick[j] < len(cs) else cs[0]
            j += 1
            words.append(restore_token(w, c))
        out.append(words)
    return out


def score(gold_sents, pred_tokens):
    """Compute every reported metric in one pass over one alignment."""
    n_tok = n_cov = ok_cov = ok_raw = n_scoring = 0
    n_exact = ok_exact = ok_all = 0
    if len(gold_sents) != len(pred_tokens):
        raise ValueError("Prediction sentence count differs from gold")
    cap_ok = cap_n = 0
    by_ncand = collections.defaultdict(lambda: [0, 0])
    tone_err = collections.Counter()
    mod_err = collections.Counter()
    pure_tone = pure_mod = mixed = 0

    for g, p in zip(gold_sents, pred_tokens):
        gt = unicodedata.normalize("NFC", g).split()
        for i, w_t in enumerate(gt):
            c = core(w_t)
            if not is_scoring_token(w_t):
                continue                      # pure punctuation: not a syllable
            n_scoring += 1
            w_p = p[i] if i < len(p) else None
            gold_low = c.lower()
            pred_core = normalized_core(w_p) if w_p is not None else None
            good = pred_core == gold_low
            ok_all += good

            # Exact positional token match; not an official seq2seq comparison.
            if w_p is not None and w_t == w_p:
                ok_raw += 1

            st = strip_all(gold_low)
            if st not in CAND:
                continue                      # uncovered: unanswerable
            n_cov += 1
            ok_cov += good
            exact = gold_low in CAND[st]
            n_exact += exact
            ok_exact += good and exact
            if good:
                cap_n += 1
                cap_ok += int(w_t[:1].isupper() == (w_p[:1].isupper() if w_p else False))

            k = min(len(CAND[st]), 9)
            by_ncand[k][0] += good
            by_ncand[k][1] += 1

            if not good and pred_core is not None:
                gt_t, pt_t = tone_of(gold_low) or "ngang", tone_of(pred_core) or "ngang"
                _, _, gm = decompose(gold_low)
                _, _, pm = decompose(pred_core)
                gm, pm = "".join(gm) or "-", "".join(pm) or "-"
                if gt_t != pt_t and gm == pm:
                    pure_tone += 1
                    tone_err[(gt_t, pt_t)] += 1
                elif gm != pm and gt_t == pt_t:
                    pure_mod += 1
                    mod_err[(gm, pm)] += 1
                else:
                    mixed += 1

    assert n_cov <= n_scoring
    assert ok_cov <= n_cov
    r = dict(
        n_scoring=n_scoring, n_covered=n_cov,
        cov=n_cov / max(n_scoring, 1),  # compatibility alias: stem coverage
        stem_cov=n_cov / max(n_scoring, 1),
        exact_cov=n_exact / max(n_scoring, 1),
        n_exact_covered=n_exact,
        exact_acc=ok_exact / max(n_exact, 1),
        all_syll_acc=ok_all / max(n_scoring, 1),
        syll_acc=ok_cov / max(n_cov, 1),
        raw_acc=ok_raw / max(n_scoring, 1),
        cap_acc=cap_ok / max(cap_n, 1),
        pure_tone=pure_tone, pure_mod=pure_mod, mixed=mixed,
        by_ncand={k: (a, t) for k, (a, t) in by_ncand.items()},
        tone_err=tone_err.most_common(6), mod_err=mod_err.most_common(4),
    )
    assert r["syll_acc"] <= 1.0 and r["cov"] <= 1.0
    return r


def fmt(tag, r):
    return (f"{tag:10s} n={r['n_scoring']:>7,}  cov={r['cov']:.4f}  "
            f"exact_cov={r['exact_cov']:.4f} syll={r['syll_acc']:.4f}  raw={r['raw_acc']:.4f}  cap={r['cap_acc']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt_v2_m1m.pt")
    ap.add_argument("--limit", type=int, default=0, help="cap sentences per test set")
    ap.add_argument("--out", default="scored_v2.json")
    ap.add_argument("--detail", action="store_true")
    args = ap.parse_args()

    d = torch.load(args.ckpt, map_location="cpu")
    CAND.update(json.load(open("data/cand.json")))
    forms = sorted(CAND)
    FORMS_BY_STRIP.update({s: i for i, s in enumerate(forms)})
    m = AttnDiac(d["nchar"], d["max_c"], window=d["window"], out_std=d["out_std"],
                 syl=192, dim=384, blocks=1, packed_chars=False).to(pb.DEV)
    m.load_state_dict(d["model"])
    m.eval()

    results = {}
    for tag, path, note in TESTSETS:
        pairs = load_pairs(path)
        if args.limit:
            pairs = pairs[:args.limit]
        preds = predict(m, [inp for inp, _ in pairs], d["char2id"], d["nchar"], d["max_c"])
        r = score([gold for _, gold in pairs], preds)
        r["inference_protocol"] = "legacy unpacked_chars; sentence batch_size=1"
        results[tag] = r
        print(fmt(tag, r), " |", note, flush=True)
        if args.detail:
            print("   by #candidates:", {k: f"{a/t:.3f}(n={t:,})"
                                         for k, (a, t) in sorted(r["by_ncand"].items())})
            print(f"   errors: pure_tone={r['pure_tone']:,} pure_mod={r['pure_mod']:,} "
                  f"mixed={r['mixed']:,}")
            print(f"   tone : {r['tone_err']}")
            print(f"   mod  : {r['mod_err']}")

    grid = [results[t] for t in ("business", "formal", "conv", "literary")]
    import numpy as np
    agg = dict(
        grid_syll=float(np.mean([g["syll_acc"] for g in grid])),
        grid_raw=float(np.mean([g["raw_acc"] for g in grid])),
        grid_cov=float(np.mean([g["cov"] for g in grid])),
        grid_tok=sum(g["n_scoring"] for g in grid),
    )
    results["GRID_MEAN"] = agg
    print(f"\n{'GRID MEAN':10s} (unweighted over 4 registers, "
          f"{agg['grid_tok']:,} syllables) cov={agg['grid_cov']:.4f} "
          f"syll={agg['grid_syll']:.4f} raw={agg['grid_raw']:.4f}")
    json.dump(results, open(args.out, "w"), indent=1, ensure_ascii=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
