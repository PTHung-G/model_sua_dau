"""Evaluation with explicit token ids.

Every test token carries (sentence_id, position, form_id, gold_slot, n_cand).
Predictions are produced per sentence and matched to labels by (sentence_id,
position), never by batch index, so batching and shuffling cannot misalign them.
"""
import json
from collections import Counter

import numpy as np
import torch

from vn import strip_all

NEG = -1e9


class Tokens:
    """Encoded sentences plus their label tensors."""

    def __init__(self, sents, cand, char2id, max_len=64):
        self.cand = cand
        self.forms = sorted(cand)
        self.fid = {s: i for i, s in enumerate(self.forms)}
        self.max_c = max(len(cand[s]) for s in self.forms)
        self.max_len = max_len
        nchars = len(char2id) + 1
        self.pad_c = nchars          # padding char id, must be < embedding size
        self.sents = []
        self.n_oov = 0
        for ws in sents:
            row, kept = [], []
            for w in ws:
                wl = w.lower()
                s = strip_all(wl)
                if s not in cand:
                    self.n_oov += 1
                    continue
                cs = cand[s]
                row.append((self.fid[s], cs.index(wl) if wl in cs else -1,
                            len(cs), [char2id.get(c, 0) for c in s[:max_len]]))
            if len(row) >= 2:
                self.sents.append(row)
        self.n_tok = sum(len(r) for r in self.sents)
        self.n_uncovered = sum(1 for r in self.sents for (_, g, _, _) in r if g < 0)

    def batches(self, bs=64, device="cpu", shuffle=False, seed=0):
        order = list(range(len(self.sents)))
        if shuffle:
            np.random.RandomState(seed).shuffle(order)
        for i in range(0, len(order), bs):
            ids = order[i:i + bs]
            B = len(ids)
            T = max(len(self.sents[j]) for j in ids)
            L = max(max((len(c) for (_, _, _, c) in self.sents[j]), default=1) for j in ids)
            xc = torch.full((B, T, L), self.pad_c, dtype=torch.long)
            cl = torch.ones((B, T), dtype=torch.long)
            f = torch.zeros((B, T), dtype=torch.long)
            y = torch.zeros((B, T), dtype=torch.long)
            ln = torch.ones((B, T), dtype=torch.long)
            ok = torch.zeros((B, T), dtype=torch.bool)   # covered targets only
            mk = torch.zeros((B, T), dtype=torch.bool)
            for b, j in enumerate(ids):
                for t, (fi, g, k, cids) in enumerate(self.sents[j]):
                    xc[b, t, :len(cids)] = torch.tensor(cids, device=xc.device)
                    cl[b, t] = max(len(cids), 1)
                    f[b, t] = fi
                    y[b, t] = max(g, 0)
                    ln[b, t] = k
                    mk[b, t] = True
                    ok[b, t] = g >= 0     # uncovered tokens cannot be right
            yield (xc.to(device), cl.to(device), f.to(device), y.to(device),
                   ln.to(device), mk.to(device), ok.to(device))


def masks(ln, mk, y):
    """ambig: >1 candidate. base_ok: majority-form prediction is correct."""
    amb = mk & (ln > 1)
    base_ok = mk & (y == 0)
    return amb, base_ok


def evaluate(predict, toks, device="cpu", bs=64, coverage_ok=None):
    """predict(xc,cl,f,ln) -> tensor of slot indices.

    Returns accuracy overall and on each subgroup, plus the two quantities that
    decide whether re-balancing actually helps:
        fixed    baseline wrong, model right
        broken   baseline right, model wrong
    """
    st = {k: [0, 0] for k in ("all", "amb", "base_ok", "base_bad")}
    fixed = broken = 0
    n = 0
    n_uncovered = 0
    for xc, cl, f, y, ln, mk, ok in toks.batches(bs, device):
        pred = predict(xc, cl, f, ln)
        n_uncovered += int((mk & ~ok).sum())
        hit = (pred == y) & ok          # an uncovered target is always wrong
        amb, b_ok = masks(ln, ok, y)
        b_bad = ok & ~b_ok
        n += int(mk.sum())
        fixed += int((hit & b_bad).sum())
        broken += int((~hit & b_ok).sum())
        for name, sel in (("all", mk), ("amb", amb), ("base_ok", b_ok), ("base_bad", b_bad)):
            c = int(sel.sum())
            st[name][0] += c
            st[name][1] += int((hit & sel).sum())
    out = {k: (v[1] / v[0] if v[0] else 0.0) for k, v in st.items()}
    out["n"] = n
    out["n_amb"] = st["amb"][0]
    out["n_base_bad"] = st["base_bad"][0]
    out["fixed"] = fixed
    out["broken"] = broken
    out["net"] = fixed - broken
    out["fix_rate"] = fixed / max(st["base_bad"][0], 1)
    out["break_rate"] = broken / max(st["base_ok"][0], 1)
    out["n_uncovered"] = n_uncovered
    out["coverage"] = 1.0 - n_uncovered / max(n, 1)
    return out


def fmt(tag, r):
    return (f"{tag:22s} ALL={r['all']:.4f} AMB={r['amb']:.4f} "
            f"fix={r['fix_rate']:.4f} break={r['break_rate']:.4f} "
            f"net={r['net']:+d} cov={r['coverage']:.4f}")
