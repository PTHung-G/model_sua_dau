"""Pilot: does re-balancing toward rare candidates help?

Architecture is fixed at the best small config found earlier: 1 attention block,
window 8, out-init 1e-1, no prior added to the logits, AdamW wd=0. Only the
training distribution and the loss weighting change:

  A  control            natural sampling, plain CE averaged over ambiguous tokens
  B  weighted loss      w = min(3, ((n_max+1)/(n_y+1))**0.5), ambiguous tokens only
  C  sampling          75% natural / 25% minority-candidate pool
  D  combined light    85/15 sampling + w = min(2, (...)**0.25)

All configs use the same number of optimizer steps and the same initialisation
for a given seed. Checkpoint selection is by validation ALL accuracy.
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import evaluate as ev
from attn_model import AttnDiac
from vn import strip_all

TORCH_THREADS = 2
if not torch.cuda.is_available():
    torch.set_num_threads(TORCH_THREADS)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NEG = -1e9


def load_sents(p):
    return [l.split() for l in open(p, encoding="utf-8")]


def logits_of(model, xc, cl, ln):
    lg = model(xc, cl)
    ar = torch.arange(lg.size(-1), device=lg.device)[None, None, :]
    return lg.masked_fill(ar >= ln[..., None], NEG)


def token_weights(cand, freq, exponent, cap):
    """Per (form, slot) loss weight, from TRAIN frequencies only."""
    w = {}
    for s, cs in cand.items():
        nmax = freq.get(cs[0], 0)
        for ci, c in enumerate(cs):
            r = (nmax + 1.0) / (freq.get(c, 0) + 1.0)
            w[(s, ci)] = min(cap, r ** exponent)
    return w


def build_pool(sents, cand, freq, min_freq=5):
    """Sentences containing at least one rare-but-valid minority candidate."""
    pool = []
    for ws in sents:
        hit = False
        for w in ws:
            wl = w.lower()
            s = strip_all(wl)
            cs = cand.get(s)
            if not cs:
                continue
            ci = cs.index(wl) if wl in cs else 0
            if ci != 0 and freq.get(wl, 0) >= min_freq:
                hit = True
                break
        if hit:
            pool.append(ws)
    return pool


def make_batches(sents, cand, char2id, max_c, bs, rng, pool=None, pool_prob=0.0,
                 nchar=None, forms=None):
    """Yield training tensors. Optionally draw a fraction of batches from `pool`."""
    n = len(sents)
    while True:
        src = sents
        if pool and rng.random() < pool_prob:
            src = pool
        idx = rng.integers(0, len(src), size=bs)
        rows = []
        for i in idx:
            row = []
            for w in src[i]:
                wl = w.lower()
                s = strip_all(wl)
                cs = cand.get(s)
                if not cs:
                    continue
                ci = cs.index(wl) if wl in cs else -1
                row.append((s, len(cs), ci))
            if len(row) >= 2:
                rows.append(row)
        if not rows:
            continue
        B = len(rows)
        T = max(len(r) for r in rows)
        L = max(max(len(s) for (s, _, _) in r) for r in rows)
        xc = torch.full((B, T, L), nchar - 1, dtype=torch.long)
        cl = torch.ones((B, T), dtype=torch.long)
        f = torch.zeros((B, T), dtype=torch.long)
        y = torch.zeros((B, T), dtype=torch.long)
        ln = torch.ones((B, T), dtype=torch.long)
        mk = torch.zeros((B, T), dtype=torch.bool)
        ok = torch.zeros((B, T), dtype=torch.bool)
        for b, r in enumerate(rows):
            for t, (s, k, ci) in enumerate(r):
                ids = [char2id.get(c, 0) for c in s[:64]]
                xc[b, t, :len(ids)] = torch.tensor(ids, device=xc.device)
                cl[b, t] = max(len(ids), 1)
                f[b, t] = forms[s]
                y[b, t] = max(ci, 0)
                ln[b, t] = k
                mk[b, t] = True
                ok[b, t] = ci >= 0
        yield (xc.to(DEV), cl.to(DEV), f.to(DEV), y.to(DEV), ln.to(DEV),
               mk.to(DEV), ok.to(DEV))


def train_one(cfg, seed, bundle, steps, bs=64):
    tr, toks_va, cand, freq, char2id, max_c, nchar, pool = bundle
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = AttnDiac(nchar, max_c, syl=32, dim=64, window=8, blocks=1,
                     out_std=1e-1).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    wmap = None
    if cfg.get("exponent"):
        wmap = token_weights(cand, freq, cfg["exponent"], cfg["cap"])
        # weights indexed by (form, slot)
        wlook = {}
        for s, cs in cand.items():
            for ci in range(len(cs)):
                wlook[(FORMS[s], ci)] = wmap[(s, ci)]

    rng = np.random.default_rng(seed)
    gen = make_batches(tr, cand, char2id, max_c, bs, rng,
                       pool=pool, pool_prob=cfg.get("pool_prob", 0.0), nchar=nchar,
                       forms=FORMS)
    t0 = time.time()
    for step in range(steps):
        xc, cl, f, y, ln, mk, ok = next(gen)
        lg = logits_of(model, xc, cl, ln)
        amb = mk & (ln > 1) & ok
        if cfg.get("ambiguous_only") and amb.any():
            tgt = amb
        else:
            tgt = ok
        ce = F.cross_entropy(lg.reshape(-1, lg.size(-1)), y.reshape(-1),
                             reduction="none").reshape(y.shape)
        loss_tok = ce
        if wmap is not None:
            wv = torch.ones_like(ce)
            fl = f[amb].tolist(); yl = y[amb].tolist()
            keys = list(zip(fl, yl))
            wv[amb] = torch.tensor([wlook[k] for k in keys], device=ce.device)
            loss = (loss_tok * wv)[tgt].sum() / wv[tgt].sum()
        else:
            loss = loss_tok[tgt].mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if (step + 1) % max(steps // 5, 1) == 0:
            print(f"      step {step+1}/{steps} loss={loss.item():.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    model.eval()
    with torch.inference_mode():
        pred = lambda xc, cl, f, ln: logits_of(model, xc, cl, ln).argmax(-1)
        r = ev.evaluate(pred, toks_va, device=DEV)
    return model, r


CONFIGS = {
    "A control": dict(ambiguous_only=True),
    "B weighted": dict(ambiguous_only=True, exponent=0.5, cap=3.0),
    "C sampling": dict(pool_prob=0.25),
    "D combined": dict(pool_prob=0.15, exponent=0.25, cap=2.0),
}


def main():
    ntr = int(sys.argv[1]) if len(sys.argv) > 1 else 60000
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    seeds = [int(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["0", "1"])]
    print(f"device={DEV} ntr={ntr} steps={steps} seeds={seeds}", flush=True)

    tr = load_sents("data/tr.txt")[:ntr]
    va = load_sents("data/va.txt")[:4000]
    cand = json.load(open("data/cand.json"))
    freq = {}
    for ws in load_sents("data/tr.txt"):
        for w in ws:
            freq[w.lower()] = freq.get(w.lower(), 0) + 1
    forms = sorted(cand)
    global FORMS
    FORMS = {s: i for i, s in enumerate(forms)}
    chars = sorted({c for s in forms for c in s})
    char2id = {c: i for i, c in enumerate(chars, start=1)}
    max_c = max(len(v) for v in cand.values())
    toks_va = ev.Tokens(va, cand, char2id)
    pool = build_pool(tr, cand, freq)
    print(f"train={len(tr):,} val_tok={toks_va.n_tok:,} pool={len(pool):,} "
          f"({len(pool)/len(tr):.3f} of train)", flush=True)

    bundle = (tr, toks_va, cand, freq, char2id, max_c, len(chars) + 2, pool)
    results = []
    for name, cfg in CONFIGS.items():
        for sd in seeds:
            print(f"  --- {name} seed={sd}", flush=True)
            _, r = train_one(cfg, sd, bundle, steps)
            r = dict(r); r["config"] = name; r["seed"] = sd
            print(f"    {ev.fmt(name, r)}", flush=True)
            results.append(r)
            json.dump(results, open("pilot_balance.json", "w"), indent=1)

    print("\n=== summary (mean over seeds) ===")
    for name in CONFIGS:
        rs = [r for r in results if r["config"] == name]
        if not rs:
            continue
        m = lambda k: sum(r[k] for r in rs) / len(rs)
        print(f"{name:12s} ALL={m('all'):.4f} AMB={m('amb'):.4f} "
              f"fix={m('fix_rate'):.4f} break={m('break_rate'):.4f} "
              f"net={m('net'):+.1f}")


if __name__ == "__main__":
    main()
