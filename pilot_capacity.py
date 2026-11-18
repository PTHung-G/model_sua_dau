"""Does a bigger model close the gap between plain CE and rare-weighted CE?

Two explanations for why weighted loss hurt (fix/break fell from 11.6:1 to 6.5:1):
  H1 capacity  - the small model cannot serve both frequent and rare candidates,
                 so a bigger model should recover the ratio.
  H2 noise     - rare = mostly misspellings / proper nouns / label errors, which
                 weighting up can only make worse; model size will not help.

Design: {small, large} x {control, weighted} x 3 seeds, everything else fixed.
The large model keeps the SAME architecture family and window, only width grows.

  small  syl=32  dim=64   1 block
  large  syl=64  dim=128  2 blocks   (~8x parameters)

If weighted loss under the large model still loses to its own control, H1 is dead
and the story is data noise, not capacity.
"""
import json
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import evaluate as ev
import pilot_balance as pb
from attn_model import AttnDiac

SIZES = {
    "small": dict(syl=32, dim=64, blocks=1),
    "large": dict(syl=64, dim=128, blocks=2),
}
CONFIGS = {
    "control": dict(ambiguous_only=True),
    "weighted": dict(ambiguous_only=True, exponent=0.5, cap=3.0),
}


def train_one(size, cfgname, seed, bundle, steps, bs=64):
    tr, toks_va, cand, freq, char2id, max_c, nchar, pool, forms = bundle
    cfg = CONFIGS[cfgname]
    torch.manual_seed(seed)
    np.random.seed(seed)
    kw = SIZES[size]
    model = AttnDiac(nchar, max_c, window=8, out_std=1e-1, **kw).to(pb.DEV)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    wlook = None
    if cfg.get("exponent"):
        wmap = pb.token_weights(cand, freq, cfg["exponent"], cfg["cap"])
        wlook = {(forms[s], ci): wmap[(s, ci)]
                 for s, cs in cand.items() for ci in range(len(cs))}

    rng = np.random.default_rng(seed)
    gen = pb.make_batches(tr, cand, char2id, max_c, bs, rng, pool=pool,
                          pool_prob=cfg.get("pool_prob", 0.0), nchar=nchar,
                          forms=forms)
    for step in range(steps):
        xc, cl, f, y, ln, mk, ok = next(gen)
        lg = pb.logits_of(model, xc, cl, ln)
        amb = mk & (ln > 1) & ok
        tgt = amb if amb.any() else ok
        ce = F.cross_entropy(lg.reshape(-1, lg.size(-1)), y.reshape(-1),
                             reduction="none").reshape(y.shape)
        if wlook is not None:
            wv = torch.ones_like(ce)
            keys = list(zip(f[amb].tolist(), y[amb].tolist()))
            wv[amb] = torch.tensor([wlook[k] for k in keys], device=ce.device)
            loss = (ce * wv)[tgt].sum() / wv[tgt].sum()
        else:
            loss = ce[tgt].mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if (step + 1) % max(steps // 3, 1) == 0:
            print(f"      step {step+1}/{steps} loss={loss.item():.4f}", flush=True)

    model.eval()
    with torch.inference_mode():
        pred = lambda xc, cl, f, ln: pb.logits_of(model, xc, cl, ln).argmax(-1)
        r = ev.evaluate(pred, toks_va, device=pb.DEV)
    r["params"] = nparam
    return r


def main():
    ntr = int(sys.argv[1]) if len(sys.argv) > 1 else 60000
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
    seeds = [int(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["0", "1", "2"])]
    print(f"device={pb.DEV} ntr={ntr} steps={steps} seeds={seeds}", flush=True)

    tr_all = pb.load_sents("data/tr.txt")
    tr = tr_all[:ntr]
    va = pb.load_sents("data/va.txt")[:4000]
    cand = json.load(open("data/cand.json"))
    freq = {}
    for ws in tr_all:
        for w in ws:
            freq[w.lower()] = freq.get(w.lower(), 0) + 1
    forms = sorted(cand)
    forms_id = {s: i for i, s in enumerate(forms)}
    chars = sorted({c for s in forms for c in s})
    char2id = {c: i for i, c in enumerate(chars, start=1)}
    max_c = max(len(v) for v in cand.values())
    toks_va = ev.Tokens(va, cand, char2id)
    pool = pb.build_pool(tr, cand, freq)
    bundle = (tr, toks_va, cand, freq, char2id, max_c, len(chars) + 2, pool, forms_id)
    print(f"train={len(tr):,} val_tok={toks_va.n_tok:,}", flush=True)

    results = []
    for size in SIZES:
        for cfgname in CONFIGS:
            for sd in seeds:
                print(f"  --- {size} / {cfgname} seed={sd}", flush=True)
                r = train_one(size, cfgname, sd, bundle, steps)
                r.update(size=size, config=cfgname, seed=sd)
                print(f"    {ev.fmt(cfgname, r)}  params={r['params']:,}", flush=True)
                results.append(r)
                json.dump(results, open("pilot_capacity.json", "w"), indent=1)

    print("\n=== summary (mean over seeds) ===")
    for size in SIZES:
        for cfgname in CONFIGS:
            rs = [r for r in results if r["size"] == size and r["config"] == cfgname]
            if not rs:
                continue
            m = lambda k: sum(r[k] for r in rs) / len(rs)
            ratio = (m("fix_rate") / m("break_rate")) if m("break_rate") else float("inf")
            print(f"{size:6s} {cfgname:9s} params={rs[0]['params']:>8,} "
                  f"ALL={m('all'):.4f} AMB={m('amb'):.4f} "
                  f"fix={m('fix_rate'):.4f} break={m('break_rate'):.4f} "
                  f"net={m('net'):+.1f} ratio={ratio:.1f}:1")


if __name__ == "__main__":
    main()
