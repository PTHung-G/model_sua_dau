"""Does a bigger model help once there is enough data?

The 60k pilot said large (6.75x params) < small, but 60k sentences is too little
for 220k parameters, so that result may only show overfitting. This run repeats
the comparison on the full clean corpus (622,571 sentences) and separates width
from depth, which the pilot conflated:

  small    syl=32  dim=64   1 block    32,609
  deep     syl=32  dim=64   2 blocks   ~58k
  wide     syl=64  dim=128  1 block    ~140k

Same steps, same seed set, same locked test set, same scorer as the final run,
so the small numbers here should reproduce the reported 0.8774 baseline.
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

VARIANTS = {
    "small": dict(syl=32, dim=64, blocks=1),
    "deep":  dict(syl=32, dim=64, blocks=2),
    "wide":  dict(syl=64, dim=128, blocks=1),
}


def train_one(kw, seed, bundle, steps, bs=64, lr=2e-3):
    tr, cand, char2id, max_c, nchar, forms = bundle
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = AttnDiac(nchar, max_c, window=8, out_std=1e-1, **kw).to(pb.DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    rng = np.random.default_rng(seed)
    gen = pb.make_batches(tr, cand, char2id, max_c, bs, rng, nchar=nchar, forms=forms)
    for step in range(steps):
        xc, cl, f, y, ln, mk, ok = next(gen)
        lg = pb.logits_of(model, xc, cl, ln)
        amb = mk & (ln > 1) & ok
        tgt = amb if amb.any() else ok
        loss = F.cross_entropy(lg.reshape(-1, lg.size(-1)), y.reshape(-1),
                               reduction="none").reshape(y.shape)[tgt].mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if (step + 1) % (steps // 4) == 0:
            print(f"      step {step+1}/{steps} loss={loss.item():.4f}", flush=True)
    model.eval()
    return model


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    seeds = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["0", "1"])]
    print(f"device={pb.DEV} steps={steps} seeds={seeds}", flush=True)

    tr_all = pb.load_sents("data/tr.txt")
    te_sents = pb.load_sents("data/te.txt")
    cand = json.load(open("data/cand.json"))
    forms = sorted(cand)
    forms_id = {s: i for i, s in enumerate(forms)}
    chars = sorted({c for s in forms for c in s})
    char2id = {c: i for i, c in enumerate(chars, start=1)}
    max_c = max(len(v) for v in cand.values())
    nchar = len(chars) + 2
    toks_te = ev.Tokens(te_sents, cand, char2id)
    print(f"train={len(tr_all):,} test_tok={toks_te.n_tok:,}", flush=True)

    bundle = (tr_all, cand, char2id, max_c, nchar, forms_id)
    results = []
    for name, kw in VARIANTS.items():
        for sd in seeds:
            print(f"  --- {name} seed={sd}", flush=True)
            model = train_one(kw, sd, bundle, steps)
            nparam = sum(p.numel() for p in model.parameters())
            with torch.inference_mode():
                pred = lambda xc, cl, f, ln: pb.logits_of(model, xc, cl, ln).argmax(-1)
                r = ev.evaluate(pred, toks_te, device=pb.DEV)
            r.update(variant=name, seed=sd, params=nparam, **kw)
            print(f"    {ev.fmt(name, r)}  params={nparam:,}", flush=True)
            results.append(r)
            json.dump(results, open("size_results.json", "w"), indent=1)
            del model
            torch.cuda.empty_cache()

    print("\n=== summary (mean over seeds) ===")
    for name in VARIANTS:
        rs = [r for r in results if r["variant"] == name]
        m = lambda k: sum(r[k] for r in rs) / len(rs)
        ratio = m("fix_rate") / m("break_rate") if m("break_rate") else float("inf")
        print(f"{name:6s} params={rs[0]['params']:>8,} ALL={m('all'):.4f} "
              f"AMB={m('amb'):.4f} fix={m('fix_rate'):.4f} "
              f"break={m('break_rate'):.4f} net={m('net'):+.0f} ratio={ratio:.1f}:1")


if __name__ == "__main__":
    main()
