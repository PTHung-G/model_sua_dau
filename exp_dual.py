"""Ablation #2 of two: DUAL-BRANCH attention (local band + full), gate-fused.

Question: does giving the model a separate full-sentence branch, fused by a learned
gate, beat a single banded window?

Motivation, both from the literature and from our own failures:
  * DeepSeek NSA (arXiv 2502.11089): with ONE attention operation, fast local
    patterns dominate and the model never learns to use longer-range structure.
  * We widened window 8 -> 16 twice and accuracy got WORSE both times
    (hyp.json: 0.8350 at +-16 vs 0.8469 at +-8; hyp2_all.json: 0.7536 full vs
    0.7594 band-8). Widening is not the same as isolating.

FOUR ARMS, all same steps / seed / data:
  local   band +-8 only                       (current backbone)
  dual    band +-8 and full, separate Q/K/V, gate-fused     <- the hypothesis
  dual_no_gate   same, but the fusion weight is fixed at 0.5 (no learned gate)
  wide    one attention, dim*1.5, so the capacity MATCHES dual

The `wide` arm is the one that matters: dual adds ~46% parameters. If `wide` matches
`dual`, the gain was capacity, not branch isolation, and the hypothesis is dead.
`dual_no_gate` separates "two branches help" from "the learned gate helps".

GATE READOUT: the gate value per syllable is logged at the end. If the model pushes
the gate to a constant, it is not using both branches and the honest conclusion is
that the hypothesis failed, regardless of the accuracy number.

Usage:
    python exp_dual.py --steps 20000 --train-sents 750000
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pilot_balance as pb
from attn_model import AttnDiac
from attn import AttnBlock

ARMS = [
    ("local",        dict(window=8,   dim=384, heads=8, dual_gate=True)),
    ("dual",         dict(window="dual", dim=384, heads=8, dual_gate=True)),
    ("dual_no_gate", dict(window="dual", dim=384, heads=8, dual_gate=False)),
    ("wide",         dict(window=8,   dim=464, heads=8, dual_gate=True)),
]


def build(arm, nchar, max_c):
    cfg = dict(ARMS)[arm]
    m = AttnDiac(nchar, max_c, syl=192, dim=cfg["dim"], heads=cfg["heads"],
                 blocks=1, window=cfg["window"], out_std=1e-1)
    m.use_dual = cfg["window"] == "dual"
    m.dual_gate = cfg["dual_gate"]
    if m.use_dual and not m.dual_gate:
        for b in m.blocks:
            with torch.no_grad():
                b.att.gate.weight.zero_()
                b.att.gate.bias.zero_()
        for b in m.blocks:
            b.att.gate.weight.requires_grad_(False)
            b.att.gate.bias.requires_grad_(False)
    return m


def train_one(arm, seed, bundle, steps, bs=64, lr=2e-3):
    tr, cand, char2id, max_c, nchar, forms = bundle
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build(arm, nchar, max_c).to(pb.DEV)
    nparam = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    rng = np.random.default_rng(seed)
    gen = pb.make_batches(tr, cand, char2id, max_c, bs, rng, nchar=nchar, forms=forms)
    t0 = time.time()
    for step in range(steps):
        xc, cl, f, y, ln, mk, ok = next(gen)
        lg = pb.logits_of(model, xc, cl, ln)
        amb = mk & (ln > 1) & ok
        tgt = amb if amb.any() else ok
        ce = F.cross_entropy(lg.reshape(-1, lg.size(-1)), y.reshape(-1),
                             reduction="none").reshape(y.shape)
        loss = ce[tgt].mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if (step + 1) % max(steps // 4, 1) == 0:
            print(f"      step {step+1}/{steps} loss={loss.item():.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return model, nparam


@torch.inference_mode()
def gate_stats(model, sents, cand, char2id, nchar, max_c):
    """Mean / std of the fusion gate over real syllables; detects gate collapse."""
    if not getattr(model, "use_dual", False):
        return None
    gs = []
    for s in sents[:2000]:
        toks = s.split()
        rows = [((pb.strip_all(w.lower())), len(cand[pb.strip_all(w.lower())]))
                for w in toks if pb.strip_all(w.lower()) in cand]
        if not rows:
            continue
        T = len(rows)
        L = max(len(r[0]) for r in rows)
        xc = torch.full((1, T, L), nchar - 1, dtype=torch.long)
        cl = torch.ones((1, T), dtype=torch.long)
        for t, (st, _) in enumerate(rows):
            ids = [char2id.get(c, 0) for c in st[:64]]
            xc[0, t, :len(ids)] = torch.tensor(ids)
            cl[0, t] = max(len(ids), 1)
        h = model.proj(model.encode_syllables(xc.to(pb.DEV), cl.to(pb.DEV)))
        blk = model.blocks[0].att
        _, (_, _, g) = blk(model.blocks[0].n1(h), return_gate=True)
        gs.append(g.flatten().cpu())
    g = torch.cat(gs)
    return dict(mean=float(g.mean()), std=float(g.std()),
                frac_local05=float((g > 0.5).float().mean()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--train-sents", type=int, default=750_000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default="local,dual,dual_no_gate,wide")
    ap.add_argument("--out", default="exp_dual.json")
    a = ap.parse_args()

    print(f"device={pb.DEV} steps={a.steps} train_sents={a.train_sents:,}", flush=True)
    te_sents = [l.rstrip("\n") for l in open("data/te.txt", encoding="utf-8")][:20000]
    tr = pb.load_sents("data/tr.txt")[:a.train_sents]
    cand = json.load(open("data/cand.json"))
    forms = sorted(cand)
    forms_id = {s: i for i, s in enumerate(forms)}
    char2id = {c: i for i, c in enumerate(sorted({c for s in forms for c in s}), 1)}
    max_c = max(len(v) for v in cand.values())
    nchar = len(char2id) + 2
    print(f"train={len(tr):,} test={len(te_sents):,} max_c={max_c}", flush=True)

    import score_all as sa
    sa.CAND.update(cand)
    sa.FORMS_BY_STRIP.update({s: i for i, s in enumerate(forms)})
    bundle = (tr, cand, char2id, max_c, nchar, forms_id)

    res = []
    for arm in a.arms.split(","):
        print(f"\n=== {arm} ===", flush=True)
        model, nparam = train_one(arm, a.seed, bundle, a.steps, a.bs)

        @torch.inference_mode()
        def pred(sents, _m=model):
            return sa.predict(_m, sents, char2id, nchar, max_c)
        r = sa.score(te_sents, pred(te_sents))
        r["arm"], r["params"] = arm, nparam
        r["gate"] = gate_stats(model, te_sents, cand, char2id, nchar, max_c)
        res.append(r)
        print(sa.fmt(arm, r), flush=True)
        if r["gate"]:
            print(f"   gate: mean={r['gate']['mean']:.3f} std={r['gate']['std']:.3f} "
                  f"frac_local>0.5={r['gate']['frac_local05']:.3f}", flush=True)
        torch.save({"model": model.state_dict(), "arm": arm, "params": nparam,
                    "steps": a.steps, "char2id": char2id, "forms": forms_id,
                    "max_c": max_c, "nchar": nchar}, f"exp_ckpt_{arm}.pt")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        json.dump(res, open(a.out, "w"), indent=1, default=str)

    base = next(r for r in res if r["arm"] == "local")
    print("\n=== deltas vs `local` (syll_acc) ===")
    for r in res:
        if r["arm"] != "local":
            print(f"  {r['arm']:14s} {(r['syll_acc']-base['syll_acc'])*100:+.2f} pt "
                  f"  params {r['params']:,} ({(r['params']/base['params']-1)*100:+.0f}%)")
    print(f"\nwrote {a.out}")
