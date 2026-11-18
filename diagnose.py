"""Diagnose WHY the tiny model fails to learn. One factor at a time.

Factors under test:
  A. weight_decay on the zero-initialised head  -> pulls it back to the prior
  B. OneCycleLR annealing the LR to ~0          -> freezes it near the prior
  C. not enough epochs
  D. context window: does the model even see enough?
"""
import math, sys, time, torch, torch.nn.functional as F
from data import load_sents
from tiny2 import build, batches, logits_of, evaluate, Tiny

torch.set_num_threads(2)


def train_variant(P, epochs=6, bs=64, lr=1e-3, wd=0.0, sched=None, hidden=96, syl=48):
    model = Tiny(P["nchar"], P["max_c"], syl=syl, hidden=hidden)
    npar = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    tr, pri = P["tr"], P["pri"]
    steps = epochs * math.ceil(len(tr) / bs)
    sch = None
    if sched == "onecycle":
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    elif sched == "cosine":
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr / 10)
    hist = []
    best = 0.0
    for e in range(epochs):
        model.train(); tot = 0; nb = 0
        for xc, cl, f, y, ln, m in batches(tr, bs, seed=e):
            lg = logits_of(model, xc, cl, f, ln, pri)
            loss = F.cross_entropy(lg[m], y[m])
            opt.zero_grad(); loss.backward()
            nn_grad_clip(model)
            opt.step()
            if sch:
                sch.step()
            tot += loss.item(); nb += 1
        acc, aa, au, n = evaluate(model, P["te"], pri)
        hist.append(round(acc, 4)); best = max(best, acc)
        print(f"    ep{e+1} loss={tot/nb:.3f} acc={acc:.4f}", flush=True)
    return dict(params=npar, last=hist[-1], best=best, hist=hist)


def nn_grad_clip(model, v=1.0):
    torch.nn.utils.clip_grad_norm_(model.parameters(), v)


if __name__ == "__main__":
    tr = load_sents("train.txt", 20000); te = load_sents("test.txt", 2000)
    P = build(tr, te, min_count=2)
    print(f"max_c={P['max_c']} vocab={P['vocab']}", flush=True)

    zero = Tiny(P["nchar"], P["max_c"])
    acc0, _, _, _ = evaluate(zero, P["te"], P["pri"])
    print(f"PRIOR-ONLY (0 context)            acc={acc0:.4f}", flush=True)

    for name, kw in [
        ("wd=0.01  onecycle lr=3e-3", dict(lr=3e-3, wd=0.01, sched="onecycle")),
        ("wd=0.00  onecycle lr=3e-3", dict(lr=3e-3, wd=0.0, sched="onecycle")),
        ("wd=0.00  constant lr=1e-3", dict(lr=1e-3, wd=0.0, sched=None)),
        ("wd=0.00  cosine   lr=2e-3", dict(lr=2e-3, wd=0.0, sched="cosine")),
    ]:
        print(f"\n== {name} ==", flush=True)
        r = train_variant(P, epochs=6, **kw)
        print(f"  -> best={r['best']:.4f} last={r['last']:.4f} hist={r['hist']}", flush=True)
