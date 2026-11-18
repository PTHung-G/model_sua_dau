"""Full-scale (300k) training + evaluation of the tiny diacritic restorer.

Runs on CPU or GPU. Key fixes carried over from the CPU debugging:
  * head is NOT zero-initialised (a zero head gives the encoder exactly zero gradient)
  * fixed learning rate (no annealing to ~0)
  * best-checkpoint selection on a validation split
"""
import json
import math
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import load_sents
from tiny2 import build, batches, Tiny, logits_of, evaluate

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def split(P, frac=0.05, seed=1234):
    tr = P["tr"]
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(tr), generator=g).tolist()
    k = int(len(tr) * frac)
    va = [tr[i] for i in idx[:k]]
    trn = [tr[i] for i in idx[k:]]
    return trn, va


def train_one(cfg, P, epochs=12, bs=256, lr=2e-3, wd=0.01, tag=""):
    trn, va = split(P)
    model = Tiny(P["nchar"], P["max_c"], **cfg).to(DEV)
    npar = sum(p.numel() for p in model.parameters())
    pri = P["pri"].to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best = 0.0
    best_state = None
    hist = []
    t0 = time.time()
    for e in range(epochs):
        model.train()
        tot = nb = 0
        for xc, cl, f, y, ln, m in batches(trn, bs, seed=e):
            xc, cl, f, y, ln, m = (t.to(DEV) for t in (xc, cl, f, y, ln, m))
            lg = logits_of(model, xc, cl, f, ln, pri)
            loss = F.cross_entropy(lg[m], y[m])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        vacc = evaluate(model, va, P["pri"].to("cpu"), bs=256)[0]
        hist.append(round(vacc, 4))
        if vacc > best:
            best = vacc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if e % 2 == 0 or e == epochs - 1:
            print(f"  [{tag}] ep{e+1} loss={tot/nb:.4f} val={vacc:.4f} "
                  f"best={best:.4f} ({time.time()-t0:.0f}s)", flush=True)
    model.load_state_dict(best_state)
    acc, aa, au, n = evaluate(model, P["te"], P["pri"].to("cpu"), bs=256)
    print(f"  [{tag}] FINAL params={npar:,} test_acc={acc:.4f} amb={aa:.4f} ({time.time()-t0:.0f}s)", flush=True)
    return dict(tag=tag, cfg=cfg, params=npar, test=acc, val_best=best, hist=hist)


if __name__ == "__main__":
    torch.set_num_threads(4 if DEV == "cpu" else 1)
    print(f"device={DEV}", flush=True)
    t = time.time()
    tr = load_sents("train.txt", int(__import__("os").environ.get("NTR","300000")))
    te = load_sents("test.txt", 20000)
    P = build(tr, te, min_count=2)
    print(f"built: train={len(P['tr'])} test={len(P['te'])} max_c={P['max_c']} "
          f"vocab={P['vocab']} ({time.time()-t:.0f}s)", flush=True)

    zero = Tiny(P["nchar"], P["max_c"])
    print(f"prior-only acc={evaluate(zero, P['te'], P['pri'])[0]:.4f}", flush=True)

    results = []
    for cfg, tag in [
        (dict(syl=16, hidden=32), "5k"),
        (dict(syl=32, hidden=64), "17k"),
        (dict(syl=64, hidden=128), "60k"),
        (dict(syl=128, hidden=256), "230k"),
    ]:
        r = train_one(cfg, P, epochs=12, tag=tag)
        results.append(r)
        json.dump(results, open("results_full.json", "w"), indent=1)
    print("\n=== SUMMARY ===")
    for r in results:
        print(f"{r['tag']:6s} params={r['params']:>8,} test={r['test']:.4f}")
