"""Final decisive experiment at 300k.

Scores every system on the subset of test tokens that are ACTUALLY LEARNABLE,
i.e. the (previous form, form) pair was seen >=2 times in training. On this subset
a context-aware model has something to use; on the rest nothing can succeed and
including it only dilutes the comparison.

Systems:
    pick0     always the most frequent accented form (no parameters)
    order1    counts of (prev form) -> candidate, argmax          (oracle-ish baseline)
    nn        attention model, prior removed
"""
import json
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from data import load_sents
from fixbuild import build_fair
from scoring import batches, gold_freq_buckets
from attn_model import AttnDiac

if not torch.cuda.is_available():
    torch.set_num_threads(2)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NEG = -1e9


def lg_of(model, xc, cl, f, ln, pri, use_prior=False):
    lg = model(xc, cl)
    if use_prior:
        lg = lg + pri.to(lg.device)[f]
    ar = torch.arange(lg.size(-1), device=lg.device)[None, None, :]
    return lg.masked_fill(ar >= ln[..., None], NEG)


def build_learnable(tr_sents, te_rows, names, cand):
    """Flag test tokens whose (prev_form, form) pair was seen >=2x in train."""
    from vn import strip_all
    form_of = {w: strip_all(w.lower()) for ws in tr_sents for w in ws}
    pair = Counter()
    for ws in tr_sents:
        prev = None
        for w in ws:
            wl = w.lower()
            if prev is not None:
                pair[(prev, wl)] += 1
            prev = wl
    mask = []
    for row in te_rows:
        flags = []
        for i, (fi, cs, ci, k) in enumerate(row):
            gold = cand[names[fi]][ci]
            if i == 0:
                flags.append(False)
            else:
                pfi = row[i - 1][0]; pci = row[i - 1][2]
                prev = cand[names[pfi]][pci]
                flags.append(pair.get((prev, gold), 0) >= 2)
        mask.append(flags)
    tot = sum(len(r) for r in te_rows)
    learn = sum(sum(m) for m in mask)
    return mask, learn, tot


def score(pred_fn, P, mask, tag):
    """mask is aligned with P["te"] (sentence list), not with batch index."""
    tot = ok = tot_l = ok_l = 0
    bi = 0
    for xc, cl, f, y, ln, m in batches(P["te"], 128, device=DEV):
        pred = pred_fn(xc, cl, f, ln)
        hit = (pred == y) & m
        for b in range(xc.size(0)):
            sent_mask = mask[bi + b] if bi + b < len(mask) else []
            for t in range(xc.size(1)):
                if not m[b, t]:
                    continue
                tot += 1
                if hit[b, t]:
                    ok += 1
                if t < len(sent_mask) and sent_mask[t]:
                    tot_l += 1
                    if hit[b, t]:
                        ok_l += 1
        bi += xc.size(0)
    print(f"{tag:10s} ALL={ok/max(tot,1):.4f}  LEARNABLE={ok_l/max(tot_l,1):.4f} "
          f"(n={tot_l:,})", flush=True)
    return dict(tag=tag, all=ok / max(tot, 1), learnable=ok_l / max(tot_l, 1), n=tot_l)


class Pick0:
    def __call__(self, xc, cl, f, ln):
        return torch.zeros_like(f)


class Order1:
    def __init__(self, tr_sents, cand, pri):
        self.table = Counter()
        for ws in tr_sents:
            prev = None
            for w in ws:
                wl = w.lower()
                if prev is not None:
                    self.table[(prev, wl)] += 1
                prev = wl
        self.cand = cand
        self.names = sorted(cand)
        self.pri = pri

    def __call__(self, xc, cl, f, ln):
        """For each test token, choose the candidate with the most counts given the
        PREVIOUS form (taking the previous form as seen, i.e. oracle context)."""
        out = self.pri.to(xc.device)[f].argmax(-1)
        fl = f.tolist()
        cache = {}
        for b in range(f.size(0)):
            for t in range(1, f.size(1)):
                pf, cf = fl[b][t - 1], fl[b][t]
                key = (pf, cf)
                if key not in cache:
                    best, bestc = -1, 0
                    for pci, pc in enumerate(self.cand[self.names[pf]]):
                        for ci, gc in enumerate(self.cand[self.names[cf]]):
                            c = self.table.get((pc, gc), 0)
                            if c > bestc:
                                bestc, best = c, ci
                    cache[key] = best
                if cache[key] >= 0 and cache[key] < int(ln[b, t]):
                    out[b, t] = cache[key]
        return out


if __name__ == "__main__":
    ntr = int(sys.argv[1]) if len(sys.argv) > 1 else 300000
    ep = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    print(f"device={DEV} ntr={ntr} epochs={ep}", flush=True)
    t0 = time.time()
    tr = load_sents("train.txt", ntr)
    te = load_sents("test.txt", 3000)
    P = build_fair(tr, te)
    names = sorted(P["cand"])
    pri = P["pri"]
    mask, learn, tot = build_learnable(tr, P["te"], names, P["cand"])
    print(f"built train={len(P['tr'])} test={len(P['te'])} "
          f"learnable={learn}/{tot}={learn/tot:.3f} ({time.time()-t0:.0f}s)", flush=True)

    res = []
    res.append(score(Pick0(), P, mask, "pick0"))
    res.append(score(Order1(tr, P["cand"], pri), P, mask, "order1"))

    torch.manual_seed(0)
    m = AttnDiac(P["nchar"], P["max_c"], syl=32, dim=64, window=8).to(DEV)
    nn.init.normal_(m.out.weight, std=1e-1); nn.init.zeros_(m.out.bias)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=0.0)
    trb = list(batches(P["tr"], 64, device=DEV))
    for e in range(ep):
        m.train()
        order = torch.randperm(len(trb)).tolist()
        tl = tn = 0
        for i in order:
            xc, cl, f, y, ln, mk = trb[i]
            lg = lg_of(m, xc, cl, f, ln, pri)
            loss = F.cross_entropy(lg[mk], y[mk])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            tl += loss.item() * mk.sum().item(); tn += mk.sum().item()
        print(f"  ep{e+1} train_loss={tl/tn:.4f} ({time.time()-t0:.0f}s)", flush=True)
    fn = lambda xc, cl, f, ln: lg_of(m, xc, cl, f, ln, pri, False).argmax(-1)
    res.append(score(fn, P, mask, "attention"))
    json.dump(res, open("final300k.json", "w"), indent=1)
    print("\n=== verdict ===")
    for r in res:
        print(f"{r['tag']:10s} LEARNABLE={r['learnable']:.4f}")
