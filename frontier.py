"""Fair size-vs-accuracy frontier: everything evaluated on the SAME test set.

For each training size we report
  * unigram table   (0 params)
  * bigram table    (0 params)
  * tiny neural models of increasing size
so the comparison is apples to apples.
"""
import json
import sys
import time
import torch
from collections import Counter, defaultdict
from data import load_sents
from tiny2 import build, Tiny, evaluate, run
from vn import strip_all

torch.set_num_threads(2)


def ngram_baselines(tr_sents, te_sents, cand, order=1):
    uni, cond = defaultdict(Counter), defaultdict(Counter)
    for ws in tr_sents:
        p = "<s>"
        for w in ws:
            wl = w.lower()
            if strip_all(wl) not in cand:
                continue
            uni[strip_all(wl)][wl] += 1
            if order >= 1:
                cond[p][wl] += 1
            p = wl
    ok = n = 0
    for ws in te_sents:
        p = "<s>"
        for w in ws:
            wl = w.lower(); s = strip_all(wl)
            if s not in uni:
                continue
            c = uni[s]
            if order == 0 or len(c) == 1:
                pred = max(c, key=c.get)
            else:
                pool = {a: v for a, v in cond.get(p, {}).items() if a in c}
                pred = max(pool, key=pool.get) if pool else max(c, key=c.get)
            n += 1; ok += (pred == wl); p = wl
    return ok / n, n


if __name__ == "__main__":
    sizes = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["20000"])]
    ep = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    te = load_sents("test.txt", 2000)
    out = []
    for ntr in sizes:
        print(f"\n=== train {ntr} sents ===", flush=True)
        t0 = time.time()
        tr = load_sents("train.txt", ntr)
        P = build(tr, te, min_count=2)
        acc_u, n = ngram_baselines(tr, te, P["cand"], order=0)
        acc_b, _ = ngram_baselines(tr, te, P["cand"], order=1)
        print(f"  max_c={P['max_c']} vocab={P['vocab']} test n={n}", flush=True)
        print(f"  unigram (0 params)  acc={acc_u:.4f}", flush=True)
        print(f"  bigram  (0 params)  acc={acc_b:.4f}", flush=True)
        res = dict(n_train=ntr, n_test=n, unigram=acc_u, bigram=acc_b, models=[])
        for cfg in [dict(syl=16, hidden=32), dict(syl=32, hidden=64), dict(syl=64, hidden=128)]:
            r = run(cfg, P, epochs=ep)
            res["models"].append(dict(params=r["params"], acc=r["acc"], cfg=cfg))
            print(f"  {str(cfg):40s} params={r['params']:>7,} acc={r['acc']:.4f}", flush=True)
        out.append(res)
        json.dump(out, open("frontier.json", "w"), indent=1)
        print(f"  ({time.time()-t0:.0f}s)", flush=True)
    print("\nDONE")
