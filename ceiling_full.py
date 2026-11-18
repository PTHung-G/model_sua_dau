"""Measure the real information ceiling at full scale (300k train sentences).

For each context order k we give the predictor the GOLD accented forms of the
previous k syllables -- an information source no real system has. This is an
upper bound on what any context-based model of order k can do.

Also reports:
  * how much of the residual error is concentrated on ambiguous positions
  * the same for a "bag of candidates" predictor (no order info)
"""
import sys
from collections import Counter, defaultdict
from data import load_sents
from vn import strip_all


def toks(sents, lo=6, hi=48):
    out = []
    for ws in sents:
        g = [w.lower() for w in ws]
        if len(g) >= lo:
            out.append(g[:hi])
    return out


def ceiling(G, GT, order):
    uni = defaultdict(Counter)
    cond = defaultdict(Counter)
    for g in G:
        for i, w in enumerate(g):
            uni[strip_all(w)][w] += 1
            if order >= 1 and i >= 1:
                key = tuple(g[max(0, i - order):i])
                cond[key][w] += 1
    ok = n = 0
    ok_amb = n_amb = ok_un = n_un = 0
    for g in GT:
        for i, w in enumerate(g):
            s = strip_all(w)
            if s not in uni:
                continue
            c = uni[s]
            cands = set(c)
            pool = {}
            if order >= 1 and i >= 1:
                key = tuple(g[max(0, i - order):i])
                pool = {a: v for a, v in cond.get(key, {}).items() if a in cands}
            pred = max(pool, key=pool.get) if pool else max(c, key=c.get)
            n += 1
            hit = pred == w
            ok += hit
            if len(cands) > 1:
                n_amb += 1; ok_amb += hit
            else:
                n_un += 1; ok_un += hit
    return ok / n, n, ok_amb / max(n_amb, 1), n_amb, ok_un / max(n_un, 1), n_un


if __name__ == "__main__":
    ntr = int(sys.argv[1]) if len(sys.argv) > 1 else 300000
    print(f"train {ntr} sents", flush=True)
    G = toks(load_sents("train.txt", ntr))
    GT = toks(load_sents("test.txt", 20000))
    print(f"-> {len(G)} train sents / {len(GT)} test sents", flush=True)
    print(f"{'ngữ cảnh':28s} {'acc':>8s} {'n':>9s} {'| nhập nhằng':>13s} {'| đơn trị':>10s}")
    for o, name in [(0, "0 (không ngữ cảnh)"), (1, "1 âm tiết gold"),
                    (2, "2 âm tiết gold"), (3, "3 âm tiết gold")]:
        acc, n, aa, na, au, nu = ceiling(G, GT, o)
        print(f"{name:28s} {acc:8.4f} {n:9,} {aa:13.4f} {au:10.4f}", flush=True)
