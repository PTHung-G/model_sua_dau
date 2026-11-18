"""Sanity checks on the scorer before any GPU is spent."""
import json, torch, numpy as np
import evaluate as ev

cand = json.load(open("data/cand.json"))
names = sorted(cand)
sents = [l.split() for l in open("data/va.txt", encoding="utf-8")][:3000]
chars = sorted({c for s in names for c in s})
c2i = {c: i for i, c in enumerate(chars, start=1)}
toks = ev.Tokens(sents, cand, c2i)
print(f"tok={toks.n_tok:,} sents={len(toks.sents):,} uncovered={toks.n_uncovered:,}")

# 1) oracle -> 100%
def oracle(xc, cl, f, ln):
    return GOLD(xc, cl, f, ln)
def stash_gold(bs=300):
    for b in toks.batches(bs):
        yield b
gold_batches = list(toks.batches(300))
it = iter(gold_batches)
GOLD = lambda xc, cl, f, ln: next(it)[3]
r = ev.evaluate(oracle, toks, bs=300)
print("oracle      :", ev.fmt("oracle", r))
# an oracle is right on every COVERED token and wrong on uncovered ones,
# so its ALL score must equal the coverage rate exactly
assert abs(r["all"] - r["coverage"]) < 1e-9, "oracle must score exactly coverage"

# 2) always slot0 must equal base rate computed by direct counting
# Reconstruct exactly what the scorer counts: rebuild rows the same way Tokens
# does (sentence kept if >=2 covered tokens, uncovered token -> slot -1) and
# verify against a second independent implementation.
from vn import strip_all
cnt = n = 0
for ws in sents:
    row = []
    for w in ws:
        wl = w.lower(); st = strip_all(wl)
        if st in cand:
            cs = cand[st]
            row.append(cs.index(wl) if wl in cs else -1)
    if len(row) >= 2:
        for g in row:
            n += 1
            cnt += (g == 0)
it2 = iter(gold_batches)
r0 = ev.evaluate(lambda xc, cl, f, ln: torch.zeros_like(f), toks, bs=300)
print(f"always0     : {ev.fmt('slot0', r0)}")
print(f"direct count: {cnt/n:.4f}  scorer: {r0['all']:.4f}")
assert abs(cnt / n - r0["all"]) < 1e-9, "scorer disagrees with direct counting"
assert r0["fixed"] == 0, "slot0 can never fix a baseline error"

# 3) batch size invariance
r0b = ev.evaluate(lambda xc, cl, f, ln: torch.zeros_like(f), toks, bs=7)
print(f"bs=7 vs bs=300: {r0b['all']:.6f} vs {r0['all']:.6f}")
assert abs(r0b["all"] - r0["all"]) < 1e-9
print("\nALL SCORER CHECKS PASSED")
