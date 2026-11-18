import json, sys
from collections import Counter
from build_clean import build_vocab_from_train
from vn import strip_all
import evaluate as ev

def load(p):
    return [l.split() for l in open(p, encoding="utf-8")]

tr, va, te = load("data/tr.txt"), load("data/va.txt"), load("data/te.txt")
cand, freq = build_vocab_from_train(tr, min_freq=2)
print(f"train sents {len(tr):,}  forms {len(cand):,}  max_c {max(len(v) for v in cand.values())}")
for name, sents in (("val", va), ("test", te)):
    n = miss = 0
    for ws in sents:
        for w in ws:
            n += 1
            s = strip_all(w.lower())
            if s not in cand or w.lower() not in cand[s]:
                miss += 1
    print(f"{name}: {n:,} tokens, not covered {miss:,} = {miss/n:.4f}  coverage {1-miss/n:.4f}")
json.dump({s: cand[s] for s in cand}, open("data/cand.json", "w"), ensure_ascii=False)
print("wrote data/cand.json")
