"""Statistical baselines: context-free (unigram) and bigram/trigram LM."""
import sys
from collections import defaultdict
from data import load_sents
from vn import strip_all


def accuracies(tr, te, order=0, alpha=0.1):
    """order=0: most frequent accented form per stripped type.
       order=1: P(acc | previous accented syllable).  order=2: trigram-style ctx."""
    uni = defaultdict(lambda: defaultdict(int))
    cond = defaultdict(lambda: defaultdict(int))
    for ws in tr:
        prev = "<s>"
        prev2 = "<s>"
        for w in ws:
            wl = w.lower()
            s = strip_all(wl)
            uni[s][wl] += 1
            if order >= 1:
                cond[prev][wl] += 1
            if order >= 2:
                cond[prev2 + "|" + prev][wl] += 1
            prev2, prev = prev, wl

    best = {s: max(c.items(), key=lambda kv: kv[1])[0] for s, c in uni.items()}

    tot = ok = 0
    ok_amb = tot_amb = 0
    ok_un = tot_un = 0
    for ws in te:
        prev = "<s>"
        prev2 = "<s>"
        for w in ws:
            wl = w.lower()
            s = strip_all(wl)
            if s not in uni:
                prev2, prev = prev, wl
                continue
            cands = uni[s]
            if order == 0 or len(cands) == 1:
                pred = best[s]
            else:
                key = prev if order == 1 else prev2 + "|" + prev
                stats = cond.get(key, {})
                pool = {a: c for a, c in stats.items() if a in cands}
                pred = max(pool.items(), key=lambda kv: kv[1])[0] if pool else best[s]
            tot += 1
            hit = pred == wl
            ok += hit
            if len(cands) > 1:
                tot_amb += 1
                ok_amb += hit
            else:
                tot_un += 1
                ok_un += hit
            prev2, prev = prev, wl
    return ok / tot, ok_amb / tot_amb, ok_un / tot_un, tot


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300000
    print("loading...")
    tr = load_sents("train.txt", n)
    te = load_sents("test.txt", 20000)
    print(f"train sents {len(tr)}, test sents {len(te)}\n")
    for order, name in ((0, "unigram (bảng tra)"), (1, "bigram (1 âm tiết trước)"),
                        (2, "trigram (2 âm tiết trước)")):
        acc, acc_a, acc_u, tot = accuracies(tr, te, order=order)
        print(f"{name:32s} acc={acc:.4f}  | nhập nhằng {acc_a:.4f}  | đơn trị {acc_u:.4f}  (n={tot:,})")
