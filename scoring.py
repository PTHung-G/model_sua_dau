"""Score metrics that are actually meaningful.

Fixes three defects found in the previous sweep:
  1. buckets were computed from the rank-1 prior form (always frequent) -> everything
     landed in "20+". Now a token is bucketed by the training frequency of its GOLD
     form, so the buckets describe the task, not the prior.
  2. train accuracy was computed on the training split after best-checkpoint reload,
     while test used a different tokenisation path -> train < test was possible.
     Now both go through the SAME function.
  3. no baselines on the same test set -> attention could not be shown to beat a
     lookup table. Now unigram / bigram / prior-only are scored identically.
"""
import torch
from collections import Counter, defaultdict

NEG = -1e9


def batches(S, bs, shuffle=False, seed=0, device="cpu"):
    idx = list(range(len(S)))
    if shuffle:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(S), generator=g).tolist()
    for i in range(0, len(S), bs):
        chunk = [S[j] for j in idx[i:i + bs]]
        B = len(chunk)
        T = max(len(r) for r in chunk)
        xc = torch.zeros(B, T, 8, dtype=torch.long)
        cl = torch.zeros(B, T, dtype=torch.long)
        ff = torch.zeros(B, T, dtype=torch.long)
        y = torch.zeros(B, T, dtype=torch.long)
        ln = torch.ones(B, T, dtype=torch.long)
        m = torch.zeros(B, T, dtype=torch.bool)
        for b, rows in enumerate(chunk):
            for t, (fi, cs, ci, k) in enumerate(rows):
                xc[b, t, :len(cs)] = torch.tensor(cs[:8])
                cl[b, t] = min(len(cs), 8)
                ff[b, t] = fi
                y[b, t] = ci
                ln[b, t] = k
                m[b, t] = True
        yield (xc.to(device), cl.to(device), ff.to(device), y.to(device),
               ln.to(device), m.to(device))


def gold_freq_buckets(tr_sents, cand):
    """Map (form, candidate slot) -> bucket of how often THAT gold form was seen.

    Bucketing per (form, slot) rather than per form is essential: slot 0 is by
    construction the most frequent form, so bucketing by slot 0 labels every
    ambiguous token 'frequent' and measures nothing."""
    ftr = Counter(w.lower() for ws in tr_sents for w in ws)
    gold_freq_buckets._names = sorted(cand)
    out = {}
    for s in cand:
        for ci, w in enumerate(cand[s]):
            c = ftr.get(w, 0)
            out[(s, ci)] = ("unseen" if c == 0 else "rare" if c < 5 else
                            "mid" if c < 20 else "freq")
    return out


@torch.no_grad()
def score_predictor(predict_fn, S, buckets, bs=128, device="cpu"):
    """Shared scorer. predict_fn(xc, cl, f, ln) -> (B,T) predicted candidate index."""
    tot = Counter(); ok = Counter()
    for xc, cl, f, y, ln, m in batches(S, bs, device=device):
        pred = predict_fn(xc, cl, f, ln)
        hit = (pred == y) & m
        # bucket PER TOKEN using that token's own gold form AND gold slot
        fl = f.tolist(); yl = y.tolist()
        names = gold_freq_buckets._names
        for b in range(f.size(0)):
            for t in range(f.size(1)):
                if not m[b, t]:
                    continue
                k = buckets.get((names[fl[b][t]], yl[b][t]), "freq")
                tot[k] += 1
                if hit[b, t]:
                    ok[k] += 1
    n = sum(tot.values())
    out = {k: (ok[k] / tot[k], tot[k]) for k in tot}
    out["ALL"] = (sum(v[0] * v[1] for v in out.values()) / n, n) if n else (0.0, 0)
    return out


class UnigramPredictor:
    """argmax of the training-frequency prior. Zero learnable parameters."""

    def __init__(self, pri):
        self.pri = pri

    def __call__(self, xc, cl, f, ln):
        return self.pri.to(xc.device)[f].argmax(-1)


class BigramPredictor:
    """P(gold | previous gold form) -- uses the GOLD previous syllable, so it is an
    ORACLE-context upper bound for any order-1 model. No neural training."""

    def __init__(self, tr_sents, cand, pri, n_back=1):
        from vn import strip_all
        forms = sorted(cand)
        self.fid = {s: i for i, s in enumerate(forms)}
        self.n_back = n_back
        table = defaultdict(Counter)
        for ws in tr_sents:
            ctx = []
            for w in ws:
                wl = w.lower(); s = strip_all(wl)
                if s in self.fid and wl in cand[s]:
                    ci = cand[s].index(wl)
                    fid_ = self.fid[s]
                    if len(ctx) >= n_back:
                        table[tuple(ctx[-n_back:])][ci] += 1
                    ctx.append(fid_)
                    if len(ctx) > n_back:
                        ctx.pop(0)
                else:
                    ctx = []          # unknown token breaks the context
        self.table = {k: v.most_common(1)[0][0] for k, v in table.items()}
        self.pri = pri

    def __call__(self, xc, cl, f, ln):
        B, T = f.shape
        out = self.pri.to(xc.device)[f].argmax(-1)
        for b in range(B):
            for t in range(1, T):
                nb = min(self.n_back, t)
                key = tuple(int(f[b, t - nb + j]) for j in range(nb))
                if key in self.table:
                    c = self.table[key]
                    if c < int(ln[b, t]):
                        out[b, t] = c
        return out


def fmt_row(name, r):
    parts = [f"{name:10s}"]
    for k in ("ALL", "freq", "mid", "rare", "unseen"):
        if k in r:
            parts.append(f"{k}={r[k][0]:.4f}")
    return "  ".join(parts)
