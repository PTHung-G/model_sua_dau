"""Tiny Vietnamese diacritic restorer.

Design
------
1. Input = characters of the stripped syllable -> a char BiGRU. No word embedding,
   so unseen syllables still get a representation.
2. Output = a candidate index among the legal accented forms of that stripped
   syllable (<= ~24), NOT a V-way classifier. Illegal output is impossible.
3. log_prior(candidate) is added as a constant offset, so the model starts at the
   unigram baseline and can only improve by using context. Without this offset a
   plain linear head simply memorises "pick candidate 0" (89% of tokens) and the
   context encoder never gets trained.
"""
import math
import time
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from data import load_sents
from vn import strip_all

if not torch.cuda.is_available():
    torch.set_num_threads(2)
MAXLEN, MAXCH = 48, 8
NEG = -1e9
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def build(tr_sents, te_sents, min_count=2):
    freq = Counter()
    for ws in tr_sents:
        for w in ws:
            freq[w.lower()] += 1
    vocab = {w for w, c in freq.items() if c >= min_count}
    cand = {}
    for w in vocab:
        cand.setdefault(strip_all(w), []).append(w)
    for s in cand:
        cand[s].sort(key=lambda w: (-freq[w], w))

    forms = sorted(cand)
    fid = {s: i for i, s in enumerate(forms)}
    max_c = max(len(cand[s]) for s in forms)
    pri = torch.full((len(forms), max_c), NEG)
    for s in forms:
        ws = cand[s]
        tot = sum(freq[w] for w in ws)
        pri[fid[s], :len(ws)] = torch.tensor([math.log(freq[w] / tot) for w in ws])

    chars = sorted({c for s in forms for c in s})
    ctoi = {c: i + 1 for i, c in enumerate(chars)}

    def enc(sents):
        out = []
        for ws in sents:
            rows = []
            for w in ws:
                wl = w.lower()
                s = strip_all(wl)
                if s not in cand or wl not in vocab:
                    continue
                rows.append((fid[s], [ctoi[c] for c in s][:MAXCH],
                             cand[s].index(wl), len(cand[s])))
            if len(rows) >= 6:
                out.append(rows[:MAXLEN])
        return out

    return dict(cand=cand, pri=pri, max_c=max_c, nchar=len(chars) + 1,
                tr=enc(tr_sents), te=enc(te_sents), vocab=len(vocab))


def batches(S, bs, shuffle=True, seed=0):
    idx = list(range(len(S)))
    if shuffle:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(S), generator=g).tolist()
    for i in range(0, len(idx), bs):
        ch = idx[i:i + bs]
        B = len(ch)
        T = max(len(S[j]) for j in ch)
        L = max(len(r[1]) for j in ch for r in S[j])
        xc = torch.zeros(B, T, L, dtype=torch.long)
        cl = torch.ones(B, T, dtype=torch.long)
        f = torch.zeros(B, T, dtype=torch.long)
        y = torch.zeros(B, T, dtype=torch.long)
        ln = torch.zeros(B, T, dtype=torch.long)
        m = torch.zeros(B, T, dtype=torch.bool)
        for r, j in enumerate(ch):
            for t, (fi, cs, ci, n) in enumerate(S[j]):
                xc[r, t, :len(cs)] = torch.tensor(cs)
                cl[r, t] = len(cs); f[r, t] = fi; y[r, t] = ci
                ln[r, t] = n; m[r, t] = True
        yield (xc.to(DEV), cl.to(DEV), f.to(DEV), y.to(DEV),
               ln.to(DEV), m.to(DEV))


class Tiny(nn.Module):
    def __init__(self, nchar, max_c, ch_dim=16, syl=32, hidden=64, ctx_layers=1, drop=0.1):
        super().__init__()
        self.cemb = nn.Embedding(nchar, ch_dim, padding_idx=0)
        self.cgru = nn.GRU(ch_dim, syl // 2, batch_first=True, bidirectional=True)
        self.cnorm = nn.LayerNorm(syl)
        self.drop = nn.Dropout(drop)
        self.ctx = nn.GRU(syl, hidden // 2, num_layers=ctx_layers, batch_first=True,
                          bidirectional=True, dropout=drop if ctx_layers > 1 else 0.0)
        self.out = nn.Linear(hidden, max_c)
        # IMPORTANT: do NOT zero-init the head. A zero weight makes d(loss)/d(encoder)
        # exactly 0, so the encoder never receives a single gradient and the model is
        # stuck at the unigram prior forever (measured: all encoder grads == 0.0).
        # A small random init keeps the start close to the prior while leaving a path.
        nn.init.normal_(self.out.weight, std=1e-2)
        nn.init.zeros_(self.out.bias)

    def forward(self, xc, cl):
        B, T, L = xc.shape
        h, _ = self.cgru(self.cemb(xc.reshape(B * T, L)))
        i = (cl.reshape(-1) - 1).clamp(min=0)          # last REAL character
        h = self.cnorm(h[torch.arange(h.size(0)), i]).view(B, T, -1)
        h, _ = self.ctx(self.drop(h))
        return self.out(h)


def logits_of(model, xc, cl, f, ln, pri):
    lg = model(xc, cl) + pri[f]
    ar = torch.arange(lg.size(-1), device=lg.device)[None, None, :]
    return lg.masked_fill(ar >= ln[..., None], NEG)


def evaluate(model, S, pri, bs=128):
    model.eval()
    tot = ok = ta = oa = tu = ou = 0
    with torch.no_grad():
        for xc, cl, f, y, ln, m in batches(S, bs, shuffle=False):
            pred = logits_of(model, xc, cl, f, ln, pri).argmax(-1)
            hit = (pred == y) & m
            tot += m.sum().item(); ok += hit.sum().item()
            a = (ln > 1) & m; u = (ln == 1) & m
            ta += a.sum().item(); oa += (hit & a).sum().item()
            tu += u.sum().item(); ou += (hit & u).sum().item()
    return ok / tot, oa / max(ta, 1), ou / max(tu, 1), tot


def run(cfg, P, epochs=3, bs=64, lr=3e-3, verbose=True):
    model = Tiny(P["nchar"], P["max_c"], **cfg)
    npar = sum(p.numel() for p in model.parameters())
    tr, pri = P["tr"], P["pri"]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(tr) / bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    t0 = time.time()
    hist = []
    for e in range(epochs):
        model.train(); tot = 0; nb = 0
        for xc, cl, f, y, ln, m in batches(tr, bs, seed=e):
            lg = logits_of(model, xc, cl, f, ln, pri)
            loss = F.cross_entropy(lg[m], y[m])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()
            tot += loss.item(); nb += 1
        acc, aa, au, n = evaluate(model, P["te"], pri)
        hist.append(acc)
        if verbose:
            print(f"    ep{e+1} loss={tot/nb:.3f} acc={acc:.4f} amb={aa:.4f} "
                  f"un={au:.4f} ({time.time()-t0:.0f}s)", flush=True)
    return dict(cfg=cfg, params=npar, acc=hist[-1], hist=hist, secs=time.time() - t0)


if __name__ == "__main__":
    ntr = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    ep = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    print(f"=== train {ntr} sents, {ep} epochs ===", flush=True)
    tr = load_sents("train.txt", ntr); te = load_sents("test.txt", 2000)
    P = build(tr, te, min_count=2)
    print(f"max_c={P['max_c']} nchar={P['nchar']} vocab={P['vocab']}", flush=True)
    # sanity: what does the pure prior (0 context params) get?
    zero = Tiny(P["nchar"], P["max_c"])
    acc, aa, au, n = evaluate(zero, P["te"], P["pri"])
    print(f"prior-only (dung unigram)        acc={acc:.4f} amb={aa:.4f} (n={n})", flush=True)
    for cfg in [dict(syl=16, hidden=32), dict(syl=32, hidden=64), dict(syl=48, hidden=96),
                dict(syl=32, hidden=64, ctx_layers=2)]:
        r = run(cfg, P, epochs=ep)
        print(f"{str(cfg):46s} params={r['params']:>7,} acc={r['acc']:.4f}", flush=True)
