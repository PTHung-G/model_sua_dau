"""Truly tiny Vietnamese diacritic restoration.

Design (why it is small):
  * input is the *characters* of the stripped syllable  -> ~30 char embeddings,
    so unseen syllables still work;
  * output is a *candidate index* within the (<=29) legal accented forms of that
    stripped syllable -> Linear(hidden, max_c) instead of Linear(hidden, V=112k).

This is constrained decoding by construction: an illegal syllable cannot be
produced.
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

torch.set_num_threads(2)  # 6 threads on tiny tensors is 20x SLOWER (oversubscription)
MAXLEN = 48
MAXCH = 8


def build(tr_sents, te_sents, min_count=2):
    freq = Counter()
    for ws in tr_sents:
        for w in ws:
            freq[w.lower()] += 1
    vocab = {w for w, c in freq.items() if c >= min_count}
    cand = {}
    for w in vocab:
        cand.setdefault(strip_all(w), []).append(w)
    # deterministic candidate order: by frequency desc, then alpha
    for s in cand:
        cand[s].sort(key=lambda w: (-freq[w], w))
    chars = sorted({c for s in cand for c in s} | {"<pad>", "<unk>"})
    ctoi = {c: i for i, c in enumerate(chars)}
    idx = {s: {w: j for j, w in enumerate(v)} for s, v in cand.items()}
    max_c = max(len(v) for v in cand.values())

    def enc(sents):
        out = []
        for ws in sents:
            rows = []
            for w in ws:
                wl = w.lower()
                s = strip_all(wl)
                if s not in cand or wl not in vocab:
                    continue
                rows.append(([ctoi[c] for c in s][:MAXCH], idx[s][wl], len(cand[s])))
            if len(rows) >= 6:
                out.append(rows[:MAXLEN])
        return out

    return dict(cand=cand, ctoi=ctoi, max_c=max_c, nchar=len(chars),
                tr=enc(tr_sents), te=enc(te_sents), vocab_size=len(vocab))


def batches(S, bs, shuffle=True, seed=0):
    idx = list(range(len(S)))
    if shuffle:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(S), generator=g).tolist()
    for i in range(0, len(idx), bs):
        ch = idx[i:i + bs]
        B = len(ch)
        T = max(len(S[j]) for j in ch)
        L = max(len(r[0]) for j in ch for r in S[j])
        xc = torch.zeros(B, T, L, dtype=torch.long)
        cl = torch.ones(B, T, dtype=torch.long)
        for r, j in enumerate(ch):
            for t, (cs, _, _) in enumerate(S[j]):
                xc[r, t, :len(cs)] = torch.tensor(cs)
                cl[r, t] = len(cs)
        y = torch.zeros(B, T, dtype=torch.long)
        ln = torch.zeros(B, T, dtype=torch.long)
        m = torch.zeros(B, T, dtype=torch.bool)
        for r, j in enumerate(ch):
            for t, (_, ci, n) in enumerate(S[j]):
                y[r, t] = ci; ln[r, t] = n; m[r, t] = True
        yield xc, cl, y, ln, m


class CharTiny(nn.Module):
    """char-BiGRU over the stripped syllable -> BiGRU over context -> candidate scores.

    No pooling: order of letters matters (an != na), and max-pooling over a ReLU
    collapsed the whole model to a constant predictor in an earlier version.
    """

    def __init__(self, nchar, max_c, ch_dim=16, hidden=64, syl=32, layers=1, drop=0.1):
        super().__init__()
        self.cemb = nn.Embedding(nchar, ch_dim, padding_idx=0)
        self.cgru = nn.GRU(ch_dim, syl // 2, batch_first=True, bidirectional=True)
        self.cnorm = nn.LayerNorm(syl)
        self.drop = nn.Dropout(drop)
        self.ctx = nn.GRU(syl, hidden // 2, num_layers=layers, batch_first=True,
                          bidirectional=True, dropout=drop if layers > 1 else 0.0)
        self.out = nn.Linear(hidden, max_c)

    def forward(self, xc, cl):
        B, T, L = xc.shape
        e = self.cemb(xc.reshape(B * T, L))
        h, _ = self.cgru(e)
        # take the state at the last REAL character; indexing -1 would read padding,
        # which made the syllable vector depend on the batch instead of the syllable.
        idx = (cl.reshape(-1) - 1).clamp(min=0)
        h = h[torch.arange(h.size(0)), idx]
        h = self.cnorm(h).view(B, T, -1)
        h, _ = self.ctx(self.drop(h))
        return self.out(h)


def evaluate(model, S, bs=128):
    model.eval()
    tot = ok = 0
    ta = oa = tu = ou = 0
    with torch.no_grad():
        for xc, cl, y, ln, m in batches(S, bs, shuffle=False):
            logits = model(xc, cl)
            logits = logits.masked_fill(
                torch.arange(logits.size(-1))[None, None, :] >= ln[..., None], -1e9)
            pred = logits.argmax(-1)
            hit = (pred == y) & m
            tot += m.sum().item(); ok += hit.sum().item()
            amb = ln > 1
            ta += (amb & m).sum().item(); oa += (amb & hit).sum().item()
            un = (~amb) & m
            tu += un.sum().item(); ou += (un & hit).sum().item()
    return ok / tot, oa / max(ta, 1), ou / max(tu, 1), tot


def run(cfg, P, epochs=3, bs=64, lr=3e-3):
    model = CharTiny(P["nchar"], P["max_c"], **cfg)
    nparam = sum(p.numel() for p in model.parameters())
    tr = P["tr"]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(tr) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for xc, cl, y, ln, m in batches(tr, bs, seed=ep):
            logits = model(xc, cl)
            logits = logits.masked_fill(
                torch.arange(logits.size(-1))[None, None, :] >= ln[..., None], -1e9)
            loss = F.cross_entropy(logits[m], y[m])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
    acc, aa, au, n = evaluate(model, P["te"])
    return dict(cfg=cfg, params=nparam, acc=acc, acc_amb=aa, acc_un=au, n=n,
                secs=time.time() - t0)


if __name__ == "__main__":
    ntr = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
    mc = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    ep = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    tr = load_sents("train.txt", ntr); te = load_sents("test.txt", 3000)
    t = time.time(); P = build(tr, te, min_count=mc)
    print(f"train {len(tr)} sents -> {len(P['tr'])} / test {len(P['te'])}  "
          f"max_c={P['max_c']} nchar={P['nchar']} vocab={P['vocab_size']} "
          f"({time.time()-t:.1f}s)", flush=True)
    for cfg in [dict(hidden=32, syl=16), dict(hidden=64, syl=32), dict(hidden=128, syl=64),
                dict(hidden=64, syl=32, layers=2)]:
        r = run(cfg, P, epochs=ep)
        print(f"{str(cfg):44s} params={r['params']:>7,} acc={r['acc']:.4f} "
              f"(nhập nhằng {r['acc_amb']:.4f} | đơn trị {r['acc_un']:.4f}) {r['secs']:.0f}s",
              flush=True)
