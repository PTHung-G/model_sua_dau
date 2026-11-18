"""Test the 4 attention hypotheses on a SMALL sample, fast.

H1  two-sided context beats one-sided          (direction)
H2  a local window (+-8) is enough              (window)
H3  long range adds nothing beyond +-8          (full vs band)
H4  the learned offset bias decays with distance (inspect the curve)

All runs share the same data, the same test set and the same budget.
"""
import json
import math
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import load_sents
from tiny2 import build, batches
from attn_model import AttnDiac, logits_of, evaluate
from attn import BandAttention, FullAttention

torch.set_num_threads(2)
NEG = -1e9


class OneSided(torch.nn.Module):
    """Wrapper that turns bidirectional attention into a left-only (causal) one."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, xc, cl, mask=None, return_att=False):
        B, T = xc.shape[0], xc.shape[1]
        # causal mask: position i may not see j > i
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=xc.device))
        h = self.inner.proj(self.inner.encode_syllables(xc, cl))
        h = h + self.inner.pos(torch.arange(T, device=xc.device))[None]
        atts = []
        for b in self.inner.blocks:
            attn_mod = b.att
            if isinstance(attn_mod, FullAttention):
                q = attn_mod.q(b.n1(h)).view(B, T, attn_mod.h, -1).transpose(1, 2)
                k = attn_mod.k(b.n1(h)).view(B, T, attn_mod.h, -1).transpose(1, 2)
                v = attn_mod.v(b.n1(h)).view(B, T, attn_mod.h, -1).transpose(1, 2)
                lg = q @ k.transpose(-1, -2) / math.sqrt(attn_mod.dk)
                lg = lg.masked_fill(~causal[None, None], NEG)
                a = F.softmax(lg, dim=-1)
                h = h + attn_mod.o((a @ v).transpose(1, 2).reshape(B, T, -1))
                atts.append(a)
            else:
                w = attn_mod.w
                idx = (torch.arange(T, device=xc.device)[:, None]
                       + torch.arange(-w, w + 1, device=xc.device)[None, :])
                offs = torch.arange(-w, w + 1, device=xc.device)[None, :]
                valid = (idx >= 0) & (idx < T) & (offs <= 0)   # left side only
                idxc = idx.clamp(0, T - 1)
                qq = attn_mod.q(b.n1(h)).view(B, T, attn_mod.h, -1).transpose(1, 2)
                kk = attn_mod.k(b.n1(h)).view(B, T, attn_mod.h, -1).transpose(1, 2)
                vv = attn_mod.v(b.n1(h)).view(B, T, attn_mod.h, -1).transpose(1, 2)
                kn = kk[:, :, idxc, :].permute(0, 1, 2, 4, 3)
                lg = (qq.unsqueeze(-1) * kn).sum(3) / math.sqrt(attn_mod.dk)
                lg = lg + attn_mod.rel[None, None, None, :]
                lg = lg.masked_fill(~valid[None, None], NEG)
                a = F.softmax(lg, dim=-1)
                vn = vv[:, :, idxc, :]
                h = h + attn_mod.o((a.unsqueeze(-1) * vn).sum(3)
                                   .transpose(1, 2).reshape(B, T, -1))
                atts.append(a)
            h = h + b.ff(b.n2(h))
        h = self.inner.norm(h)
        out = self.inner.out(h)
        return (out, atts) if return_att else out


def prep(run_prep_first=True):
    if run_prep_first:
        tr = load_sents("train.txt", int(sys.argv[1]) if len(sys.argv) > 1 else 30000)
        te = load_sents("test.txt", 3000)
        P = build(tr, te, min_count=2)
        P["tr_b"] = list(batches(P["tr"], 64, shuffle=False))
        P["te_b"] = list(batches(P["te"], 128, shuffle=False))
        return P
    return None


def train(model, P, epochs=6, bs=64, lr=2e-3, wd=0.01, pri=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best = 0.0
    hist = []
    # validation = last 5% of train (already batched so we can reuse)
    trb = P["tr_b"]
    cut = max(1, len(trb) // 20)
    va = trb[:cut]
    trn = trb[cut:]
    for e in range(epochs):
        model.train()
        order = torch.randperm(len(trn)).tolist()
        tot = nb = 0
        for i in order:
            xc, cl, f, y, ln, m = trn[i]
            lg = logits_of(model, xc, cl, f, ln, pri)
            loss = F.cross_entropy(lg[m], y[m])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); nb += 1
        vacc = eval_batched(model, va, pri)
        hist.append(round(vacc, 4))
        best = max(best, vacc)
    acc = eval_batched(model, P["te_b"], pri)
    return acc, best, hist


@torch.no_grad()
def eval_batched(model, batched, pri):
    model.eval()
    tot = ok = 0
    for xc, cl, f, y, ln, m in batched:
        pred = logits_of(model, xc, cl, f, ln, pri).argmax(-1)
        tot += m.sum().item(); ok += ((pred == y) & m).sum().item()
    return ok / tot


if __name__ == "__main__":
    t0 = time.time()
    P = prep()
    pri = P["pri"]
    print(f"train={len(P['tr'])} test={len(P['te'])} max_c={P['max_c']} "
          f"({time.time()-t0:.0f}s)", flush=True)
    zero = AttnDiac(P["nchar"], P["max_c"], syl=16, dim=32, window=8)
    print(f"prior-only acc={eval_batched(zero, P['te_b'], pri):.4f}", flush=True)

    res = []
    variants = [
        ("2 phia, full", dict(window=0), False),
        ("2 phia, +-2", dict(window=2), False),
        ("2 phia, +-8", dict(window=8), False),
        ("2 phia, +-16", dict(window=16), False),
        ("1 phia trai, full", dict(window=0), True),
        ("1 phia trai, +-8", dict(window=8), True),
    ]
    for name, kw, onesided in variants:
        torch.manual_seed(0)
        m = AttnDiac(P["nchar"], P["max_c"], syl=16, dim=32, **kw)
        if onesided:
            m = OneSided(m)
        npar = sum(p.numel() for p in m.parameters())
        acc, best, hist = train(m, P, epochs=6, pri=pri)
        res.append(dict(name=name, params=npar, test=acc, val=best, hist=hist))
        print(f"{name:20s} params={npar:7,} test={acc:.4f} val={best:.4f} {hist}", flush=True)
    json.dump(res, open("hyp.json", "w"), indent=1)
    print("\n=== SUMMARY ===")
    for r in res:
        print(f"{r['name']:20s} test={r['test']:.4f}")
