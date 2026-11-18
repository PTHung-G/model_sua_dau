"""Tiny neural models for Vietnamese diacritic restoration (CPU-friendly).

Key trick: instead of a (B,T,V) mask, we gather only the logits of the legal
candidate accented forms per position -> (B,T,max_c). Same maths, ~30x cheaper.
"""
import math
import time
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import load_sents, build_vocab
from vn import strip_all

torch.set_num_threads(6)


class TinyDiac(nn.Module):
    def __init__(self, n_in, n_out, hidden=128, layers=1, arch="lstm", drop=0.1):
        super().__init__()
        self.arch = arch
        self.emb = nn.Embedding(n_in, hidden, padding_idx=0)
        self.drop = nn.Dropout(drop)
        if arch == "lstm":
            self.enc = nn.LSTM(hidden, hidden // 2, num_layers=layers, batch_first=True,
                               bidirectional=True, dropout=drop if layers > 1 else 0.0)
            out_dim = hidden
        elif arch == "conv":
            self.enc = nn.ModuleList([nn.Conv1d(hidden, hidden, 3, padding=1) for _ in range(layers)])
            out_dim = hidden
        elif arch == "trans":
            layer = nn.TransformerEncoderLayer(hidden, 4, hidden * 2, drop,
                                               batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, layers)
            self.pos = nn.Embedding(64, hidden)
            out_dim = hidden
        self.out = nn.Linear(out_dim, n_out)

    def forward(self, x):
        h = self.drop(self.emb(x))
        if self.arch == "lstm":
            h, _ = self.enc(h)
        elif self.arch == "conv":
            for c in self.enc:
                h = torch.relu(c(h.transpose(1, 2))).transpose(1, 2)
        else:
            p = self.pos(torch.arange(x.size(1), device=x.device))
            h = self.enc(h + p, src_key_padding_mask=x.eq(0))
        return self.out(h)


def prepare(tr_sents, te_sents):
    itos, stoi, cand = build_vocab(tr_sents)
    in_stoi = {"<pad>": 0, "<unk>": 1}
    for s in cand:
        in_stoi.setdefault(s, len(in_stoi))
    max_c = max(len(v) for v in cand.values())
    pos_in_cand = {k: {tid: j for j, tid in enumerate(v)} for k, v in cand.items()}

    ctab = torch.zeros(len(in_stoi), max_c, dtype=torch.long)
    clen = torch.zeros(len(in_stoi), dtype=torch.long)
    for s, vs in cand.items():
        k = in_stoi[s]
        ctab[k, :len(vs)] = torch.tensor(vs)
        clen[k] = len(vs)

    def enc(sents):
        X, Y, C = [], [], []
        for ws in sents:
            xs, ys, cs = [], [], []
            for w in ws:
                wl = w.lower()
                s = strip_all(wl)
                if s not in cand or wl not in stoi:
                    continue
                xs.append(in_stoi[s])
                ys.append(stoi[wl])
                cs.append(pos_in_cand[s][stoi[wl]])
            if len(xs) >= 6:
                X.append(xs); Y.append(ys); C.append(cs)
        return X, Y, C

    return dict(itos=itos, stoi=stoi, in_stoi=in_stoi, cand=cand, ctab=ctab,
                clen=clen, V=len(itos), max_c=max_c,
                tr=enc(tr_sents), te=enc(te_sents))


def batches(X, Y, C, bs, maxlen=48, shuffle=True, seed=0):
    idx = list(range(len(X)))
    if shuffle:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(X), generator=g).tolist()
    for i in range(0, len(idx), bs):
        ch = idx[i:i + bs]
        T = max(min(len(X[j]), maxlen) for j in ch)
        xi = torch.zeros(len(ch), T, dtype=torch.long)
        ci = torch.zeros(len(ch), T, dtype=torch.long)
        m = torch.zeros(len(ch), T, dtype=torch.bool)
        for r, j in enumerate(ch):
            n = min(len(X[j]), maxlen)
            xi[r, :n] = torch.tensor(X[j][:n])
            ci[r, :n] = torch.tensor(C[j][:n])
            m[r, :n] = True
        yield xi, ci, m


def forward_c(logits, xi, ctab, clen):
    """Gather candidate logits: (B,T,max_c) and the valid-count (B,T)."""
    ids = ctab[xi]                       # (B,T,max_c)
    ln = clen[xi]                        # (B,T)
    cl = logits.gather(2, ids)           # (B,T,max_c)
    cl = cl.masked_fill(torch.arange(cl.size(-1))[None, None, :] >= ln[..., None], -1e9)
    return cl, ln


def evaluate(model, P):
    model.eval()
    X, Y, C = P["te"]
    tot = ok = 0
    ta = oa = tu = ou = 0
    with torch.no_grad():
        for xi, ci, m in batches(X, Y, C, 128, shuffle=False):
            cl, ln = forward_c(model(xi), xi, P["ctab"], P["clen"])
            pred = cl.argmax(-1)
            hit = (pred == ci) & m
            tot += m.sum().item(); ok += hit.sum().item()
            amb = ln > 1
            ta += (amb & m).sum().item(); oa += (amb & hit).sum().item()
            un = (~amb) & m
            tu += un.sum().item(); ou += (un & hit).sum().item()
    return ok / tot, oa / max(ta, 1), ou / max(tu, 1), tot


def run(cfg, P, epochs=3, bs=32, lr=3e-3, tag=""):
    model = TinyDiac(len(P["in_stoi"]), P["V"], **cfg)
    nparam = sum(p.numel() for p in model.parameters())
    X, Y, C = P["tr"]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(X) / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for xi, ci, m in batches(X, Y, C, bs, seed=ep):
            cl, _ = forward_c(model(xi), xi, P["ctab"], P["clen"])
            loss = F.cross_entropy(cl[m], ci[m])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
    acc, acc_a, acc_u, n = evaluate(model, P)
    return dict(tag=tag, cfg=cfg, params=nparam, acc=acc, acc_amb=acc_a,
                acc_un=acc_u, n=n, secs=time.time() - t0)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50000
    tr = load_sents("train.txt", n)
    te = load_sents("test.txt", 3000)
    print(f"train {len(tr)} sents / test {len(te)} sents", flush=True)
    P = prepare(tr, te)
    print(f"V={P['V']} in_vocab={len(P['in_stoi'])} max_c={P['max_c']} "
          f"train_sents={len(P['tr'][0])} test_sents={len(P['te'][0])}", flush=True)
    for cfg in [
        dict(hidden=64, layers=1, arch="lstm"),
        dict(hidden=128, layers=1, arch="lstm"),
        dict(hidden=128, layers=2, arch="lstm"),
        dict(hidden=128, layers=2, arch="conv"),
    ]:
        r = run(cfg, P, epochs=2)
        print(f"{str(cfg):46s} params={r['params']:>8,} acc={r['acc']:.4f} "
              f"(nhập nhằng {r['acc_amb']:.4f} | đơn trị {r['acc_un']:.4f}) {r['secs']:.0f}s",
              flush=True)
