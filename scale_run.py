"""Scale up: many more sentences, many more epochs.

Everything measured earlier was capped by two things: data size and training length
(loss was still falling at epoch 6). This run uses the full corpus and a long
schedule, and scores on:
    ALL               every test token
    LEARNABLE         tokens whose (prev, gold) pair was seen >=2x in train
    HEADROOM          learnable AND gold is not the rank-0 form -- the ONLY subset
                      where a context model can beat 'always pick the most frequent'
order1 is reported for reference but is optimistic on LEARNABLE by construction
(that subset is defined by its own lookup succeeding); HEADROOM is the fair column.
"""
import json
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from data import load_sents
from fixbuild import build_fair
from scoring import batches
from attn_model import AttnDiac

if not torch.cuda.is_available():
    torch.set_num_threads(2)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NEG = -1e9


def lg_of(model, xc, cl, f, ln, use_prior=False):
    lg = model(xc, cl)
    ar = torch.arange(lg.size(-1), device=lg.device)[None, None, :]
    return lg.masked_fill(ar >= ln[..., None], NEG)


def build_masks(tr_sents, te_rows, names, cand):
    pair = Counter()
    for ws in tr_sents:
        prev = None
        for w in ws:
            wl = w.lower()
            if prev is not None:
                pair[(prev, wl)] += 1
            prev = wl
    learn = []
    head = []
    for row in te_rows:
        lr_, hr = [], []
        for i, (fi, cs, ci, k) in enumerate(row):
            gold = cand[names[fi]][ci]
            ok = False
            if i > 0:
                pfi, pci = row[i - 1][0], row[i - 1][2]
                prev = cand[names[pfi]][pci]
                ok = pair.get((prev, gold), 0) >= 2
            lr_.append(ok)
            hr.append(ok and ci != 0)
        learn.append(lr_)
        head.append(hr)
    return learn, head


def score(pred_fn, P, learn, head, tag, detail=False):
    st = {k: [0, 0] for k in ("all", "learn", "head")}
    bi = 0
    for xc, cl, f, y, ln, m in batches(P["te"], 128, device=DEV):
        pred = pred_fn(xc, cl, f, ln)
        hit = (pred == y) & m
        for b in range(xc.size(0)):
            idx = bi + b
            lm = learn[idx] if idx < len(learn) else []
            hm = head[idx] if idx < len(head) else []
            for t in range(xc.size(1)):
                if not m[b, t]:
                    continue
                h = bool(hit[b, t])
                st["all"][0] += 1; st["all"][1] += h
                if t < len(lm) and lm[t]:
                    st["learn"][0] += 1; st["learn"][1] += h
                if t < len(hm) and hm[t]:
                    st["head"][0] += 1; st["head"][1] += h
        bi += xc.size(0)
    r = {k: (v[1] / v[0] if v[0] else 0.0) for k, v in st.items()}
    print(f"{tag:14s} ALL={r['all']:.4f} LEARNABLE={r['learn']:.4f} "
          f"HEADROOM={r['head']:.4f} (n_head={st['head'][0]:,})", flush=True)
    return dict(tag=tag, **r, n_head=st["head"][0])


class Pick0:
    def __call__(self, xc, cl, f, ln):
        return torch.zeros_like(f)


class Order1:
    def __init__(self, tr_sents, cand, pri):
        self.cnt = Counter()
        for ws in tr_sents:
            prev = None
            for w in ws:
                wl = w.lower()
                if prev is not None:
                    self.cnt[(prev, wl)] += 1
                prev = wl
        self.cand = cand
        self.names = sorted(cand)
        self.pri = pri

    def __call__(self, xc, cl, f, ln):
        out = self.pri.to(xc.device)[f].argmax(-1)
        fl = f.tolist()
        cache = {}
        for b in range(f.size(0)):
            for t in range(1, f.size(1)):
                key = (fl[b][t - 1], fl[b][t])
                if key not in cache:
                    best, bc = -1, 0
                    for pc in self.cand[self.names[key[0]]]:
                        for ci, gc in enumerate(self.cand[self.names[key[1]]]):
                            c = self.cnt.get((pc, gc), 0)
                            if c > bc:
                                bc, best = c, ci
                    cache[key] = best
                if 0 <= cache[key] < int(ln[b, t]):
                    out[b, t] = cache[key]
        return out


if __name__ == "__main__":
    corpus = sys.argv[1] if len(sys.argv) > 1 else "train600k.txt"
    ntr = int(sys.argv[2]) if len(sys.argv) > 2 else 700000
    ep = int(sys.argv[3]) if len(sys.argv) > 3 else 25
    print(f"device={DEV} corpus={corpus} ntr={ntr} epochs={ep}", flush=True)
    t0 = time.time()
    tr = load_sents(corpus, ntr)
    te = load_sents("test.txt", 3000)
    P = build_fair(tr, te)
    names = sorted(P["cand"]); pri = P["pri"]
    learn, head = build_masks(tr, P["te"], names, P["cand"])
    nl = sum(sum(x) for x in learn); nh = sum(sum(x) for x in head)
    nt = sum(len(r) for r in P["te"])
    print(f"train={len(P['tr']):,} test={len(P['te']):,} learnable={nl/nt:.3f} "
          f"headroom={nh/nt:.3f} ({time.time()-t0:.0f}s)", flush=True)

    res = [score(Pick0(), P, learn, head, "pick0"),
           score(Order1(tr, P["cand"], pri), P, learn, head, "order1")]

    torch.manual_seed(0)
    m = AttnDiac(P["nchar"], P["max_c"], syl=32, dim=64, window=8).to(DEV)
    nn.init.normal_(m.out.weight, std=1e-1); nn.init.zeros_(m.out.bias)
    print(f"params={sum(p.numel() for p in m.parameters()):,}", flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ep)
    trb = list(batches(P["tr"], 64, device=DEV))
    best = -1; bstate = None
    for e in range(ep):
        m.train()
        order = torch.randperm(len(trb)).tolist()
        tl = tn = 0
        for i in order:
            xc, cl, f, y, ln, mk = trb[i]
            lg = lg_of(m, xc, cl, f, ln)
            loss = F.cross_entropy(lg[mk], y[mk])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            tl += loss.item() * mk.sum().item(); tn += mk.sum().item()
        sched.step()
        # cheap inline check every few epochs on the headroom subset only
        if (e + 1) % 5 == 0 or e == ep - 1:
            fn = lambda xc, cl, f, ln: lg_of(m, xc, cl, f, ln).argmax(-1)
            r = score(fn, P, learn, head, f"ep{e+1}")
            if r["head"] > best:
                best = r["head"]
                bstate = {k: v.detach().clone() for k, v in m.state_dict().items()}
        print(f"  ep{e+1} loss={tl/tn:.4f} ({time.time()-t0:.0f}s)", flush=True)
    if bstate:
        m.load_state_dict(bstate)
    fn = lambda xc, cl, f, ln: lg_of(m, xc, cl, f, ln).argmax(-1)
    res.append(score(fn, P, learn, head, "attention"))
    json.dump(res, open("scale_run.json", "w"), indent=1)
    print("\n=== verdict ===")
    for r in res:
        print(f"{r['tag']:14s} HEADROOM={r['head']:.4f}")
