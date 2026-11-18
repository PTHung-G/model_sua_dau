"""Can performance go higher? Close the gap to the oracle order-1 baseline.

Diagnosis of the remaining error: order-1 lookup conditions on the GOLD accented
previous syllable, while the model only sees the *stripped* syllables and must
infer the neighbour's accent itself. That is the entire 1.06-point gap on the
headroom subset.

Configs (all trained identically on the 589k corpus, scored on the full 20k test):
  A  1 block, window 8, no feedback        -- reproduces the previous best
  B  2 blocks, window 8                    -- lets the model resolve a neighbour
                                              accent before using it (2 steps)
  C  2 blocks, window 16                   -- more context
  D  2 blocks, window 16, accent feedback  -- two-pass decoding: pass 2 is given
                                              pass-1 predictions, i.e. exactly the
                                              information order-1 exploits
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


def lg_of(model, xc, cl, ln, acc_ids=None):
    lg = model(xc, cl, acc_ids=acc_ids)
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
    learn, head = [], []
    for row in te_rows:
        lr_, hr = [], []
        for i, (fi, cs, ci, k) in enumerate(row):
            gold = cand[names[fi]][ci]
            ok = False
            if i > 0:
                pfi, pci = row[i - 1][0], row[i - 1][2]
                ok = pair.get((cand[names[pfi]][pci], gold), 0) >= 2
            lr_.append(ok); hr.append(ok and ci != 0)
        learn.append(lr_); head.append(hr)
    return learn, head


def pad_masks(rows, n, T):
    t = torch.zeros(n, T, dtype=torch.bool)
    for i, r in enumerate(rows[:n]):
        for j, v in enumerate(r[:T]):
            t[i, j] = v
    return t


def score(pred_fn, P, lm, hm, tag):
    st = {k: [0, 0] for k in ("all", "learn", "head")}
    bi = 0
    for xc, cl, f, y, ln, m in batches(P["te"], 256, device=DEV):
        B, T = f.shape
        pred = pred_fn(xc, cl, f, ln)
        hit = (pred == y) & m
        L = lm[bi:bi + B, :T].to(m.device)
        H = hm[bi:bi + B, :T].to(m.device)
        st["all"][0] += m.sum().item(); st["all"][1] += hit.sum().item()
        st["learn"][0] += (m & L).sum().item(); st["learn"][1] += (hit & L).sum().item()
        st["head"][0] += (m & H).sum().item(); st["head"][1] += (hit & H).sum().item()
        bi += B
    r = {k: (v[1] / v[0] if v[0] else 0.0) for k, v in st.items()}
    print(f"{tag:26s} ALL={r['all']:.4f} LEARNABLE={r['learn']:.4f} "
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
        self.cand = cand; self.names = sorted(cand); self.pri = pri

    def __call__(self, xc, cl, f, ln):
        out = self.pri.to(xc.device)[f].argmax(-1)
        fl = f.tolist(); cache = {}
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


def train_model(P, epochs, blocks, window, feedback, t0):
    torch.manual_seed(0)
    m = AttnDiac(P["nchar"], P["max_c"], syl=32, dim=64, window=window,
                 blocks=blocks, acc_feedback=feedback, out_std=1e-1).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    trb = list(batches(P["tr"], 64, device=DEV))
    for e in range(epochs):
        m.train()
        order = torch.randperm(len(trb)).tolist()
        tl = tn = 0
        for i in order:
            xc, cl, f, y, ln, mk = trb[i]
            acc = None
            if feedback:
                # half the batches see gold accents (teacher forcing), half see
                # nothing, so the model works in both one-pass and two-pass mode
                if torch.rand(1).item() < 0.5:
                    acc = y.clone()
                    acc[~mk] = 0
            lg = lg_of(m, xc, cl, ln, acc)
            loss = F.cross_entropy(lg[mk], y[mk])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            tl += loss.item() * mk.sum().item(); tn += mk.sum().item()
        sched.step()
        print(f"    ep{e+1}/{epochs} loss={tl/tn:.4f} ({time.time()-t0:.0f}s)", flush=True)
    return m


def make_pred(m, feedback):
    if not feedback:
        return lambda xc, cl, f, ln: lg_of(m, xc, cl, ln).argmax(-1)

    def two_pass(xc, cl, f, ln):
        p1 = lg_of(m, xc, cl, ln).argmax(-1)   # pass 1: no accent info
        # p1 is already a valid candidate slot (< ln), so it indexes acc_emb safely.
        # Padded positions get slot 0 == "no information", which is neutral.
        acc = p1.clamp(0, m.acc_emb.num_embeddings - 1)
        return lg_of(m, xc, cl, ln, acc).argmax(-1)    # pass 2: use pass-1 accents
    return two_pass


if __name__ == "__main__":
    corpus = sys.argv[1] if len(sys.argv) > 1 else "train600k.txt"
    ntr = int(sys.argv[2]) if len(sys.argv) > 2 else 600000
    ep = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    nte = int(sys.argv[4]) if len(sys.argv) > 4 else 20000
    print(f"device={DEV} corpus={corpus} ntr={ntr} epochs={ep} nte={nte}", flush=True)
    t0 = time.time()
    tr = load_sents(corpus, ntr)
    te = load_sents("test.txt", nte)
    P = build_fair(tr, te)
    names = sorted(P["cand"]); pri = P["pri"]
    learn, head = build_masks(tr, P["te"], names, P["cand"])
    nl = sum(sum(x) for x in learn); nh = sum(sum(x) for x in head)
    nt = sum(len(r) for r in P["te"])
    print(f"train={len(P['tr']):,} test={len(P['te']):,} learnable={nl/nt:.3f} "
          f"headroom={nh/nt:.3f} ({time.time()-t0:.0f}s)", flush=True)
    T = max(len(r) for r in P["te"])
    lm = pad_masks(learn, len(P["te"]), T)
    hm = pad_masks(head, len(P["te"]), T)

    res = [score(Pick0(), P, lm, hm, "pick0"),
           score(Order1(tr, P["cand"], pri), P, lm, hm, "order1")]

    configs = [("A 1blk w8", 1, 8, False),
               ("B 2blk w8", 2, 8, False),
               ("C 2blk w16", 2, 16, False),
               ("D 2blk w16 +feedback", 2, 16, True)]
    for tag, blocks, window, fb in configs:
        print(f"  --- {tag}", flush=True)
        m = train_model(P, ep, blocks, window, fb, t0)
        npar = sum(p.numel() for p in m.parameters())
        r = score(make_pred(m, fb), P, lm, hm, tag)
        r["params"] = npar
        print(f"      params={npar:,}", flush=True)
        res.append(r)
        json.dump(res, open("improve.json", "w"), indent=1)

    print("\n=== verdict (full test) ===")
    for r in res:
        print(f"{r['tag']:26s} ALL={r['all']:.4f} HEADROOM={r['head']:.4f}")
