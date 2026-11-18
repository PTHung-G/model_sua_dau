"""Decisive test: does the neural model learn ANYTHING beyond the unigram prior?

The whole session measured models with `logits = model + pri`. Because candidates
are sorted by training frequency, argmax(pri)==0 for every form, so "unigram",
"pick index 0" and "bigram" all collapse to the same predictor: always choose the
most frequent accented form. A model that never beats that has demonstrated nothing.

This script isolates the question by comparing, on the SAME tokens:
    pick0       always candidate 0                       (no parameters)
    prior       argmax(pri)                              (== pick0, sanity check)
    prior+nn    model + pri                              (what we ran all session)
    nn-only     model, prior REMOVED                     (does it learn?)
and additionally trains on AMBIGUOUS positions only, so the gradient is never
dominated by the ~87% of tokens where there is only one candidate.
"""
import json
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import load_sents
from fixbuild import build_fair
from scoring import batches, gold_freq_buckets, score_predictor, UnigramPredictor, fmt_row
from attn_model import AttnDiac, rel_bias_report

if not torch.cuda.is_available():
    torch.set_num_threads(2)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NEG = -1e9


def logits_of(model, xc, cl, f, ln, pri, use_prior=True):
    lg = model(xc, cl)
    if use_prior:
        lg = lg + pri.to(lg.device)[f]
    ar = torch.arange(lg.size(-1), device=lg.device)[None, None, :]
    return lg.masked_fill(ar >= ln[..., None], NEG)


def train(model, P, pri, epochs=8, bs=64, lr=2e-3, wd=0.01, use_prior=True,
          only_ambiguous=False):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    trb = list(batches(P["tr"], bs, device=DEV))
    cut = max(1, len(trb) // 20)
    va, trn = trb[:cut], trb[cut:]

    def acc(bs_):
        model.eval(); t = o = 0
        with torch.no_grad():
            for xc, cl, f, y, ln, m in bs_:
                if only_ambiguous:
                    m = m & (ln > 1)
                    if not m.any():
                        continue
                pr = logits_of(model, xc, cl, f, ln, pri, use_prior).argmax(-1)
                t += m.sum().item(); o += ((pr == y) & m).sum().item()
        return o / max(t, 1)

    best = -1; bstate = None
    for e in range(epochs):
        model.train()
        order = torch.randperm(len(trn)).tolist()
        for i in order:
            xc, cl, f, y, ln, m = trn[i]
            if only_ambiguous:
                m = m & (ln > 1)
                if not m.any():
                    continue
            lg = logits_of(model, xc, cl, f, ln, pri, use_prior)
            loss = F.cross_entropy(lg[m], y[m])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        v = acc(va)
        if v > best:
            best = v
            bstate = {k: t.detach().clone() for k, t in model.state_dict().items()}
    if bstate:
        model.load_state_dict(bstate)
    return best


if __name__ == "__main__":
    ntr = int(sys.argv[1]) if len(sys.argv) > 1 else 60000
    ep = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    print(f"device={DEV} ntr={ntr} epochs={ep}", flush=True)
    tr = load_sents("train.txt", ntr)
    te = load_sents("test.txt", 8000)
    P = build_fair(tr, te)
    pri = P["pri"]
    buckets = gold_freq_buckets(tr, P["cand"])
    # ambiguity rate tells us how much of the task is even resolvable
    n = sum(len(r) for r in P["te"])
    amb = sum(1 for r in P["te"] for (_, _, _, k) in r if k > 1)
    print(f"train={len(P['tr'])} test={len(P['te'])} max_c={P['max_c']} "
          f"ambiguous_tokens={amb/n:.3f}", flush=True)

    class Pick0:
        def __call__(self, xc, cl, f, ln):
            return torch.zeros_like(f)

    rows = {}
    rows["pick0"] = score_predictor(Pick0(), P["te"], buckets, device=DEV)
    print(fmt_row("pick0", rows["pick0"]), flush=True)
    rows["prior"] = score_predictor(UnigramPredictor(pri), P["te"], buckets, device=DEV)
    print(fmt_row("prior", rows["prior"]), flush=True)

    for tag, use_prior, only_amb in [("nn-only", False, False),
                                     ("prior+nn", True, False),
                                     ("nn-amb-only", False, True)]:
        torch.manual_seed(0)
        m = AttnDiac(P["nchar"], P["max_c"], syl=32, dim=64, window=8).to(DEV)
        t0 = time.time()
        v = train(m, P, pri, epochs=ep, use_prior=use_prior, only_ambiguous=only_amb)
        fn = lambda xc, cl, f, ln: logits_of(m, xc, cl, f, ln, pri, use_prior).argmax(-1)
        r = score_predictor(fn, P["te"], buckets, device=DEV)
        rows[tag] = r
        npar = sum(p.numel() for p in m.parameters())
        print(fmt_row(tag, r) + f"  params={npar:,} val={v:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if tag == "nn-only":
            rb = rel_bias_report(m)
            json.dump(rb, open("rel_bias.json", "w"), indent=1)
            print("rel bias mean over heads (offset -8..+8):",
                  [round(x, 3) for x in rb["block0"]["mean_over_heads"]], flush=True)
    json.dump(rows, open("decisive.json", "w"), indent=1)
    print("\n=== verdict ===")
    base = rows["pick0"]["ALL"][0]
    for k, r in rows.items():
        print(f"{k:14s} ALL={r['ALL'][0]:.4f}  delta_vs_pick0={r['ALL'][0]-base:+.4f}")
