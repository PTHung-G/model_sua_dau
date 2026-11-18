"""Find the fix for the gradient-starvation bug.

Diagnosis so far: with out-head init std=1e-2, gradients reaching the attention
block are ~1e-4 while weight_decay=0.01 exerts a ~1e-2 pull, i.e. regularisation
is ~100x stronger than the learning signal. Attention therefore cannot learn and
the model collapses to "always pick slot 0".

This sweeps the candidate remedies, scoring all of them identically against
pick0 (always the most frequent accented form), so any gain is real.
"""
import json
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import load_sents
from fixbuild import build_fair
from scoring import batches, score_predictor, gold_freq_buckets, fmt_row
from attn_model import AttnDiac

torch.set_num_threads(2)
NEG = -1e9


def lg_of(model, xc, cl, f, ln, pri, use_prior=True):
    lg = model(xc, cl)
    if use_prior:
        lg = lg + pri[f]
    ar = torch.arange(lg.size(-1))[None, None, :]
    return lg.masked_fill(ar >= ln[..., None], NEG)


class Pick0:
    def __call__(self, xc, cl, f, ln):
        return torch.zeros_like(f)


def make_head_param_ids(model):
    return {id(p) for n, p in model.named_parameters() if n.startswith("out.")}


def run(name, P, pri, buckets, base, epochs=6, **kw):
    head_std = kw.pop("head_std", 1e-1)
    wd = kw.pop("wd", 0.0)
    head_wd = kw.pop("head_wd", 0.0)
    lr = kw.pop("lr", 2e-3)
    lr_head = kw.pop("lr_head", None)
    use_prior = kw.pop("use_prior", False)
    enc_layers = kw.pop("enc_layers", 1)

    torch.manual_seed(0)
    m = AttnDiac(P["nchar"], P["max_c"], syl=32, dim=64, window=8, blocks=enc_layers)
    nn.init.normal_(m.out.weight, std=head_std)
    nn.init.zeros_(m.out.bias)

    head_ids = make_head_param_ids(m)
    groups = [{"params": [p for n, p in m.named_parameters() if id(p) not in head_ids],
               "lr": lr, "weight_decay": wd},
              {"params": [p for n, p in m.named_parameters() if id(p) in head_ids],
               "lr": lr_head if lr_head is not None else lr,
               "weight_decay": head_wd}]
    opt = torch.optim.AdamW(groups)
    trb = list(batches(P["tr"], 64))
    t0 = time.time()
    for ep in range(epochs):
        m.train()
        order = torch.randperm(len(trb)).tolist()
        for i in order:
            xc, cl, f, y, ln, mk = trb[i]
            lg = lg_of(m, xc, cl, f, ln, pri, use_prior)
            loss = F.cross_entropy(lg[mk], y[mk])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
    fn = lambda xc, cl, f, ln: lg_of(m, xc, cl, f, ln, pri, use_prior).argmax(-1)
    r = score_predictor(fn, P["te"], buckets)
    with torch.no_grad():
        xc, cl, f, y, ln, mk = next(iter(batches(P["te"], 32)))
        pred0 = (fn(xc, cl, f, ln)[mk] == 0).float().mean().item()
    d = r["ALL"][0] - base["ALL"][0]
    print(fmt_row(f"{name:18s}", r) +
          f"  delta={d:+.4f} pred0={pred0:.3f} ({time.time()-t0:.0f}s)", flush=True)
    return dict(name=name, **{k: v[0] for k, v in r.items()}, delta=d, pred0=pred0)


if __name__ == "__main__":
    ntr = int(sys.argv[1]) if len(sys.argv) > 1 else 15000
    tr = load_sents("train.txt", ntr)
    te = load_sents("test.txt", 2000)
    P = build_fair(tr, te); pri = P["pri"]
    buckets = gold_freq_buckets(tr, P["cand"])
    base = score_predictor(Pick0(), P["te"], buckets)
    print(f"ntr={ntr} max_c={P['max_c']}", flush=True)
    print(fmt_row("pick0", base), flush=True)

    configs = [
        ("A current", dict(head_std=1e-2, wd=0.01)),
        ("B big head", dict(head_std=1e-1, wd=0.0)),
        ("C big head + wd", dict(head_std=1e-1, wd=1e-4)),
        ("D asym lr", dict(head_std=1e-1, wd=0.0, lr=1e-2, lr_head=5e-3)),
        ("E asym + prior", dict(head_std=1e-1, wd=0.0, lr=1e-2, lr_head=5e-3,
                                use_prior=True)),
        ("F 2 blocks", dict(head_std=1e-1, wd=0.0, lr=1e-2, lr_head=5e-3,
                            enc_layers=2)),
    ]
    out = []
    for name, kw in configs:
        out.append(run(name, P, pri, buckets, base, **kw))
    json.dump(out, open("sweep_fix.json", "w"), indent=1)
    print("\n=== best ===")
    for r in sorted(out, key=lambda x: -x["ALL"]):
        print(f"{r['name']:18s} ALL={r['ALL']:.4f} delta={r['delta']:+.4f}")
