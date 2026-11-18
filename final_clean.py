"""Final run: config A on the full clean corpus, evaluated once on the locked test set.

Uses the clean article-level split and the verified scorer. Three seeds so the
reported number comes with a spread rather than a single lucky draw.

Test is touched exactly once, after the configuration is fixed. Config A is the
control that beat weighted loss, sampling, and a 6.75x larger model in the pilot.
"""
import json
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import evaluate as ev
import pilot_balance as pb
from attn_model import AttnDiac

STEPS = 20000
BS = 64


def train_one(seed, bundle, steps, lr=2e-3):
    tr, cand, freq, char2id, max_c, nchar, forms = bundle
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = AttnDiac(nchar, max_c, syl=32, dim=64, window=8, blocks=1,
                     out_std=1e-1).to(pb.DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    rng = np.random.default_rng(seed)
    gen = pb.make_batches(tr, cand, char2id, max_c, BS, rng, nchar=nchar,
                          forms=forms)
    for step in range(steps):
        xc, cl, f, y, ln, mk, ok = next(gen)
        lg = pb.logits_of(model, xc, cl, ln)
        amb = mk & (ln > 1) & ok
        tgt = amb if amb.any() else ok
        loss = F.cross_entropy(lg.reshape(-1, lg.size(-1)), y.reshape(-1),
                               reduction="none").reshape(y.shape)[tgt].mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if (step + 1) % (steps // 5) == 0:
            print(f"      step {step+1}/{steps} loss={loss.item():.4f}", flush=True)
    model.eval()
    return model


def main():
    ntr = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else STEPS
    seeds = [int(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3
                              else ["0", "1", "2"])]
    print(f"device={pb.DEV} limit={ntr or 'all'} steps={steps} seeds={seeds}",
          flush=True)

    tr_all = pb.load_sents("data/tr.txt")
    tr = tr_all[:ntr] if ntr else tr_all
    te_sents = pb.load_sents("data/te.txt")
    cand = json.load(open("data/cand.json"))
    freq = {}
    for ws in tr_all:
        for w in ws:
            freq[w.lower()] = freq.get(w.lower(), 0) + 1
    forms = sorted(cand)
    forms_id = {s: i for i, s in enumerate(forms)}
    chars = sorted({c for s in forms for c in s})
    char2id = {c: i for i, c in enumerate(chars, start=1)}
    max_c = max(len(v) for v in cand.values())
    nchar = len(chars) + 2

    toks_te = ev.Tokens(te_sents, cand, char2id)
    print(f"train={len(tr):,} test_sents={len(toks_te.sents):,} "
          f"test_tok={toks_te.n_tok:,}", flush=True)

    # Baselines computed with the same scorer so the comparison is apples to apples.
    r0 = ev.evaluate(lambda xc, cl, f, ln: torch.zeros_like(f), toks_te, device=pb.DEV)
    print(ev.fmt("pick most frequent", r0), flush=True)

    bundle = (tr, cand, freq, char2id, max_c, nchar, forms_id)
    results = []
    for sd in seeds:
        print(f"  --- seed {sd}", flush=True)
        model = train_one(sd, bundle, steps)
        with torch.inference_mode():
            pred = lambda xc, cl, f, ln: pb.logits_of(model, xc, cl, ln).argmax(-1)
            r = ev.evaluate(pred, toks_te, device=pb.DEV)
        r["seed"] = sd
        r["params"] = sum(p.numel() for p in model.parameters())
        print("    " + ev.fmt(f"seed {sd}", r), flush=True)
        results.append(r)
        json.dump(results, open("final_results.json", "w"), indent=1)

    print("\n=== final (mean +- sd over seeds) ===")
    m = lambda k: sum(r[k] for r in results) / len(results)
    s = lambda k: float(np.std([r[k] for r in results]))
    for k in ("all", "amb", "base_ok", "base_bad", "fix_rate", "break_rate",
              "net", "coverage"):
        print(f"  {k:12s} {m(k):.4f} +- {s(k):.4f}")
    ratio = m("fix_rate") / m("break_rate") if m("break_rate") else float("inf")
    print(f"  fix/break ratio  {ratio:.1f} : 1")
    print(f"  params           {results[0]['params']:,}")
    print(f"  pick0 ALL        {r0['all']:.4f}")
    print(f"  gain over pick0  {m('all') - r0['all']:+.4f}")


if __name__ == "__main__":
    main()
