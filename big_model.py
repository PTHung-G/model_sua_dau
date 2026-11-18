"""1M-parameter model on the expanded corpus (4.16x more training sentences).

Corpus: 2,590,435 train / 142,247 val / 138,497 test sentences from 916,918 articles
(90/5/5 split by article, candidates built from train only).

Model: AttnDiac, syl=192, dim=384, 1 block, window 8 -> 1,040,809 parameters.

The 20k-step budget used on the 622k corpus is only 0.49 epochs here, so the step
count scales with the data instead: `steps` is an argument, default 120k (~3 epochs).

Baseline `small` (32,609 params) is trained at the SAME step count for comparison, so
the size effect is not confounded with the extra epochs the big model gets.
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import evaluate as ev
import pilot_balance as pb
from attn_model import AttnDiac

VARIANTS = {
    "small": dict(syl=32, dim=64, blocks=1),
    "m1m":   dict(syl=192, dim=384, blocks=1),
}


def save_ckpt(model, name, seed, step, nparam, extra=None):
    """Persist weights AND everything needed to reload them for inference.

    A checkpoint is useless without the exact tokenizer/candidate mapping that
    produced it, so the vocab is stored inside the file.
    """
    payload = {
        "model": model.state_dict(),
        "variant": name,
        "seed": seed,
        "step": step,
        "params": nparam,
        "window": 8,
        "out_std": 1e-1,
    }
    if extra:
        payload.update(extra)
    path = f"ckpt_{name}_s{seed}.pt"
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)          # atomic: never leave a half-written file
    print(f"      saved {path} ({os.path.getsize(path)/1e6:.1f} MB)", flush=True)
    return path


def train_one(name, seed, bundle, steps, bs=64, lr=2e-3, eval_every=None):
    tr, cand, char2id, max_c, nchar, forms, toks_te = bundle
    kw = VARIANTS[name]
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = AttnDiac(nchar, max_c, window=8, out_std=1e-1, **kw).to(pb.DEV)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    rng = np.random.default_rng(seed)
    gen = pb.make_batches(tr, cand, char2id, max_c, bs, rng, nchar=nchar, forms=forms)
    t0 = time.time()
    marks = {}
    for step in range(steps):
        xc, cl, f, y, ln, mk, ok = next(gen)
        lg = pb.logits_of(model, xc, cl, ln)
        amb = mk & (ln > 1) & ok
        tgt = amb if amb.any() else ok
        ce = F.cross_entropy(lg.reshape(-1, lg.size(-1)), y.reshape(-1),
                             reduction="none").reshape(y.shape)
        loss = ce[tgt].mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if (step + 1) % max(steps // 6, 1) == 0:
            vr = evaluate(model, toks_te)
            marks[step + 1] = vr
            print(f"      step {step+1}/{steps} loss={loss.item():.4f} "
                  f"ALL={vr['all']:.4f} fix={vr['fix_rate']:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            # Save at every evaluation point, not only at the end: a Kaggle session
            # can die at any time, and previously we lost the weights entirely.
            save_ckpt(model, name, seed, step + 1, nparam)
    return model, marks, nparam


def evaluate(model, toks, bs=128):
    was = model.training
    model.eval()
    with torch.inference_mode():
        pred = lambda xc, cl, f, ln: pb.logits_of(model, xc, cl, ln).argmax(-1)
        r = ev.evaluate(pred, toks, device=pb.DEV, bs=bs)
    model.train(was)
    return r


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 120000
    seeds = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["0"])]
    variants = (sys.argv[3].split(",") if len(sys.argv) > 3 else ["small", "m1m"])
    print(f"device={pb.DEV} steps={steps} seeds={seeds} variants={variants}", flush=True)

    tr = pb.load_sents("data/tr.txt")
    te_sents = pb.load_sents("data/te.txt")
    cand = json.load(open("data/cand.json"))
    forms = sorted(cand)
    forms_id = {s: i for i, s in enumerate(forms)}
    chars = sorted({c for s in forms for c in s})
    char2id = {c: i for i, c in enumerate(chars, start=1)}
    max_c = max(len(v) for v in cand.values())
    nchar = len(chars) + 2
    toks_te = ev.Tokens(te_sents, cand, char2id)
    print(f"train={len(tr):,} test_sents={len(toks_te.sents):,} test_tok={toks_te.n_tok:,} "
          f"max_c={max_c}", flush=True)
    print(f"epochs at {steps} steps = {steps / max(len(tr)//64,1):.2f}", flush=True)

    r0 = ev.evaluate(lambda xc, cl, f, ln: torch.zeros_like(f), toks_te, device=pb.DEV)
    print(ev.fmt("pick most frequent", r0), flush=True)

    bundle = (tr, cand, char2id, max_c, nchar, forms_id, toks_te)
    out = []
    for name in variants:
        for sd in seeds:
            print(f"  --- {name} seed={sd}", flush=True)
            model, marks, nparam = train_one(name, sd, bundle, steps)
            final = evaluate(model, toks_te)
            final.update(variant=name, seed=sd, params=nparam, marks=marks)
            print(f"    FINAL {ev.fmt(name, final)}  params={nparam:,}", flush=True)
            save_ckpt(model, name, sd, steps, nparam,
                      extra={"vocab": forms_id, "char2id": char2id,
                             "max_c": max_c, "nchar": nchar, "metrics": final})
            out.append(final)
            json.dump(out, open(f"big_results_{name}.json", "w"), indent=1)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\n=== summary ===")
    for name in variants:
        rs = [r for r in out if r["variant"] == name]
        if not rs:
            continue
        m = lambda k: sum(r[k] for r in rs) / len(rs)
        print(f"{name:6s} params={rs[0]['params']:>9,} ALL={m('all'):.4f} "
              f"AMB={m('amb'):.4f} fix={m('fix_rate'):.4f} break={m('break_rate'):.4f} "
              f"net={m('net'):+.0f}")
    print(f"pick0 ALL {r0['all']:.4f}")


if __name__ == "__main__":
    main()
