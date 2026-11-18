"""Ablation #1 of two: FACTORIZED tone + modifier output.

Question: does splitting the output into 'tone (6)' and 'vowel modifier (7)' beat a
flat softmax over whole candidates?

Diagnosis that motivates it (score_all.py on the v2 model, 2.64M wiki syllables):
    wrong tone only      159,324
    wrong modifier only   24,510
    wrong both            68,698
So tone is the dominant error, and 63% of errors also get the modifier wrong. A flat
softmax over 34 near-identical slots has to get both right in one draw.

Design: two heads from the same hidden state, forced ORTHOGONAL by the candidate
inventory itself. If every (tone, modifier) pair among a syllable's candidates spells
a DIFFERENT syllable, the joint table is exactly a product space and argmax over it
equals (argmax tone, argmax modifier). The trainer asserts this property and the
checker verifies it, so this experiment cannot silently test the wrong thing.

Controls: same config, same seed, same data, same steps, same batch order - the only
difference is the output layer. Params are reported so any win cannot be mistaken for
capacity.

Usage:
    python exp_factorized.py --variant m1m --steps 20000 --train-sents 750000
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pilot_balance as pb
from attn_model import AttnDiac
from vn import tone_of, decompose

TONES = ["ngang", "sac", "huyen", "hoi", "nga", "nang"]
MODS = ["", "circumflex", "breve", "horn", "circumflex+horn", "circumflex+breve",
        "other"]


def tone_idx(w):
    return TONES.index(tone_of(w) or "ngang")


def mod_idx(w):
    _, _, mods = decompose(w)
    m = "+".join(sorted(mods))
    return MODS.index(m) if m in MODS else MODS.index("other")


def build_targets(cand):
    """Per (stripped form, slot) -> (tone_id, mod_id); also verify orthogonality."""
    t_of, m_of = {}, {}
    nonorth = 0
    for s, cs in cand.items():
        seen = {}
        for i, w in enumerate(cs):
            t, m = tone_idx(w), mod_idx(w)
            t_of[(s, i)] = t
            m_of[(s, i)] = m
            key = (t, m)
            if key in seen and seen[key] != i:
                nonorth += 1
            seen[key] = i
    return t_of, m_of, nonorth


def lookup(tab, f, y):
    return torch.tensor([tab[(int(a), int(b))] for a, b in zip(f.reshape(-1), y.reshape(-1))],
                        dtype=torch.long)


def train_one(factorized, seed, bundle, steps, bs=64, lr=2e-3):
    tr, cand, char2id, max_c, nchar, forms, t_of, m_of = bundle
    torch.manual_seed(seed)
    np.random.seed(seed)
    kw = dict(syl=192, dim=384, blocks=1)
    model = AttnDiac(nchar, max_c, window=8, out_std=1e-1,
                     factorized=factorized, **kw).to(pb.DEV)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    rng = np.random.default_rng(seed)
    gen = pb.make_batches(tr, cand, char2id, max_c, bs, rng, nchar=nchar, forms=forms)
    t0 = time.time()
    for step in range(steps):
        xc, cl, f, y, ln, mk, ok = next(gen)
        amb = mk & (ln > 1) & ok
        tgt = amb if amb.any() else ok
        out = model(xc, cl)
        if factorized:
            lt, lm = out
            t_id = lookup(t_of, f, y).to(pb.DEV).reshape(y.shape)
            m_id = lookup(m_of, f, y).to(pb.DEV).reshape(y.shape)
            ce_t = F.cross_entropy(lt.reshape(-1, lt.size(-1)), t_id.reshape(-1),
                                   reduction="none").reshape(y.shape)
            ce_m = F.cross_entropy(lm.reshape(-1, lm.size(-1)), m_id.reshape(-1),
                                   reduction="none").reshape(y.shape)
            loss = (ce_t[tgt].mean() + ce_m[tgt].mean()) / 2
        else:
            lg = out
            ar = torch.arange(lg.size(-1), device=lg.device)[None, None, :]
            lg = lg.masked_fill(ar >= ln[..., None], -1e9)
            ce = F.cross_entropy(lg.reshape(-1, lg.size(-1)), y.reshape(-1),
                                 reduction="none").reshape(y.shape)
            loss = ce[tgt].mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if (step + 1) % max(steps // 4, 1) == 0:
            print(f"      step {step+1}/{steps} loss={loss.item():.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return model, nparam


@torch.inference_mode()
def predict_slots(model, sents, cand, char2id, forms_bys, max_c, nchar, factorized,
                  t_of, m_of):
    """Return per-sentence predicted slot indices (aligned to covered syllables)."""
    out = []
    inv = {}
    for (s, i), t in t_of.items():
        inv.setdefault((s, t, m_of[(s, i)]), i)
    for s in sents:
        toks = s.split()
        rows = []
        for w in toks:
            st = pb.strip_all(w.lower())
            cs = cand.get(st)
            if cs:
                rows.append((st, len(cs)))
        if not rows:
            out.append((toks, []))
            continue
        T = len(rows)
        L = max(len(r[0]) for r in rows)
        xc = torch.full((1, T, L), nchar - 1, dtype=torch.long)
        cl = torch.ones((1, T), dtype=torch.long)
        ln = torch.ones((1, T), dtype=torch.long)
        for t, (st, k) in enumerate(rows):
            ids = [char2id.get(c, 0) for c in st[:64]]
            xc[0, t, :len(ids)] = torch.tensor(ids)
            cl[0, t] = max(len(ids), 1)
            ln[0, t] = k
        o = model(xc.to(pb.DEV), cl.to(pb.DEV))
        if factorized:
            lt, lm = o
            pt = lt.argmax(-1)[0].tolist()
            pm = lm.argmax(-1)[0].tolist()
            picks = []
            for t, (st, k) in enumerate(rows):
                j = inv.get((st, pt[t], pm[t]))
                if j is None or j >= k:
                    j = 0
                picks.append(j)
        else:
            ar = torch.arange(o.size(-1), device=o.device)[None, None, :]
            lg = o.masked_fill(ar >= ln[..., None], -1e9)
            picks = lg.argmax(-1)[0].tolist()
        # rebuild words
        words, j = [], 0
        for w in toks:
            st = pb.strip_all(w.lower())
            cs = cand.get(st)
            if not cs:
                words.append(w)
                continue
            c = cs[picks[j]] if picks[j] < len(cs) else cs[0]
            j += 1
            words.append(c.capitalize() if w[:1].isupper() else c)
        out.append((words, picks))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--train-sents", type=int, default=750_000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="exp_factorized.json")
    a = ap.parse_args()

    print(f"device={pb.DEV} steps={a.steps} train_sents={a.train_sents:,}", flush=True)
    te_sents = [l.rstrip("\n") for l in open("data/te.txt", encoding="utf-8")][:20000]
    tr = pb.load_sents("data/tr.txt")[:a.train_sents]
    cand = json.load(open("data/cand.json"))
    forms = sorted(cand)
    forms_id = {s: i for i, s in enumerate(forms)}
    char2id = {c: i for i, c in enumerate(sorted({c for s in forms for c in s}), 1)}
    max_c = max(len(v) for v in cand.values())
    nchar = len(char2id) + 2
    print(f"train={len(tr):,} test={len(te_sents):,} max_c={max_c}", flush=True)

    t_of, m_of, nonorth = build_targets(cand)
    print(f"orthogonality check: {nonorth} (tone,modifier) collisions across the "
          f"whole inventory - 0 means the product-space assumption is exact",
          flush=True)

    bundle = (tr, cand, char2id, max_c, nchar, forms_id, t_of, m_of)
    res = []
    for factorized in (False, True):
        tag = "factorized" if factorized else "flat (control)"
        print(f"\n=== {tag} ===", flush=True)
        model, nparam = train_one(factorized, a.seed, bundle, a.steps, a.bs)
        # evaluate on wiki sample via the same word-level scorer used elsewhere
        import score_all as sa
        sa.CAND.update(cand)
        sa.FORMS_BY_STRIP.update({s: i for i, s in enumerate(forms)})

        @torch.inference_mode()
        def pred(sents, _m=model, _f=factorized):
            outs = predict_slots(_m, sents, cand, char2id, forms_id, max_c, nchar,
                                 _f, t_of, m_of)
            return [o[0] for o in outs]
        r = sa.score(te_sents, pred(te_sents))
        r["tag"] = tag
        r["params"] = nparam
        res.append(r)
        print(sa.fmt(tag, r), flush=True)
        torch.save({"model": model.state_dict(), "factorized": factorized,
                    "params": nparam, "steps": a.steps,
                    "char2id": char2id, "forms": forms_id, "max_c": max_c,
                    "nchar": nchar},
                   f"exp_ckpt_{'fac' if factorized else 'flat'}.pt")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    json.dump(res, open(a.out, "w"), indent=1)
    d = res[1]["syll_acc"] - res[0]["syll_acc"]
    print(f"\nfactorized - flat = {d*100:+.2f} pt  "
          f"(params {res[1]['params']:,} vs {res[0]['params']:,})")
    print(f"wrote {a.out}")
