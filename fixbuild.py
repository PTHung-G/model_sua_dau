"""Fixed data build: vocabulary defined over train+test so both sets share the
same distribution. Only the *label prior* comes from train (no leakage).
"""
import math
from collections import Counter
import torch
from vn import strip_all

NEG = -1e9


def build_fair(tr_sents, te_sents, min_freq=2):
    # candidate inventory from the UNION, so rare-but-real forms are not silently
    # dropped from test while kept in train (that bug made test 98% ambiguous)
    freq_all = Counter()
    for ws in list(tr_sents) + list(te_sents):
        for w in ws:
            freq_all[w.lower()] += 1
    freq_tr = Counter()
    for ws in tr_sents:
        for w in ws:
            freq_tr[w.lower()] += 1

    vocab = {w for w, c in freq_all.items() if c >= min_freq}
    cand = {}
    for w in sorted(vocab):
        cand.setdefault(strip_all(w), []).append(w)
    # order candidates by TRAIN frequency (the prior we are allowed to use)
    for s in cand:
        cand[s].sort(key=lambda w: (-freq_tr.get(w, 0), w))

    forms = sorted(cand)
    fid = {s: i for i, s in enumerate(forms)}
    max_c = max(len(cand[s]) for s in forms)
    pri = torch.full((len(forms), max_c), NEG)
    for s in forms:
        ws = cand[s]
        tot = sum(freq_tr.get(w, 0) for w in ws)
        if tot == 0:
            pri[fid[s], :len(ws)] = math.log(1.0 / len(ws))
        else:
            pri[fid[s], :len(ws)] = torch.tensor(
                [math.log(max(freq_tr.get(w, 0), 0.5) / tot) for w in ws])

    chars = sorted({c for s in forms for c in s})
    ctoi = {c: i + 1 for i, c in enumerate(chars)}

    def enc(sents):
        out = []
        for ws in sents:
            rows = []
            for w in ws:
                wl = w.lower()
                s = strip_all(wl)
                if s not in cand or wl not in vocab:
                    continue
                rows.append((fid[s], [ctoi[c] for c in s][:8],
                             cand[s].index(wl), len(cand[s])))
            if len(rows) >= 6:
                out.append(rows[:48])
        return out

    return dict(cand=cand, pri=pri, max_c=max_c, nchar=len(chars) + 1,
                tr=enc(tr_sents), te=enc(te_sents), vocab=len(vocab))
