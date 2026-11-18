"""Loading + encoding for the Vietnamese diacritic restoration task."""
import json
from collections import Counter
from vn import strip_all

PAD, UNK = "<pad>", "<unk>"


def load_sents(path, limit=None):
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            out.append(line.split())
    return out


def build_vocab(sents, min_count=1):
    """Accented syllable vocabulary + map stripped -> candidate ids."""
    cnt = Counter()
    for ws in sents:
        for w in ws:
            cnt[w.lower()] += 1
    itos = [PAD, UNK] + [w for w, c in cnt.most_common() if c >= min_count]
    stoi = {w: i for i, w in enumerate(itos)}
    cand = {}
    for w in itos[2:]:
        cand.setdefault(strip_all(w), []).append(stoi[w])
    for s in cand:
        cand[s].sort()
    return itos, stoi, cand


def encode(sents, stoi, cand, max_cands=None):
    """Return (inputs[list of stripped strings], targets[list of id]) per token."""
    X, Y = [], []
    for ws in sents:
        xs, ys = [], []
        for w in ws:
            wl = w.lower()
            s = strip_all(wl)
            if s not in cand or wl not in stoi:
                continue
            c = cand[s]
            if max_cands and len(c) > max_cands:
                continue
            xs.append(s)
            ys.append(stoi[wl])
        if len(xs) >= 6:
            X.append(xs)
            Y.append(ys)
    return X, Y


if __name__ == "__main__":
    tr = load_sents("train.txt", 2000)
    itos, stoi, cand = build_vocab(tr)
    print("accented vocab:", len(itos))
    print("stripped types:", len(cand))
    X, Y = encode(tr, stoi, cand)
    print("encoded sents:", len(X), "tokens:", sum(len(x) for x in X))
    print(X[0][:8])
    print([itos[i] for i in Y[0][:8]])
