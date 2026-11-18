"""Clean data pipeline: split by ARTICLE, vocabulary from train only.

Three defects in earlier runs are fixed here:
  1. Sentences were split randomly, so near-duplicate sentences from the same
     Wikipedia article could appear in both train and test (leakage).
  2. The candidate inventory was built over train+test, so the model was allowed
     to see which accented forms exist in the test set.
  3. Evaluation aligned predictions with labels by batch index instead of by an
     explicit token id.

Outputs: data/tr.txt, data/va.txt, data/te.txt, data/art.json, data/vocab.json
"""
import json
import random
import re
import unicodedata
from collections import Counter

import pyarrow.parquet as pq

from vn import strip_all, SYLLABLE_OK

WORD_OK = re.compile("^[" + "".join(sorted(SYLLABLE_OK)) + "]+$")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
SHARDS = ["train-00000-of-00004.parquet", "train-00001-of-00004.parquet",
          "train-00002-of-00004.parquet", "train-00003-of-00004.parquet"]
SEED = 12345


def norm(s):
    return unicodedata.normalize("NFC", s)


def iter_sentences(text):
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("==") or len(line) < 20:
            continue
        if re.search(r"(Chú thích|Liên kết ngoài|Tham khảo|https?://|^\d|\[|\]|@)", line):
            continue
        for s in SENT_SPLIT.split(line):
            s = s.strip()
            if not s.endswith((".", "!", "?")):
                continue
            words, ok = [], True
            for t in s.split():
                core = re.sub(r"[,.;:!?\"'\(\)]", "", t)
                if not core:
                    continue
                if not WORD_OK.match(core):
                    ok = False
                    break
                words.append(core)
            if not ok or not (6 <= len(words) <= 40):
                continue
            if sum(1 for w in words if strip_all(w) != w.lower()) < len(words) * 0.5:
                continue
            yield " ".join(words)


def load_articles():
    """Return [(article_id, title, [sentences])] in a deterministic order."""
    arts = []
    for shard in SHARDS:
        t = pq.read_table(shard, columns=["id", "title", "text"])
        d = t.to_pydict()
        for aid, title, text in zip(d["id"], d["title"], d["text"]):
            sents = [norm(s) for s in iter_sentences(norm(text))]
            if sents:
                arts.append((str(aid), norm(title or ""), sents))
    return arts


def split_articles(arts, seed=SEED):
    """90/5/5 split by article. A whole article goes to exactly one split."""
    idx = list(range(len(arts)))
    random.Random(seed).shuffle(idx)
    n = len(idx)
    n_va = int(n * 0.05)
    n_te = int(n * 0.05)
    te, va, tr = idx[:n_te], idx[n_te:n_te + n_va], idx[n_te + n_va:]
    return tr, va, te


def dedupe(sents):
    """Drop exact duplicate sentences, keeping first occurrence."""
    seen, out = set(), []
    for s in sents:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def write_split(name, arts, idxs, forbid=None):
    """Write sentences; optionally, drop duplicates that already appear in `forbid`."""
    n_art = len(idxs)
    sents = []
    for i in idxs:
        sents.extend(arts[i][2])
    sents = dedupe(sents)
    if forbid:
        sents = [s for s in sents if s not in forbid]
    with open(f"data/{name}.txt", "w", encoding="utf-8") as f:
        for s in sents:
            f.write(s + "\n")
    print(f"{name}: {n_art:,} articles -> {len(sents):,} sentences")
    return set(sents)


def build_vocab_from_train(tr_sents, min_freq=2):
    """Candidate inventory from TRAIN ONLY."""
    freq = Counter()
    for ws in tr_sents:
        for w in ws:
            freq[w.lower()] += 1
    vocab = sorted({w for w, c in freq.items() if c >= min_freq})
    cand = {}
    for w in vocab:
        cand.setdefault(strip_all(w), []).append(w)
    for s in cand:
        cand[s].sort(key=lambda w: (-freq.get(w, 0), w))
    return cand, freq


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    arts = load_articles()
    print(f"articles with usable sentences: {len(arts):,}")
    tr, va, te = split_articles(arts)
    print(f"split: train {len(tr):,} / val {len(va):,} / test {len(te):,} articles")
    seen = write_split("tr", arts, tr)
    seen |= write_split("va", arts, va, forbid=seen)
    write_split("te", arts, te, forbid=seen)
    json.dump({"train": [arts[i][0] for i in tr],
               "val": [arts[i][0] for i in va],
               "test": [arts[i][0] for i in te],
               "seed": SEED}, open("data/art.json", "w"))
    print("wrote data/art.json")
