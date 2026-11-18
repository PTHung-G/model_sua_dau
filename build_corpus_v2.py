"""Corpus v2: Wikipedia (as before) enriched with open Vietnamese news.

WHY
---
Measured on the public 4-register benchmark (nrl-ai/vn-diacritic-eval), the
candidate inventory built from Wikipedia only covers 78-79% of tokens
(business 77.9 / formal 87.8 / conv 74.5 / literary 77.0). ~22% of tokens are
unanswerable by construction. That is a vocabulary gap, not an architecture gap.

Wikipedia also repeats itself: 2.59M sentences from 916,918 articles, and the 1M
model saturated at 93.31% after ~3 epochs. More Wikipedia is not the answer;
different registers are.

WHAT CHANGES
------------
  * Wikipedia: split, article IDs and test set are BYTE-IDENTICAL to corpus v1,
    so `data/te.txt` stays a valid, uncontaminated holdout for v2.
  * News: BKAINewsCorpus + vietnamnet-news, used for TRAIN ONLY. No article-level
    identity is available for the dump, so news is never evaluated on directly.
  * NEW holdout `data/te_news.txt`: news sentences that were held out BEFORE the
    training sample was drawn, for cross-register measurement.
  * Wikipedia train is DEDUPLICATED against news so identical sentences cannot
    appear on both sides of a train/test boundary.
  * Wikipedia is downsampled to cap its dominance (news must not be a 20% minority
    that the model ignores).

Outputs: data/tr.txt (rewritten), data/tr_wiki.txt, data/tr_news.txt,
         data/te_news.txt, data/cand.json (rebuilt), data/v2_report.json.
         data/va.txt, data/te.txt, data/art.json are untouched.

Usage:
    python build_corpus_v2.py --wiki-limit 1500000 --news-cap 1500000
"""
import argparse
import hashlib
import json
import os
import random
import re
import unicodedata
from collections import Counter

from vn import strip_all, SYLLABLE_OK
from build_clean import WORD_OK, SENT_SPLIT, iter_sentences, norm

SEED = 12345
NEWS_FILES = ["domain/BKAINewsCorpus.txt", "domain/vietnamnet-news.txt"]
NEWS_HOLDOUT_FRAC = 0.04      # ~4% of news sentences reserved for te_news.txt


def sents_from_blob(text):
    """Same acceptance rules as build_clean.iter_sentences, applied to one blob."""
    return [norm(s) for s in iter_sentences(norm(text))]


def stream_news(path, cap, rng):
    """Yield sentences from a one-document-per-line file, deduped, capped.

    Reservoir-free: we take the first `cap` unique sentences. News dumps are
    roughly time-ordered, so this is a convenience sample, not a random one;
    accept for training, which only needs volume and variety.
    """
    seen, out = set(), []
    with open(path, encoding="utf-8") as f:
        for line in f:
            for s in sents_from_blob(line):
                if s in seen:
                    continue
                seen.add(s)
                out.append(s)
                if len(out) >= cap:
                    return out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki-limit", type=int, default=1_500_000,
                    help="max Wikipedia train sentences kept (v1 had 2,590,435)")
    ap.add_argument("--news-cap", type=int, default=1_500_000,
                    help="max news sentences before the holdout split")
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(SEED)

    # ---------- 1. Wikipedia train (reuse v1 file; do NOT rebuild the split) -----
    wiki_tr = [l.rstrip("\n") for l in open("data/tr.txt", encoding="utf-8")]
    print(f"wikipedia train (v1): {len(wiki_tr):,} sentences", flush=True)
    rng.shuffle(wiki_tr)
    wiki_tr = wiki_tr[:args.wiki_limit]
    print(f"wikipedia train (v2, downsampled): {len(wiki_tr):,}", flush=True)

    # ---------- 2. News: hold out BEFORE sampling the training pool ------------
    news_tr, news_ho = [], []
    per_source = max(1, args.news_cap // len(NEWS_FILES))
    for path in NEWS_FILES:
        got = stream_news(path, per_source, rng)
        cut = int(len(got) * NEWS_HOLDOUT_FRAC)
        news_ho.extend(got[:cut])
        news_tr.extend(got[cut:])
        print(f"  {os.path.basename(path):28s} kept {len(got):,} "
              f"(holdout {cut:,})", flush=True)
    print(f"news train={len(news_tr):,}  news holdout={len(news_ho):,}", flush=True)

    # ---------- 3. Decontaminate -------------------------------------------------
    # Wikipedia test stays frozen in data/te.txt; drop any news/train sentence that
    # collides with it, and drop wikipedia train sentences that collide with news
    # holdout. Exact-duplicate collisions only — cheap and unambiguous.
    te_frozen = set(l.rstrip("\n") for l in open("data/te.txt", encoding="utf-8"))
    va_frozen = set(l.rstrip("\n") for l in open("data/va.txt", encoding="utf-8"))
    ho_set = set(news_ho)

    news_tr = [s for s in news_tr if s not in te_frozen and s not in va_frozen]
    news_ho = [s for s in news_ho if s not in te_frozen and s not in va_frozen]
    n_before = len(wiki_tr)
    wiki_tr = [s for s in wiki_tr if s not in ho_set]
    print(f"decontaminate: wiki_train dropped {n_before - len(wiki_tr):,} "
          f"sentences repeating the news holdout; news filtered against frozen test",
          flush=True)

    # interleave rather than concatenate: batches must mix registers, otherwise the
    # model sees whole epochs of one style and the last block dominates.
    mixed = []
    it_w, it_n = iter(wiki_tr), iter(news_tr)
    exhausted = {"w": False, "n": False}
    while not (exhausted["w"] and exhausted["n"]):
        for key, it, sink in (("w", it_w, wiki_tr), ("n", it_n, news_tr)):
            if exhausted[key]:
                continue
            try:
                mixed.append(next(it))
            except StopIteration:
                exhausted[key] = True
    print(f"mixed train: {len(mixed):,} sentences "
          f"({len(wiki_tr)/len(mixed):.1%} wiki / {len(news_tr)/len(mixed):.1%} news)",
          flush=True)

    for name, data in [("tr", mixed), ("tr_wiki", wiki_tr), ("tr_news", news_tr),
                       ("te_news", news_ho)]:
        with open(os.path.join(args.out_dir, f"{name}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(data) + "\n")
        print(f"wrote {name}.txt: {len(data):,}")

    # ---------- 4. Rebuild candidate inventory from the NEW train only -----------
    freq = Counter()
    for s in mixed:
        for w in s.split():
            freq[w.lower()] += 1
    vocab = sorted({w for w, c in freq.items() if c >= 2})
    cand = {}
    for w in vocab:
        cand.setdefault(strip_all(w), []).append(w)
    for s in cand:
        cand[s].sort(key=lambda w: (-freq.get(w, 0), w))
    json.dump(cand, open(os.path.join(args.out_dir, "cand.json"), "w"),
              ensure_ascii=False)

    old = json.load(open("data/cand.json")) if os.path.exists("data/cand.json") else {}
    n_forms = sum(len(v) for v in cand.values())
    print(f"candidate inventory: {len(cand):,} stripped groups / {n_forms:,} forms "
          f"(max_c={max(len(v) for v in cand.values())})", flush=True)

    # ---------- 5. Coverage report, before anyone trains ------------------------
    import subprocess, sys
    report = {
        "wiki_train_sents": len(wiki_tr),
        "news_train_sents": len(news_tr),
        "news_holdout_sents": len(news_ho),
        "mixed_train_sents": len(mixed),
        "groups": len(cand),
        "forms": n_forms,
        "max_c": max(len(v) for v in cand.values()),
        "nchar": len({c for s in cand for c in s}) + 2,
    }
    json.dump(report, open(os.path.join(args.out_dir, "v2_report.json"), "w"), indent=1)
    print("\nv2 report:", json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
