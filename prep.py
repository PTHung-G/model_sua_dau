"""Build train/test sentence files from Vietnamese Wikipedia and report ambiguity stats."""
import re
import sys
import pandas as pd
from collections import Counter
from vn import strip_all, SYLLABLE_OK as _SOK
import re as _re
WORD_OK = _re.compile("^[" + "".join(sorted(_SOK)) + "]+$")

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def iter_sentences(text):
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("==") or len(line) < 20:
            continue
        # drop wiki-ish lines (captions, references, link lists)
        if re.search(r"(Chú thích|Liên kết ngoài|Tham khảo|https?://|^\d|\[|\]|@)", line):
            continue
        for s in SENT_SPLIT.split(line):
            s = s.strip()
            if not s.endswith((".", "!", "?")):
                continue
            toks = s.split()
            words = []
            ok = True
            for t in toks:
                core = re.sub(r"[,.;:!?\"'\(\)]", "", t)
                if not core:
                    continue
                if not WORD_OK.match(core):
                    ok = False
                    break
                words.append(core)
            if not ok:
                continue
            if not (6 <= len(words) <= 40):
                continue
            if sum(1 for t in words if strip_all(t) != t.lower()) < len(words) * 0.5:
                continue  # mostly Vietnamese with diacritics
            yield " ".join(words)


def build(parquet, out, limit, skip=0):
    df = pd.read_parquet(parquet, columns=["text"])
    n = 0
    seen = 0
    with open(out, "w", encoding="utf-8") as f:
        for text in df.text:
            if seen < skip:
                seen += 1
                continue
            seen += 1
            for s in iter_sentences(text):
                f.write(s + "\n")
                n += 1
                if n >= limit:
                    print(f"{out}: {n} cau")
                    return n
    print(f"{out}: {n} cau")
    return n


def append_from(parquet, out, limit):
    """Append more sentences from another parquet shard."""
    df = pd.read_parquet(parquet, columns=["text"])
    n = 0
    with open(out, "a", encoding="utf-8") as f:
        for text in df.text:
            for s in iter_sentences(text):
                f.write(s + "\n")
                n += 1
                if n >= limit:
                    print(f"appended {n} cau")
                    return n
    print(f"appended {n} cau")
    return n


def ambiguity_report(path, max_lines=None):
    """How many accented forms compete for a stripped form?"""
    cand = {}
    n_tok = 0
    n_amb = 0
    freq = Counter()
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            for w in line.split():
                wl = w.lower()
                s = strip_all(wl)
                cand.setdefault(s, set()).add(wl)
                freq[s] += 1
                n_tok += 1
                if len(cand[s]) > 1:
                    n_amb += 1
    amb_final = sum(1 for s, c in cand.items() if len(c) > 1)
    print(f"\ntoken: {n_tok:,}   kieu am tiet khong dau: {len(cand):,}")
    print(f"kieu co >=2 ung vien: {amb_final:,} ({100*amb_final/len(cand):.1f}%)")
    print(f"token nam o kieu co >=2 ung vien: {n_amb:,} ({100*n_amb/n_tok:.1f}%)")
    ks = sorted(len(c) for c in cand.values())
    import statistics
    print(f"so ung vien / kieu: median {statistics.median(ks):.0f}  max {max(ks)}")
    print("\nvi du kieu nhieu ung vien nhat:")
    for s, c in sorted(cand.items(), key=lambda x: -len(x[1]))[:6]:
        print(f"   {s:7s} -> {', '.join(sorted(c))}")
    return cand


if __name__ == "__main__":
    build("train-00001-of-00004.parquet", "train.txt", int(sys.argv[1]) if len(sys.argv) > 1 else 300000)
    build("train-00002-of-00004.parquet", "test.txt", 20000)
    ambiguity_report("train.txt", max_lines=100000)
