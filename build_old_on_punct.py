"""Project punctuation-corpus test rows into the PREVIOUS best model's format.

The old model has its own candidate inventory and only case/boundary features,
so it cannot consume the new tensors. This builder reproduces its exact input
format for the same sentences, making the comparison fair: identical source
rows, identical scoring, each model in its native operating mode.
"""
import argparse, json, re
from pathlib import Path
import numpy as np

TOK = re.compile(r'[^\W_]+', re.UNICODE)
CHW = 16


def strip_diac(s):
    import unicodedata
    d = unicodedata.normalize('NFD', s)
    return ''.join(c for c in d if not unicodedata.combining(c)).replace('đ', 'd').replace('Đ', 'D')


def case_id(s):
    if s.islower(): return 1
    if s.isupper(): return 3
    if s.istitle(): return 2
    return 4


def boundary_id(gap, first=False):
    if any(c in gap for c in ".!?…\n"): return 3
    if "," in gap: return 2
    if gap.strip(): return 4
    return 5 if first else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', default='punctuation_corpus_v1')
    ap.add_argument('--old-inventory', default='charexp/rebalance_v1/train')
    ap.add_argument('--out', default='old_on_punct_v1')
    ap.add_argument('--splits', nargs='+', default=['validation', 'test'])
    a = ap.parse_args()

    inv = Path(a.old_inventory)
    bases = json.loads(Path(str(inv) + '_bases.json').read_text())
    vocab = json.loads(Path(str(inv) + '_vocab.json').read_text())
    ctab = np.load(str(inv) + '_ctab.npy')
    bid = {b: i + 1 for i, b in enumerate(bases)}
    vid = {v: i for i, v in enumerate(vocab)}
    lookup = [{} for _ in range(len(bases) + 1)]
    for bi in range(1, len(bases) + 1):
        for ci in range(ctab.shape[1]):
            vi = int(ctab[bi, ci])
            if vi < 0: break
            lookup[bi][vi] = ci

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    report = {}
    for split in a.splits:
        sent, base, gold, feat = [], [], [], []
        kept = skipped = 0
        with (Path(a.corpus) / (split + '.jsonl')).open(encoding='utf8') as f:
            for line in f:
                row = json.loads(line)
                source, target = row['input'], row['target']
                # Old model saw only its own tokenisation of stripped text.
                stripped = strip_diac(target)
                matches = list(TOK.finditer(stripped))
                forms = [m.group().lower() for m in TOK.finditer(target)]
                if len(matches) != len(forms) or not matches or len(matches) > 255:
                    skipped += 1; continue
                rb, rg, rf = [], [], []
                ok = True
                end = 0
                for i, (m, form) in enumerate(zip(matches, forms)):
                    b = strip_diac(form).lower()
                    bi = bid.get(b)
                    ci = lookup[bi].get(vid.get(form, -1)) if bi else None
                    if ci is None:
                        ok = False; break
                    rb.append(bi); rg.append(ci)
                    gap = stripped[end:m.start()]
                    rf.append((case_id(m.group()), boundary_id(gap, i == 0)))
                    end = m.end()
                if not ok:
                    skipped += 1; continue
                sent.append(len(rb)); base.extend(rb); gold.extend(rg); feat.extend(rf)
                kept += 1
        pref = out / split
        np.savez_compressed(str(pref) + '.npz', sent=np.asarray(sent, np.int64),
                            base=np.asarray(base, np.int32),
                            gold=np.asarray(gold, np.int16))
        np.save(str(pref) + '_input_features.npy', np.asarray(feat, np.uint8))
        report[split] = {'rows': kept, 'tokens': len(base), 'skipped': skipped,
                         'coverage_of_corpus_rows': kept / max(1, kept + skipped)}
        print(split, report[split], flush=True)
    (out / 'report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
