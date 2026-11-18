"""Dung du lieu muc am tiet: moi vi tri la mot am tiet, kem tap ung vien.

Luu rat gon: moi am tiet chi chiem 3 byte (2 byte ma dang bo dau + 1 byte chi so
ung vien dung). Tap ung vien duoc tra cuu tu bang, khong luu lap lai.
"""
import json, re, sys, unicodedata
import numpy as np

TOK = re.compile(r'[^\W_]+', re.UNICODE)
MAXS = 255          # so am tiet toi da moi cau
KMAX = 24           # so ung vien toi da moi vi tri

def strip_diac(s):
    d = unicodedata.normalize('NFD', s)
    d = ''.join(c for c in d if not unicodedata.combining(c))
    return d.replace('đ', 'd').replace('Đ', 'D')

def build_tables(cand_path):
    cand = json.load(open(cand_path))
    single = {k.lower(): v for k, v in cand.items() if ' ' not in k}
    bases = sorted(single)
    bid = {b: i + 1 for i, b in enumerate(bases)}        # 0 = khong dung
    # bang ung vien theo dang bo dau
    ctab = np.full((len(bases) + 1, KMAX), -1, dtype=np.int32)
    cn = np.zeros(len(bases) + 1, dtype=np.int8)
    vocab = {}
    for b in bases:
        vs = single[b][:KMAX]
        i = bid[b]
        ids = []
        for v in vs:
            v = v.lower()
            if v not in vocab:
                vocab[v] = len(vocab)
            ids.append(vocab[v])
        ctab[i, :len(ids)] = ids
        cn[i] = len(ids)
    cpos = {i: {int(v): j for j, v in enumerate(ctab[i, :cn[i]])} for i in range(1, len(bases) + 1)}
    return bid, ctab, cn, vocab, cpos

def prep(text_path, out_prefix, max_lines, bid, ctab, cn, vocab, cpos):
    idx_of = {v: i for i, v in enumerate(vocab)}
    sents = []            # so am tiet moi cau
    base_ids, gold = [], []
    skipped_uncov = skipped_long = n_tok = 0
    with open(text_path, encoding='utf-8') as f:
        for ln, line in enumerate(f):
            if max_lines and ln >= max_lines:
                break
            toks = TOK.findall(line)
            if not toks or len(toks) > MAXS:
                skipped_long += 1
                continue
            bs, gs, ok = [], [], True
            for t in toks:
                low = t.lower()
                b = strip_diac(low)
                i = bid.get(b)
                if i is None:
                    ok = False
                    break
                c = idx_of.get(low)
                m = cpos[i] if c is not None else None
                pos = m.get(c) if m else None
                if pos is None:
                    ok = False
                    break
                bs.append(i); gs.append(pos)
            if not ok:
                skipped_uncov += 1
                continue
            base_ids.extend(bs); gold.extend(gs); sents.append(len(bs))
            n_tok += len(bs)
    np.savez_compressed(out_prefix + '.npz',
                        sent=np.array(sents, np.uint8),
                        base=np.array(base_ids, np.uint32),
                        gold=np.array(gold, np.uint8))
    print('%s: %d cau, %d am tiet | bo %d cau thieu ung vien, %d cau qua dai' %
          (out_prefix, len(sents), n_tok, skipped_uncov, skipped_long))
    return n_tok

if __name__ == '__main__':
    text = sys.argv[1]
    out = sys.argv[2]
    maxl = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    bid, ctab, cn, vocab, cpos = build_tables('/home/dsh/ML/diac/data/cand.json')
    print('dang bo dau:', len(bid), '| dau ra:', len(vocab), '| max ung vien:', int(cn.max()))
    np.save(out + '_ctab.npy', ctab); np.save(out + '_cn.npy', cn)
    json.dump(list(vocab), open(out + '_vocab.json', 'w'), ensure_ascii=False)
    json.dump(sorted(bid), open(out + '_bases.json', 'w'), ensure_ascii=False)
    prep(text, out, maxl, bid, ctab, cn, vocab, cpos)
