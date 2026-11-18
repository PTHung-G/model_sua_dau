import numpy as np, json, collections, sys

SPLITS = [('train','../data/char_train',25_000_000),
          ('val','../data/char_val',None),
          ('test','../data/char_test',None),
          ('testnews','../data/char_testnews',None)]

def read(p):
    return open(p, encoding='utf-8').read().split('\n')

# 1. vocab từ target của mọi split
chars = collections.Counter()
raw = {}
for name, pref, _ in SPLITS:
    src = read(pref + '.src'); tgt = read(pref + '.tgt')
    raw[name] = (src, tgt)
    for s in tgt: chars.update(s)
vocab = ['<pad>'] + sorted(chars)
stoi = {c: i for i, c in enumerate(vocab)}
print('vocab size:', len(vocab))

PAD = 0
def encode(name, cap):
    src, tgt = raw[name]
    s_ids, t_ids, lens = [], [], []
    total = 0
    for a, b in zip(src, tgt):
        if cap and total >= cap: break
        n = len(a)
        if n < 8: continue
        if n > 256:
            a, b, n = a[:256], b[:256], 256
        try:
            sa = [stoi[c] for c in a]
        except KeyError:
            continue          # ký tự lạ, bỏ dòng
        if any(c not in stoi for c in b):
            continue
        s_ids.extend(sa); t_ids.extend([stoi[c] for c in b]); lens.append(n)
        total += n
    np.savez(f'{name}.npz', src=np.array(s_ids, np.int32), tgt=np.array(t_ids, np.int32),
             lens=np.array(lens, np.int32))
    print(f'{name}: {len(lens)} dòng, {total:,} ký tự, maxlen {max(lens)}')

for name, _, cap in SPLITS:
    encode(name, cap)

json.dump(vocab, open('vocab.json','w'), ensure_ascii=False)
