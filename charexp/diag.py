"""Chan doan loi cua char-CRF: bao nhieu loi co the cuu bang rang buoc ung vien.

Voi moi am tiet trong cau, ta biet dang bo dau (chinh la dau vao) va tap ung vien
co the co. Neu mo hinh doan ra mot dang KHONG nam trong tap do thi rang buoc
ung vien co the sua duoc. Neu doan ra mot dang NAM TRONG tap nhung sai thi
rang buoc khong giup gi, phai co ngu canh tot hon.
"""
import json, sys
import numpy as np
import torch

sys.path.insert(0, '.')
from models import BiLSTMCRF

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
L = 128
BS = 128

VOC = json.load(open('vocab.json'))
V = len(VOC)
ALNUM = np.array([c.isalnum() for c in VOC], dtype=bool)
CAND = json.load(open('/home/dsh/ML/diac/data/cand.json'))

d = np.load('test.npz')
src = d['src'].astype(np.int64)
tgt = d['tgt'].astype(np.int64)
lens = d['lens'].astype(np.int64)
off = np.zeros(len(lens), dtype=np.int64)
np.cumsum(lens[:-1], out=off[1:])
idx = np.arange(min(N, len(lens)))

model = BiLSTMCRF(V)
model.load_state_dict(torch.load('crf_full.pt', map_location='cpu'))
model.eval()
print('tham so', sum(p.numel() for p in model.parameters()), '| so cau', len(idx), flush=True)

wins = []
for i in idx:
    n = int(lens[i])
    for st in range(0, n, L):
        wins.append((int(i), st, min(L, n - st)))

X = np.zeros((len(wins), L), dtype=np.int64)
for k, (i, st, m) in enumerate(wins):
    X[k, :m] = src[off[i] + st: off[i] + st + m]

outs = []
with torch.no_grad():
    for k in range(0, len(wins), BS):
        chunk = wins[k:k + BS]
        x = torch.from_numpy(X[k:k + BS])
        ln = torch.tensor([c[2] for c in chunk], dtype=torch.long)
        outs.append(model.crf.viterbi(model.emissions(x), ln).numpy())
P = np.concatenate(outs, 0)

slot = {int(i): j for j, i in enumerate(idx)}
preds = [np.zeros(int(lens[i]), dtype=np.int64) for i in idx]
for k, (i, st, m) in enumerate(wins):
    preds[slot[i]][st:st + m] = P[k, :m]

n_syl = n_ok = 0
wrong_in_set = wrong_off_set = no_cand = 0
examples_off, examples_in = [], []
for j, i in enumerate(idx):
    n = int(lens[i])
    g = tgt[off[i]: off[i] + n]
    s = src[off[i]: off[i] + n]
    p = preds[j]
    al = ALNUM[g].astype(np.int8)
    dd = np.diff(np.concatenate(([0], al, [0])))
    for a, b in zip(np.where(dd == 1)[0], np.where(dd == -1)[0]):
        n_syl += 1
        gs = ''.join(VOC[c] for c in g[a:b])
        ps = ''.join(VOC[c] for c in p[a:b])
        bs = ''.join(VOC[c] for c in s[a:b])
        if gs == ps:
            n_ok += 1
            continue
        cands = CAND.get(bs.lower())
        if not cands:
            no_cand += 1
            continue
        low = {c.lower() for c in cands}
        if ps.lower() in low:
            wrong_in_set += 1
            if len(examples_in) < 8:
                examples_in.append((bs, gs, ps))
        else:
            wrong_off_set += 1
            if len(examples_off) < 12:
                examples_off.append((bs, gs, ps, cands[:6]))

print('\n=== KET QUA ===')
print('am tiet            %d' % n_syl)
print('dung               %d (%.2f%%)' % (n_ok, 100 * n_ok / n_syl))
err = n_syl - n_ok
print('sai tong           %d' % err)
print('  sai, ngoai tap ung vien   %d (%.1f%% so loi)  <- rang buoc co the cuu' %
      (wrong_off_set, 100 * wrong_off_set / max(err, 1)))
print('  sai, trong tap ung vien   %d (%.1f%% so loi)  <- can ngu canh tot hon' %
      (wrong_in_set, 100 * wrong_in_set / max(err, 1)))
print('  khong co trong bang ung vien %d' % no_cand)
print('\nvi du SAI VA NGOAI TAP (co the cuu bang rang buoc):')
for bs, gs, ps, cd in examples_off:
    print('  bo dau %-10s dung %-10s doan %-10s | ung vien %s' % (bs, gs, ps, cd))
print('\nvi du SAI NHUNG TRONG TAP (rang buoc khong giup):')
for bs, gs, ps in examples_in:
    print('  bo dau %-10s dung %-10s doan %-10s' % (bs, gs, ps))
