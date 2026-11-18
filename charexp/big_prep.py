"""Dung lai tap train muc ky tu tu toan bo tr.txt, giu nguyen val/test cu.

Ky tu goc -> ky tu bo dau la anh xa 1:1 va giu thu tu, nen chi can tach dau
tung ky tu la duoc cap (src, tgt) thang hang.
"""
import json, sys, unicodedata
import numpy as np

VOCAB = json.load(open('vocab.json'))
STOI = {c: i for i, c in enumerate(VOCAB)}

# bang dich ky tu -> mot code unit, de encode nhanh qua UTF-16
TBL = str.maketrans({c: chr(i) for c, i in STOI.items() if len(c) == 1})

CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 300_000_000
OUT = sys.argv[2] if len(sys.argv) > 2 else 'train_big.npz'


def strip_diac(s):
    d = unicodedata.normalize('NFD', s)
    d = ''.join(c for c in d if not unicodedata.combining(c))
    return d.replace('đ', 'd').replace('Đ', 'D')


def encode(s):
    return np.frombuffer(s.translate(TBL).encode('utf-16-le'), dtype='<u2')


def kept(s):
    return all(c in STOI for c in s)


src_buf = np.empty(CAP, dtype=np.uint8)
tgt_buf = np.empty(CAP, dtype=np.uint8)
lens = []
total = 0
bad = 0
n_lines = 0

with open('../data/tr.txt', encoding='utf-8') as f:
    for line in f:
        if total >= CAP:
            break
        line = line.rstrip('\n')
        if len(line) < 8:
            continue
        if len(line) > 256:
            line = line[:256]
        s = strip_diac(line)
        if len(s) != len(line):
            bad += 1
            continue
        if not (kept(line) and kept(s)):
            bad += 1
            continue
        n = len(line)
        if total + n > CAP:
            break
        src_buf[total:total + n] = encode(s)
        tgt_buf[total:total + n] = encode(line)
        lens.append(n)
        total += n
        n_lines += 1

print('giu %d dong, %d ky tu, bo %d dong' % (n_lines, total, bad))
np.savez_compressed(OUT,
                    src=src_buf[:total], tgt=tgt_buf[:total],
                    lens=np.array(lens, dtype=np.int32))
import os
print('ghi %s, %.0f MB' % (OUT, os.path.getsize(OUT) / 1e6))
