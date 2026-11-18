"""Matched RAW-line scoring of the frozen character CRF; no training or row filtering.

Keep the baseline's native 128-character non-overlapping windows and fixed-size
padding. Score case-sensitive alphanumeric runs at original gold character spans.
Unknown input characters are fed as PAD and counted, never cause a row to drop.
"""
import hashlib
import json
import re
import time
import unicodedata
from pathlib import Path
import numpy as np
import torch
from models import BiLSTMCRF

TOK = re.compile(r'[^\W_]+', re.UNICODE)

def strip_diac(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if not unicodedata.combining(c)).replace('đ','d').replace('Đ','D')

@torch.inference_mode()
def score_raw(model, vocab, path, limit=20000, bs=64, output=None):
    device = next(model.parameters()).device
    stoi = {c:i for i,c in enumerate(vocab)}
    lines = []
    with open(path,encoding='utf-8') as f:
        for line in f:
            if len(lines) == limit:
                break
            lines.append(line.rstrip('\r\n'))
    sources = [strip_diac(s) for s in lines]
    assert all(len(g)==len(s) for g,s in zip(lines,sources)), 'Non 1:1 normalization'
    predictions = [np.zeros(len(s),dtype=np.int32) for s in sources]
    windows = [(i,a,min(128,len(s)-a)) for i,s in enumerate(sources) for a in range(0,len(s),128)]
    unknown = sum(c not in stoi for s in sources for c in s)
    model.eval()
    start = time.perf_counter()
    for offset in range(0,len(windows),bs):
        chunk = windows[offset:offset+bs]
        x = np.zeros((len(chunk),128),dtype=np.int64)
        for j,(i,a,n) in enumerate(chunk):
            x[j,:n] = [stoi.get(c,0) for c in sources[i][a:a+n]]
        lens = torch.tensor([n for _,_,n in chunk], device=device)
        pred = model.crf.viterbi(model.emissions(torch.from_numpy(x).to(device)),lens).cpu().numpy()
        for j,(i,a,n) in enumerate(chunk):
            predictions[i][a:a+n]=pred[j,:n]
        if offset % (bs*100)==0:
            print('baseline windows',offset,'/',len(windows),flush=True)
    elapsed = time.perf_counter()-start
    correct=total=0
    records=[]
    for row,(g,ids) in enumerate(zip(lines,predictions)):
        p=[vocab[i] if len(vocab[i])==1 else '\ufffd' for i in ids]
        for match in TOK.finditer(g):
            a,b=match.span();pred=''.join(p[a:b]);hit=pred==match.group()
            total+=1;correct+=hit
            records.append((row,a,b,match.group(),pred,int(hit)))
    result=dict(lines=len(lines),tokens=total,correct=correct,accuracy=correct/max(total,1),
                unknown_input_chars=unknown,raw_file_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                inference_seconds=elapsed,window_length=128,device=str(device),
                protocol='First raw lines; all alphanumeric runs; case-sensitive; no gold filtering; native character CRF decoder')
    if output:
        with open(output,'w',encoding='utf8') as f:
            for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--vocab',required=True)
    p.add_argument('--raw',required=True);p.add_argument('--limit',type=int,default=20000);p.add_argument('--out',required=True)
    a=p.parse_args();torch.set_num_threads(2)
    vocab=json.loads(Path(a.vocab).read_text());model=BiLSTMCRF(len(vocab))
    model.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=True))
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    result=score_raw(model,vocab,a.raw,a.limit,output=a.out+'.predictions.jsonl')
    Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
