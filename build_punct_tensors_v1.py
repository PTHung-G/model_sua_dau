"""Build punctuation-aware syllable tensors from punctuation_corpus_v1 JSONL.
Only train targets build the candidate inventory; validation/test labels never do.
Each word position receives case, exact preceding punctuation-gap ID, and exact
following punctuation-gap ID derived from the input only.
"""
import argparse, collections, json, re, unicodedata
from pathlib import Path
import numpy as np

TOK = re.compile(r'[^\W_]+', re.UNICODE)
KMAX = 32
CHW = 16
PAD_GAP = '<PAD>'
UNK_GAP = '<UNK>'
SCHEMA = 2

def strip_diac(s):
    d = unicodedata.normalize('NFD', s)
    return ''.join(c for c in d if not unicodedata.combining(c)).replace('đ','d').replace('Đ','D')

def case_id(s):
    if s.islower(): return 1
    if s.isupper(): return 3
    if s.istitle(): return 2
    return 4

def gaps(source):
    ms=list(TOK.finditer(source)); out=[]
    for i,m in enumerate(ms):
        prev=source[ms[i-1].end():m.start()] if i else source[:m.start()]
        nxt=source[m.end():ms[i+1].start()] if i+1<len(ms) else source[m.end():]
        # Preserve all punctuation and whitespace in a compact exact context key.
        out.append((''.join(c for c in prev if not c.isalnum()), ''.join(c for c in nxt if not c.isalnum())))
    return ms,out

def iter_rows(path):
    with open(path, encoding='utf-8') as f:
        for line in f:
            x=json.loads(line); yield x['input'],x['target']


def validate_pair(source,target):
    if unicodedata.normalize('NFC',target) != target or strip_diac(target) != source:
        raise ValueError('Pair is not NFC target with exact stripped input')
    ms=list(TOK.finditer(source)); gt=list(TOK.finditer(target))
    if len(ms)!=len(gt) or [strip_diac(m.group()).lower() for m in gt] != [m.group().lower() for m in ms]:
        raise ValueError('Word alignment mismatch')
    if len(source)!=len(target) or TOK.sub('',source)!=TOK.sub('',target):
        raise ValueError('Punctuation/spacing or character alignment mismatch')

def build_table(train_path, max_rows=0):
    forms=collections.defaultdict(collections.Counter); pc=collections.Counter(); n=0
    for source,target in iter_rows(train_path):
        validate_pair(source,target)
        ms,gs=gaps(source); gt=[m.group().lower() for m in TOK.finditer(target)]
        if len(ms)!=len(gt): continue
        for m,(pre,nxt),gold in zip(ms,gs,gt):
            base=strip_diac(gold).lower()
            if not base: continue
            forms[base][gold]=forms[base].get(gold,0)+1
            pc[pre]+=1;pc[nxt]+=1
        n+=1
        if max_rows and n>=max_rows: break
    # Deduplicate: an empty gap can also appear in most_common and would otherwise
    # be assigned two different ids, leaving one id outside the embedding range.
    punct=[PAD_GAP, UNK_GAP, '']
    for x,_ in pc.most_common(8189):
        if x not in punct: punct.append(x)
    pid={x:i for i,x in enumerate(punct)}
    bases=sorted(forms); vocab=[]; vid={}
    ctab=np.full((len(bases)+1,KMAX),-1,np.int32)
    for bi,b in enumerate(bases,1):
        for form,_ in forms[b].most_common(KMAX):
            if form not in vid: vid[form]=len(vocab);vocab.append(form)
            if np.sum(ctab[bi]>=0)<KMAX: ctab[bi,np.sum(ctab[bi]>=0)]=vid[form]
    return bases,vocab,ctab,pid,n

def encode(path,out,bases,vocab,ctab,pid,max_rows=0,progress=0):
    from array import array
    bid={b:i+1 for i,b in enumerate(bases)}; vid={v:i for i,v in enumerate(vocab)}
    lookup=[{} for _ in range(len(bases)+1)]
    for bi in range(1,len(bases)+1):
        for ci in range(ctab.shape[1]):
            vi=int(ctab[bi,ci])
            if vi<0: break
            lookup[bi][vi]=ci
    # Compact typed accumulators: 9 bytes/token, not ~36 bytes per Python int.
    sent=array('i'); base=array('i'); gold=array('h'); feat=array('B'); gap=array('H')
    kept=skipped=0; scanned=0; source_rows=array('q')
    for source,target in iter_rows(path):
        scanned+=1
        if progress and scanned%progress==0:
            print(f'  encode scanned={scanned:,} kept={kept:,} tokens={len(base):,}',flush=True)
        validate_pair(source,target)
        ms,gs=gaps(source); forms=[m.group().lower() for m in TOK.finditer(target)]
        if len(ms)!=len(forms) or not ms or len(ms)>256: skipped+=1;continue
        rowb=[];rowg=[];rowf=[];ok=True
        for m,(pre,_nxt),form in zip(ms,gs,forms):
            b=strip_diac(form).lower(); bi=bid.get(b)
            if bi is None: ok=False;break
            ci=lookup[bi].get(vid.get(form,-1))
            if ci is None: ok=False;break
            rowb.append(bi);rowg.append(ci);rowf.append(case_id(m.group()))
        if not ok: skipped+=1;continue
        for m,(pre,_nxt) in zip(ms,gs):
            gap.append(pid.get(pre,1))
        gap.append(pid.get(gs[-1][1],1))
        source_rows.append(scanned-1)
        sent.append(len(rowb));base.extend(rowb);gold.extend(rowg);feat.extend(rowf);kept+=1
        if max_rows and kept>=max_rows:break
    if sorted(pid.values()) != list(range(len(pid))):
        raise ValueError('Gap IDs must be dense and ordered')
    npid=len(pid)
    if gap:
        worst=max(gap)
        assert worst<npid, f'gap id {worst} outside vocabulary of {npid}'
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    np.savez(str(out)+'.npz',sent=np.frombuffer(sent,dtype=np.int32).astype(np.int64),
             base=np.frombuffer(base,dtype=np.int32).astype(np.int32),
             gold=np.frombuffer(gold,dtype=np.int16).astype(np.int16))
    np.save(str(out)+'_input_features.npy',np.frombuffer(feat,dtype=np.uint8))
    np.save(str(out)+'_source_rows.npy',np.frombuffer(source_rows,dtype=np.int64))
    np.save(str(out)+'_gaps.npy',np.frombuffer(gap,dtype=np.uint16))
    np.save(str(out)+'_ctab.npy',ctab)
    (Path(str(out)+'_bases.json')).write_text(json.dumps(bases,ensure_ascii=False))
    (Path(str(out)+'_vocab.json')).write_text(json.dumps(vocab,ensure_ascii=False))
    (Path(str(out)+'_punct.json')).write_text(json.dumps([k for k,v in sorted(pid.items(),key=lambda kv:kv[1])],ensure_ascii=False))
    return {'pipeline_schema':SCHEMA,'selection':'candidate-covered training/selection subset ONLY; never use as headline raw benchmark','rows':kept,'tokens':len(gold),'skipped':skipped,'bases':len(bases),'forms':len(vocab),'punct_vocab':len(pid)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--corpus',default='punctuation_corpus_v1');ap.add_argument('--out',default='punct_tensors_v1');ap.add_argument('--max-train-rows',type=int,default=0);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    root=Path(a.corpus); out=Path(a.out);out.mkdir(exist_ok=False)
    # Smoke mode must bound the inventory scan too, not only the encoded rows.
    inventory_rows=a.max_train_rows or (4000 if a.smoke else 0)
    bases,vocab,ctab,pid,n=build_table(root/'train.jsonl',inventory_rows)
    result={'pipeline_schema':SCHEMA,'inventory_rows':n,'train':encode(root/'train.jsonl',out/'train',bases,vocab,ctab,pid,256 if a.smoke else 0,500000),'validation':encode(root/'validation.jsonl',out/'validation',bases,vocab,ctab,pid,64 if a.smoke else 0,25000),'test':encode(root/'test.jsonl',out/'test',bases,vocab,ctab,pid,64 if a.smoke else 0,25000)}
    (out/'manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
