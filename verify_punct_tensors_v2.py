"""Exhaustive schema2 tensor-to-raw audit. No model training or gold filtering.
Run after the builder has completed; verifies every retained row and reports dropped rows.
"""
import json,sys
from pathlib import Path
import numpy as np
import build_punct_tensors_v1 as b
ROOT=Path(__file__).resolve().parent

def verify(root,corpus):
    manifest=json.loads((root/'manifest.json').read_text())
    assert manifest['pipeline_schema']==2
    reports={}
    bases=json.loads((root/'train_bases.json').read_text());vocab=json.loads((root/'train_vocab.json').read_text());punct=json.loads((root/'train_punct.json').read_text())
    assert len(set(punct))==len(punct) and punct[:3]==[b.PAD_GAP,b.UNK_GAP,'']
    pid={s:i for i,s in enumerate(punct)}
    ctab=np.load(root/'train_ctab.npy',mmap_mode='r')
    assert ctab.shape==(len(bases)+1,b.KMAX) and ctab.min()==-1 and ctab.max()<len(vocab)
    for name in ('train','validation','test'):
        with np.load(root/(name+'.npz')) as z:sent=z['sent'];base=z['base'];gold=z['gold']
        features=np.load(root/(name+'_input_features.npy'),mmap_mode='r');gap=np.load(root/(name+'_gaps.npy'),mmap_mode='r');source_rows=np.load(root/(name+'_source_rows.npy'),mmap_mode='r')
        assert len(source_rows)==len(sent) and np.all(source_rows[1:]>source_rows[:-1])
        assert int(sent.sum())==len(base)==len(gold)==len(features) and len(gap)==len(base)+len(sent)
        assert sent.min()>0 and sent.max()<=256 and gap.max()<len(punct)
        k=wo=go=raw_count=0
        for raw_id,(source,target) in enumerate(b.iter_rows(corpus/(name+'.jsonl'))):
            raw_count+=1
            if k>=len(sent) or raw_id!=int(source_rows[k]):continue
            b.validate_pair(source,target)
            ms,gs=b.gaps(source);targets=[m.group().lower() for m in b.TOK.finditer(target)];n=len(ms)
            assert n==int(sent[k]),(name,k,'sentence length')
            bb=base[wo:wo+n];gg=gold[wo:wo+n]
            assert [bases[int(i)-1] for i in bb]==[m.group().lower() for m in ms],(name,k,'bases')
            assert [vocab[int(ctab[int(i),int(j)])] for i,j in zip(bb,gg)]==targets,(name,k,'targets')
            assert features[wo:wo+n].tolist()==[b.case_id(m.group()) for m in ms],(name,k,'case')
            assert gap[go:go+n+1].tolist()==[pid.get(x[0],1) for x in gs]+[pid.get(gs[-1][1],1)],(name,k,'gaps')
            k+=1;wo+=n;go+=n+1
            if k%500000==0:print(name,k,'verified',flush=True)
        assert k==len(sent) and wo==len(base) and go==len(gap)
        reports[name]={'raw_rows':raw_count,'verified_rows':k,'verified_tokens':wo,'excluded_rows':raw_count-k}
        print(name,reports[name],flush=True)
    result={'pipeline_schema':2,'all_retained_rows_match_raw':True,'splits':reports}
    (root/'alignment_verification.json').write_text(json.dumps(result,indent=2));return result
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=ROOT/'punct_tensors_audit_v2');ap.add_argument('--corpus',type=Path,default=ROOT/'punctuation_corpus_v1');a=ap.parse_args();verify(a.root,a.corpus)
