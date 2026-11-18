"""Cache scores once; calibration analysis must not modify model weights."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import hybrid_rerank_model as hm
import context_pilot as cp

ROOT=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(4)
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',default=str(ROOT/'correction_calibration'))
    parser.add_argument('--splits',nargs='+',default=['calibration','evaluation'])
    args=parser.parse_args()
    out=Path(args.output)
    samples=json.loads((out/'samples.json').read_text())
    checkpoint=ROOT/'kaggle_hybrid_rerank_output/hybrid_rerank_hybrid_results/hybrid_rerank_latest.pt'
    model,payload=hm.load_reranker_checkpoint(checkpoint)
    manifest={'checkpoint':str(checkpoint),'step':payload['step'],'splits':{}}
    for split in args.splits:
        if (out/f'{split}_scores.npz').exists():raise FileExistsError('Do not overwrite cached scores')
        entries=samples[split]
        raw=[e['words'] for e in entries]
        base_all=[];rerank_all=[];gold=[];sentence=[];position=[];length=[];counts=[];metadata=[]
        with torch.inference_mode():
            for offset,inputs,rows,tensors in cp.raw_batches(raw,payload,bs=32):
                xc,cl,f,ln,mask=tensors
                _,base=model.frozen(xc,cl,f,ln,mask)
                scores=model.logits(xc,cl,f,ln,mask)
                base=base.cpu().numpy();scores=scores.cpu().numpy()
                for b,ws in enumerate(inputs):
                    known={pos:(t,stem) for t,(pos,stem) in enumerate(rows[b])}
                    for pos,word in enumerate(raw[offset+b]):
                        if not cp.sa.is_scoring_token(word):continue
                        target=cp.sa.normalized_core(word)
                        if pos in known:
                            t,stem=known[pos];options=[cp.sa.normalized_core(w) for w in payload['cand'][stem]]
                            bv=base[b,t];rv=scores[b,t];n=len(options)
                            y=options.index(target) if target in options else -1
                        else:
                            stem=cp.sa.normalized_core(ws[pos]);options=[stem];n=1
                            bv=np.full(model.max_c,-1e9,dtype=np.float32);bv[0]=0;rv=bv.copy()
                            y=0 if stem==target else -1
                        base_all.append(bv);rerank_all.append(rv);gold.append(y);counts.append(n)
                        sentence.append(offset+b);position.append(pos)
                        length.append(sum(cp.sa.is_scoring_token(w) for w in raw[offset+b]))
                        metadata.append(dict(sentence=offset+b,position=pos,stem=stem,target=target,
                                             options=options,known=pos in known))
                print(split,min(offset+32,len(raw)),len(raw),flush=True)
        np.savez_compressed(out/f'{split}_scores.npz',base=np.array(base_all),rerank=np.array(rerank_all),
                            gold=np.array(gold),sentence=np.array(sentence),position=np.array(position),
                            length=np.array(length),counts=np.array(counts))
        (out/f'{split}_tokens.json').write_text(json.dumps(metadata,ensure_ascii=False))
        manifest['splits'][split]={'sentences':len(raw),'tokens':len(gold)}
    manifest['checkpoint_sha256']=hashlib.file_digest(checkpoint.open('rb'),'sha256').hexdigest()
    (out/'score_manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
