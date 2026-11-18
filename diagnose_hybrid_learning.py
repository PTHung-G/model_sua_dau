"""Read-only fixed-batch CE and correction diagnostics for saved rerankers."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import hybrid_rerank_model as hm
import pilot_arch as pa
import context_pilot as cp
import sentence_loss_epoch as se
import sentence_loss_pilot as sl

ROOT=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(4)
    output=ROOT/'kaggle_hybrid_rerank_output'
    # Fixed subset of actually consumed source rows, not a newly sampled training set.
    order=np.load(output/'hybrid_rerank_control_results/epoch_order.npy')
    indices=order[:640000][np.random.default_rng(1731).choice(640000,512,replace=False)]
    wanted=set(map(int,indices));found={}
    with (ROOT/'data/tr.txt').open() as f:
        for i,line in enumerate(f):
            if i in wanted:found[i]=line.strip().split()
            if len(found)==len(wanted):break
    train=[found[int(i)] for i in indices]
    val,manifest=pa.reservoir(ROOT/'data/va.txt',512,1732)
    result={'description':'Fixed 512 consumed train rows and independent diagnostic reservoir of 512 validation rows; eval-mode CE on identical targets. Not a held-out test.',
            'validation_manifest':manifest,'arms':{}}
    for arm in ('control','hybrid'):
        folder=output/f'hybrid_rerank_{arm}_results'
        result['arms'][arm]={}
        for kind in ('best','latest'):
            model,payload=hm.load_reranker_checkpoint(folder/f'hybrid_rerank_{kind}.pt')
            measured={}
            for split,rows in [('consumed_train',train),('validation',val)]:
                totals=dict(tokens=0,base_ce_sum=0.,rerank_ce_sum=0.,changed_top1=0,
                            fixed_tokens=0,broken_tokens=0,live_candidates=0,abs_correction_sum=0.)
                with torch.inference_mode():
                    for start in range(0,len(rows),32):
                        batch,_=se.collate(rows[start:start+32],payload,'cpu')
                        if batch is None:continue
                        xc,cl,f,y,ln,mask,ok=batch
                        target=sl.training_target(mask,ok,ln)
                        _,base=model.frozen(xc,cl,f,ln,mask)
                        scores=model.logits(xc,cl,f,ln,mask)
                        totals['tokens']+=int(target.sum())
                        totals['base_ce_sum']+=float(F.cross_entropy(base[target],y[target],reduction='sum'))
                        totals['rerank_ce_sum']+=float(F.cross_entropy(scores[target],y[target],reduction='sum'))
                        before=base.argmax(-1)[target];after=scores.argmax(-1)[target];gold=y[target]
                        totals['changed_top1']+=int((before!=after).sum())
                        totals['fixed_tokens']+=int(((before!=gold)&(after==gold)).sum())
                        totals['broken_tokens']+=int(((before==gold)&(after!=gold)).sum())
                        live=target[...,None]&(torch.arange(scores.shape[-1])<ln[...,None])
                        totals['live_candidates']+=int(live.sum())
                        totals['abs_correction_sum']+=float((scores-base)[live].abs().sum())
                n=totals['tokens']
                totals.update(base_ce=totals['base_ce_sum']/n,rerank_ce=totals['rerank_ce_sum']/n,
                              mean_abs_correction=totals['abs_correction_sum']/totals['live_candidates'])
                measured[split]=totals
            result['arms'][arm][kind]=dict(step=payload['step'],metrics=measured)
            print(arm,kind,json.dumps(result['arms'][arm][kind]),flush=True)
            del model,payload
    (output/'learning_diagnostics.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
