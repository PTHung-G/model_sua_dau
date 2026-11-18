"""CPU-only real-source preflight; does not perform optimizer updates."""
import hashlib,json,tempfile,sys,unittest
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'charexp'))
import build_punct_tensors_v1 as b
import syltrain_punct as t
import sylmodel_punct as m

def main():
    torch.set_num_threads(2)
    rows=[]
    with (ROOT/'punctuation_corpus_v1/train.jsonl').open() as f:
        for _,line in zip(range(2000),f):rows.append(json.loads(line))
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);source=root/'rows.jsonl';source.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        bases,vocab,cid,pid,n=b.build_table(source)
        stats=b.encode(source,root/'train',bases,vocab,cid,pid)
        (root/'manifest.json').write_text(json.dumps({'pipeline_schema':2}))
        data=t.load_data(root/'train');ids=np.load(root/'train_source_rows.npy')
        starts,lengths=t.segments(data[2],data[3],64)
        rng=np.random.default_rng(42);checks=0
        for idx in t.epoch_batches(lengths,16,rng):
            x,y,ln,ff,gg=t.make_batch(data,starts,lengths,idx,'cpu')
            for batch_row,k in enumerate(idx):
                start=int(starts[k]);doc=int(np.searchsorted(data[3],start,side='right')-1);delta=start-int(data[3][doc]);row=rows[int(ids[doc])]
                ms,gs=b.gaps(row['input']);gold=[z.group().lower() for z in b.TOK.finditer(row['target'])]
                expected=[pid.get(g[0],1) for g in gs]+[pid.get(gs[-1][1],1)]
                count=int(ln[batch_row])
                assert gg[batch_row,:count+1].tolist()==expected[delta:delta+count+1]
                reconstructed=[vocab[int(cid[int(bi),int(gi)])] for bi,gi in zip(x[batch_row,:count],y[batch_row,:count])]
                assert reconstructed==gold[delta:delta+count]
                checks+=count
        assert checks==len(data[0])
        nc,bc,cc,candidates,_=t.build_tensors(str(root/'train'))
        model=m.PunctSylCRF(nc,bc,cc,candidates,len(pid)).eval()
        x,y,ln,ff,gg=t.make_batch(data,starts,lengths,np.arange(min(4,len(starts))),'cpu')
        e,tr,mask=model(x,ln,ff,gg);loss=m.nll(e,tr,mask,y,ln)
        assert torch.isfinite(loss)
        loss.backward() # gradient smoke only, NO optimizer / training.
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        result={'pipeline_schema':2,'source_rows':len(rows),'retained_rows':stats['rows'],'tokens_checked_exactly':checks,'segments_checked':len(starts),'shuffled_batch_alignment':'PASS','candidate_targets_reconstructed':'PASS','forward_backward_finite':'PASS','optimizer_updates':0,'gpu_training':False}
    (ROOT/'punct_audit_real_preflight.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
