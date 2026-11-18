"""Strict checkpoint, sample and batch-invariance verification for context pilot."""
import argparse,json
from pathlib import Path
import torch
import context_pilot as cp
import pilot_arch as pa

def main(directory):
    root=Path(directory);r=json.loads((root/'results.json').read_text())
    assert r['status']=='complete'
    torch.set_num_threads(4)
    proofs=[]
    for arm in ('baseline','context'):
        for kind in ('latest','best'):
            model,p=cp.load_pilot_checkpoint(root/f'{arm}_{kind}.pt',device='cpu')
            assert p['provenance']['train']['sample_sha256']==r['provenance']['train']['sample_sha256']
            assert p['provenance']['validation']['sample_sha256']==r['provenance']['validation']['sample_sha256']
            assert sum(x.numel() for x in model.parameters())==p['params']
            expected=r['config']['steps'] if kind=='latest' else r['arms'][arm]['best']['step']
            assert p['step']==expected
            for k in ('optimizer','scheduler','rng','cand','forms','tone_table','mod_table'):assert k in p
            raw=[['toi','di','hoc'],['toi']*17]
            one=list(cp.raw_batches(raw[:1],p,1))[0][-1]
            batch=list(cp.raw_batches(raw,p,2))[0][-1]
            tables=tuple(p[k] for k in ('tone_table','mod_table'))
            with torch.inference_mode():
                a=pa.candidate_logits(model,'factorized',*one,tables)[0]
                b=pa.candidate_logits(model,'factorized',*batch,tables)[0,:a.shape[0]]
            delta=float((a-b).abs().max());assert torch.allclose(a,b,atol=1e-4,rtol=1e-5),delta
            assert torch.equal(a.argmax(-1),b.argmax(-1))
            proofs.append(dict(arm=arm,kind=kind,step=p['step'],params=p['params'],max_logit_delta=delta))
    for arm in ('baseline','context'):
        assert r['arms'][arm]['initial']['sentence_errors']==r['base_init']['sentence_errors']
    result=dict(verified=True,checkpoints=proofs)
    (root/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');main(p.parse_args().directory)
