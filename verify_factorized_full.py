"""Strictly reload a trusted full factorized checkpoint and check padding invariance."""
import argparse
import json
from pathlib import Path
import torch
from pilot_arch import FactorizedDiac, candidate_logits


def verify(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    assert ck['arm'] == 'factorized'
    cfg = ck['config']
    assert cfg['packed_chars'] and cfg['window'] == 8
    sig = ck['signatures']
    model = FactorizedDiac(ck['nchar'], ck['max_c'], len(sig['modifier']),
                          len(sig['tone']), **cfg).eval()
    model.load_state_dict(ck['model'], strict=True)
    assert sum(p.numel() for p in model.parameters()) == ck['params']
    for key in ('cand','forms','char2id','optimizer','scheduler','rng','manifests'):
        assert key in ck, key
    fid = {s:i for i,s in enumerate(ck['forms'])}
    stems = ['toi','di','hoc']
    pad = ck['nchar']-1
    x = torch.full((1,3,3),pad,dtype=torch.long)
    cl = torch.tensor([[len(s) for s in stems]])
    f = torch.tensor([[fid[s] for s in stems]])
    ln = torch.tensor([[len(ck['cand'][s]) for s in stems]])
    for j,s in enumerate(stems):
        x[0,j,:len(s)] = torch.tensor([ck['char2id'][c] for c in s])
    bx = torch.full((2,7,10),pad,dtype=torch.long)
    bx[0,:3,:3] = x[0]
    bx[1] = 1
    bc = torch.ones((2,7),dtype=torch.long);bc[0,:3]=cl[0];bc[1]=10
    bf = torch.zeros((2,7),dtype=torch.long);bf[0,:3]=f[0]
    bl = torch.ones_like(bf);bl[0,:3]=ln[0]
    mask = torch.zeros_like(bf,dtype=torch.bool);mask[0,:3]=True;mask[1]=True
    tables = ck['tone_table'],ck['mod_table']
    with torch.inference_mode():
        a=candidate_logits(model,'factorized',x,cl,f,ln,torch.ones_like(f,dtype=torch.bool),tables)[0]
        b=candidate_logits(model,'factorized',bx,bc,bf,bl,mask,tables)[0,:3]
    delta=float((a-b).abs().max())
    assert torch.allclose(a,b,atol=1e-4,rtol=1e-5),delta
    assert torch.equal(a.argmax(-1),b.argmax(-1))
    r={'verified':True,'checkpoint':str(Path(path).resolve()),'step':ck['step'],
       'params':ck['params'],'padding_max_difference':delta,
       'tone_classes':len(sig['tone']),'modifier_classes':len(sig['modifier'])}
    Path(str(path)+'.verified.json').write_text(json.dumps(r,indent=2))
    print(json.dumps(r,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('checkpoint')
    verify(p.parse_args().checkpoint)
