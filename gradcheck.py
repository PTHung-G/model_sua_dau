import torch, torch.nn as nn, torch.nn.functional as F
from data import load_sents
from fixbuild import build_fair
from scoring import batches
from attn_model import AttnDiac
torch.set_num_threads(2)
NEG=-1e9
tr=load_sents("train.txt",4000); te=load_sents("test.txt",500)
P=build_fair(tr,te); pri=P["pri"]
for std in (1e-2, 1e-1):
    torch.manual_seed(0)
    m=AttnDiac(P["nchar"],P["max_c"],syl=32,dim=64,window=8)
    nn.init.normal_(m.out.weight,std=std); nn.init.zeros_(m.out.bias)
    xc,cl,f,y,ln,mk=next(iter(batches(P["tr"],64)))
    lg=m(xc,cl)
    ar=torch.arange(lg.size(-1))[None,None,:]
    lg=lg.masked_fill(ar>=ln[...,None],NEG)
    loss=F.cross_entropy(lg[mk],y[mk]); loss.backward()
    g={n:p.grad.abs().mean().item() for n,p in m.named_parameters() if p.grad is not None}
    print(f"--- out init std={std}  loss={loss.item():.4f}")
    for n in ["out.weight","norm.weight","blocks.0.att.q.weight","blocks.0.att.rel",
              "blocks.0.ff.0.weight","proj.weight","cgru.weight_hh_l0"]:
        print(f"   {n:24s} grad={g.get(n,0):.3e}")
