import torch, torch.nn as nn, torch.nn.functional as F
from data import load_sents
from fixbuild import build_fair
from scoring import batches
from attn_model import AttnDiac
torch.set_num_threads(2)
NEG=-1e9
tr=load_sents("train.txt",3000); te=load_sents("test.txt",400)
P=build_fair(tr,te); pri=P["pri"]
trb=list(batches(P["tr"],64))
# Overfit test: train on ONE batch of 64 sentences only. A working model must reach ~100%.
one=[trb[0]]
for std,lr,ep,wd in [(1e-2,2e-3,300,0.0),(1e-1,2e-3,300,0.0),(1e-1,1e-2,300,0.0)]:
    torch.manual_seed(0)
    m=AttnDiac(P["nchar"],P["max_c"],syl=32,dim=64,window=8)
    nn.init.normal_(m.out.weight,std=std); nn.init.zeros_(m.out.bias)
    opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=wd)
    xc,cl,f,y,ln,mk=one[0]
    for s in range(ep):
        m.train()
        lg=m(xc,cl); ar=torch.arange(lg.size(-1))[None,None,:]
        lg=lg.masked_fill(ar>=ln[...,None],NEG)
        loss=F.cross_entropy(lg[mk],y[mk]); opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    m.eval()
    with torch.no_grad():
        lg=m(xc,cl); ar=torch.arange(lg.size(-1))[None,None,:]
        lg=lg.masked_fill(ar>=ln[...,None],NEG)
        pr=lg.argmax(-1)
        amb=(ln>1)&mk
        acc=((pr==y)&mk).sum().item()/mk.sum().item()
        acc_amb=((pr==y)&amb).sum().item()/max(amb.sum().item(),1)
        print(f"std={std} lr={lr}: loss={loss.item():.4f} acc={acc:.4f} acc_ambiguous={acc_amb:.4f} pred0={(pr[mk]==0).float().mean().item():.3f}",flush=True)
