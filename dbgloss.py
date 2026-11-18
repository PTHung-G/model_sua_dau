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
for std,wd,lr in [(1e-2,0.01,2e-3),(1e-1,0.0,2e-3),(1e-1,0.0,1e-2),(1e-1,0.0,3e-2)]:
    torch.manual_seed(0)
    m=AttnDiac(P["nchar"],P["max_c"],syl=32,dim=64,window=8)
    nn.init.normal_(m.out.weight,std=std); nn.init.zeros_(m.out.bias)
    opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=wd)
    for ep in range(6):
        m.train()
        for xc,cl,f,y,ln,mk in trb:
            lg=m(xc,cl); ar=torch.arange(lg.size(-1))[None,None,:]
            lg=lg.masked_fill(ar>=ln[...,None],NEG)
            loss=F.cross_entropy(lg[mk],y[mk]); opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    m.eval()
    with torch.no_grad():
        L=0;n=0;cor=0;tot=0;p0=0
        for xc,cl,f,y,ln,mk in trb:
            lg=m(xc,cl); ar=torch.arange(lg.size(-1))[None,None,:]
            lg=lg.masked_fill(ar>=ln[...,None],NEG)
            L+=F.cross_entropy(lg[mk],y[mk],reduction='sum').item(); n+=mk.sum().item()
            pr=lg.argmax(-1); cor+=((pr==y)&mk).sum().item(); tot+=mk.sum().item()
            p0+=(pr[mk]==0).sum().item()
        print(f"std={std} wd={wd} lr={lr}: TRAIN loss={L/n:.4f} acc={cor/tot:.4f} pred0={p0/tot:.3f}", flush=True)
