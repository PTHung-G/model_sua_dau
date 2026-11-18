import torch, torch.nn as nn, torch.nn.functional as F
from data import load_sents
from fixbuild import build_fair
from scoring import batches
from attn_model import AttnDiac
torch.set_num_threads(2)
NEG=-1e9
tr=load_sents("train.txt",8000); te=load_sents("test.txt",1500)
P=build_fair(tr,te); pri=P["pri"]
torch.manual_seed(0)
m=AttnDiac(P["nchar"],P["max_c"],syl=32,dim=64,window=8)
def lg_of(model,xc,cl,f,ln,use_prior):
    lg=model(xc,cl)
    if use_prior: lg=lg+pri[f]
    ar=torch.arange(lg.size(-1))[None,None,:]
    return lg.masked_fill(ar>=ln[...,None],NEG)
opt=torch.optim.AdamW(m.parameters(),lr=2e-3,weight_decay=0.01)
trb=list(batches(P["tr"],64))
for ep in range(3):
    m.train()
    for xc,cl,f,y,ln,mk in trb:
        lg=lg_of(m,xc,cl,f,ln,False)
        loss=F.cross_entropy(lg[mk],y[mk]); opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
m.eval()
with torch.no_grad():
    xc,cl,f,y,ln,mk=next(iter(batches(P["te"],64)))
    pred=lg_of(m,xc,cl,f,ln,False).argmax(-1)
    print("PRED slots:", torch.bincount(pred[mk],minlength=P["max_c"]).tolist())
    print("GOLD slots:", torch.bincount(y[mk],minlength=P["max_c"]).tolist())
    print("frac pred slot0:", round((pred[mk]==0).float().mean().item(),4))
    print("frac GOLD  slot0:", round((y[mk]==0).float().mean().item(),4))
