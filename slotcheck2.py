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
w0=m.out.weight.detach().clone()
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
print("out.weight moved:", (m.out.weight.detach()-w0).abs().mean().item())
print("out.weight rows (first 3, first 5 dims):")
for r in range(3):
    print("  ", [round(v,3) for v in m.out.weight.detach()[r,:5].tolist()])
print("row norms:", [round(v,4) for v in m.out.weight.detach().norm(dim=1).tolist()[:6]])
# key check: is the encoder output nearly CONSTANT across positions?
m.eval()
with torch.no_grad():
    xc,cl,f,y,ln,mk=next(iter(batches(P["te"],32)))
    h=m.norm(m.blocks[0](m.proj(m.encode_syllables(xc,cl)))[0][0])
    v = h.reshape(-1,h.size(-1))
    print("hidden std across positions:", round(v.std(0).mean().item(),5))
    print("hidden std across batch:     ", round(v.std(1).mean().item(),5))
