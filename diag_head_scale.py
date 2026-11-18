"""Is the failure an output bottleneck rather than 'model too small'?

Diagnostic: keep the SAME architecture and data, but scale the output head so its
logits (currently ~0.3) are comparable to the learned attention bias (~0.76).
If accuracy jumps -> the encoder already carried the signal and the head was
squeezing it. If not -> the architecture genuinely lacks the information.
"""
import torch, torch.nn as nn, torch.nn.functional as F, sys
from data import load_sents
from fixbuild import build_fair
from scoring import batches, score_predictor, gold_freq_buckets, fmt_row
from attn_model import AttnDiac
torch.set_num_threads(2)
NEG=-1e9
DEV="cpu"
tr=load_sents("train.txt",int(sys.argv[1]) if len(sys.argv)>1 else 15000)
te=load_sents("test.txt",2000)
P=build_fair(tr,te); pri=P["pri"]
buckets=gold_freq_buckets(tr,P["cand"])
def lg_of(model,xc,cl,f,ln,use_prior):
    lg=model(xc,cl)
    if use_prior: lg=lg+pri[f]
    ar=torch.arange(lg.size(-1))[None,None,:]
    return lg.masked_fill(ar>=ln[...,None],NEG)
class Pick0:
    def __call__(self,xc,cl,f,ln): return torch.zeros_like(f)
base=score_predictor(Pick0(),P["te"],buckets)
print(fmt_row("pick0",base), flush=True)
for init_std, lr, tag in [(1e-2,2e-3,"std=1e-2 (current)"),
                          (1e-1,2e-3,"std=1e-1"),
                          (3e-1,2e-3,"std=3e-1"),
                          (1e-1,1e-2,"std=1e-1 lr=1e-2")]:
    torch.manual_seed(0)
    m=AttnDiac(P["nchar"],P["max_c"],syl=32,dim=64,window=8)
    nn.init.normal_(m.out.weight,std=init_std); nn.init.zeros_(m.out.bias)
    opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=0.01)
    trb=list(batches(P["tr"],64))
    for ep in range(6):
        m.train()
        for xc,cl,f,y,ln,mk in trb:
            lg=lg_of(m,xc,cl,f,ln,False)
            loss=F.cross_entropy(lg[mk],y[mk]); opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    fn=lambda xc,cl,f,ln: lg_of(m,xc,cl,f,ln,False).argmax(-1)
    r=score_predictor(fn,P["te"],buckets)
    with torch.no_grad():
        xc,cl,f,y,ln,mk=next(iter(batches(P["te"],32)))
        pr=fn(xc,cl,f,ln)
        f0=(pr[mk]==0).float().mean().item()
        wn=m.out.weight.norm(dim=1).mean().item()
    d=r["ALL"][0]-base["ALL"][0]
    print(fmt_row(tag,r)+f"  delta={d:+.4f} pred0={f0:.3f} headnorm={wn:.3f}", flush=True)
