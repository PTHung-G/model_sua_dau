"""Hypotheses reevaluated with the SKEW FIXED and results reported PER FREQUENCY
BUCKET. Bucketing is on training frequency of the gold accented form, so it is
fair: every system is scored on the same tokens in every bucket.
"""
import json, sys, time, torch, torch.nn as nn, torch.nn.functional as F
from collections import Counter
from data import load_sents
from tiny2 import build
from fixbuild import build_fair
from attn_model import AttnDiac
from attn import BandAttention, FullAttention
torch.set_num_threads(2)
NEG=-1e9

def logits_of(model,xc,cl,f,ln,pri):
    lg=model(xc,cl)+pri[f]
    ar=torch.arange(lg.size(-1),device=lg.device)[None,None,:]
    return lg.masked_fill(ar>=ln[...,None],NEG)

def batched(S,bs,shuffle=False,seed=0):
    idx=list(range(len(S)))
    if shuffle:
        g=torch.Generator().manual_seed(seed); idx=torch.randperm(len(S),generator=g).tolist()
    for i in range(0,len(S),bs):
        chunk=[S[j] for j in idx[i:i+bs]]
        B=len(chunk); T=max(len(r) for r in chunk)
        xc=torch.zeros(B,T,8,dtype=torch.long); cl=torch.zeros(B,T,dtype=torch.long)
        ff=torch.zeros(B,T,dtype=torch.long); y=torch.zeros(B,T,dtype=torch.long)
        ln=torch.ones(B,T,dtype=torch.long); m=torch.zeros(B,T,dtype=torch.bool)
        for b,rows in enumerate(chunk):
            for t,(fi,cs,ci,k) in enumerate(rows):
                xc[b,t,:len(cs)]=torch.tensor(cs); cl[b,t]=len(cs)
                ff[b,t]=fi; y[b,t]=ci; ln[b,t]=k; m[b,t]=True
        yield xc,cl,ff,y,ln,m

def bucket_eval(model,S,pri,buckets):
    model.eval(); tot=Counter(); ok=Counter()
    with torch.no_grad():
        for xc,cl,f,y,ln,m in batched(S,128):
            pred=logits_of(model,xc,cl,f,ln,pri).argmax(-1)
            hit=((pred==y)&m)
            for b in range(xc.size(0)):
                if not m[b].any(): continue
                k=buckets.get((int(f[b,0]),),'20+')
                tot[k]+=m[b].sum().item(); ok[k]+=hit[b].sum().item()
    return {k:(ok[k]/tot[k],tot[k]) for k in ('hard','0','20+') if tot[k]}

def train(model,P,pri,epochs,bs=64,lr=2e-3,wd=0.01):
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd)
    trb=list(batched(P["tr"],bs))
    cut=max(1,len(trb)//20); va=trb[:cut]; trn=trb[cut:]
    def acc(bs_):
        model.eval(); t=o=0
        with torch.no_grad():
            for xc,cl,f,y,ln,m in bs_:
                pr=logits_of(model,xc,cl,f,ln,pri).argmax(-1)
                t+=m.sum().item(); o+=((pr==y)&m).sum().item()
        return o/t
    best=-1; bstate=None
    for e in range(epochs):
        model.train()
        order=torch.randperm(len(trn)).tolist()
        for i in order:
            xc,cl,f,y,ln,m=trn[i]
            lg=logits_of(model,xc,cl,f,ln,pri)
            loss=F.cross_entropy(lg[m],y[m])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        v=acc(va)
        if v>best: best=v; bstate={k:t.detach().clone() for k,t in model.state_dict().items()}
    if bstate: model.load_state_dict(bstate)
    return acc(trb), best

if __name__=="__main__":
    ntr=int(sys.argv[1]) if len(sys.argv)>1 else 60000
    ep=int(sys.argv[2]) if len(sys.argv)>2 else 8
    tr=load_sents("train.txt",ntr); te=load_sents("test.txt",4000)
    P=build_fair(tr,te)
    pri=P["pri"]
    # frequency bucket per gold form
    from collections import Counter as C
    ftr=C(w.lower() for ws in tr for w in ws)
    fof={fi:None for fi in range(len(P["cand"]))}
    forms=sorted(P["cand"]); 
    for fi,s in enumerate(forms):
        top=P["cand"][s][0]
        c=ftr.get(top,0)
        fof[fi]="0" if c==0 else "20+" if c>=20 else "hard"
    buckets=fof
    print(f"ntr={ntr} train={len(P['tr'])} test={len(P['te'])} max_c={P['max_c']}",flush=True)
    res=[]
    for name,kw in [("band8",dict(window=8)),("full",dict(window=0)),("band2",dict(window=2))]:
        torch.manual_seed(0)
        m=AttnDiac(P["nchar"],P["max_c"],syl=32,dim=64,**kw)
        t0=time.time()
        tra,v=train(m,P,pri,ep)
        b=bucket_eval(m,P["te"],pri,buckets)
        npar=sum(p.numel() for p in m.parameters())
        allacc=sum(vv[0]*vv[1] for vv in b.values())/sum(vv[1] for vv in b.values())
        print(f"{name:7s} params={npar:7,} test={allacc:.4f} train={tra:.4f} "
              f"| 20+={b.get('20+',(0,0))[0]:.4f} | hard={b.get('hard',(0,0))[0]:.4f} "
              f"| 0={b.get('0',(0,0))[0]:.4f} ({time.time()-t0:.0f}s)",flush=True)
        res.append(dict(name=name,test=allacc,**{k:v[0] for k,v in b.items()}))
    json.dump(res,open(f"hyp2_{ntr}.json","w"),indent=1)
