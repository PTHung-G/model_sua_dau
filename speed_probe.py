import json, time, numpy as np, torch, torch.nn.functional as F
import pilot_balance as pb, exp_dual as ed
cand=json.load(open('data/cand.json'))
forms=sorted(cand); fid={s:i for i,s in enumerate(forms)}
chars=sorted({c for s in forms for c in s}); cid={c:i for i,c in enumerate(chars,1)}
max_c=max(len(v) for v in cand.values()); nchar=len(cid)+2
tr=pb.load_sents('data/tr.txt')[:20000]
for arm in ['local','dual','wide']:
    m=ed.build(arm,nchar,max_c).to(pb.DEV).train()
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=2e-3)
    rng=np.random.default_rng(0)
    gen=pb.make_batches(tr,cand,cid,max_c,64,rng,nchar=nchar,forms=fid)
    t0=time.time(); n=0
    for i in range(9):
        xc,cl,f,y,ln,mk,ok=next(gen)
        lg=pb.logits_of(m,xc,cl,ln)
        amb=mk&(ln>1)&ok; tgt=amb if amb.any() else ok
        ce=F.cross_entropy(lg.reshape(-1,lg.size(-1)),y.reshape(-1),reduction='none').reshape(y.shape)
        loss=ce[tgt].mean(); opt.zero_grad(); loss.backward(); opt.step()
        if i>=2: n+=1
    dt=(time.time()-t0)/n
    print(f"{arm:6s} {dt*1000:6.0f} ms/step  -> 20k steps = {dt*20000/60:5.1f} min", flush=True)
