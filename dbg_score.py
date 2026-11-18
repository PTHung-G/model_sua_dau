import json, torch
import evaluate as ev
from vn import strip_all
cand=json.load(open("data/cand.json"))
names=sorted(cand)
sents=[l.split() for l in open("data/va.txt",encoding="utf-8")][:3000]
chars=sorted({c for s in names for c in s}); c2i={c:i for i,c in enumerate(chars,start=1)}
toks=ev.Tokens(sents,cand,c2i)
print("n_tok reported:",toks.n_tok, "sum rows:",sum(len(r) for r in toks.sents))
# how many tokens does the batch generator actually emit?
tot=0
for xc,cl,f,y,ln,mk in toks.batches(64):
    tot+=int(mk.sum())
print("tokens emitted by batches:",tot)
# count slot0 directly over emitted batches
tot2=ok2=0
for xc,cl,f,y,ln,mk in toks.batches(64):
    tot2+=int(mk.sum()); ok2+=int(((torch.zeros_like(f)==y)&mk).sum())
print(f"slot0 over emitted: {ok2}/{tot2} = {ok2/tot2:.6f}")
# independent recount
cnt=n=0
for ws in sents:
    row=[]
    for w in ws:
        wl=w.lower(); st=strip_all(wl)
        if st in cand:
            cs=cand[st]; row.append(cs.index(wl) if wl in cs else -1)
    if len(row)>=2:
        for g in row:
            n+=1; cnt+=(g==0)
print(f"independent: {cnt}/{n} = {cnt/n:.6f}")
print("diff:", tot2-n, ok2-cnt)
