from collections import Counter
from data import load_sents
from fixbuild import build_fair
tr=load_sents("train600k.txt",600000); te=load_sents("test.txt",3000)
P=build_fair(tr,te)
pair=Counter()
for ws in tr:
    prev=None
    for w in ws:
        wl=w.lower()
        if prev is not None: pair[(prev,wl)]+=1
        prev=wl
names=sorted(P["cand"]); cand=P["cand"]
n=l=h=0
for row in P["te"]:
    for i,(fi,cs,ci,k) in enumerate(row):
        n+=1
        if i==0: continue
        gold=cand[names[fi]][ci]
        prev=cand[names[row[i-1][0]]][row[i-1][2]]
        if pair.get((prev,gold),0)>=2:
            l+=1
            if ci!=0: h+=1
print(f"train_sents={len(P['tr']):,} forms={len(cand):,} max_c={P['max_c']}")
print(f"test_tokens={n:,} learnable={l/n:.3f} headroom={h/n:.3f} (n={h:,})")
