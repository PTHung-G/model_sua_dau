"""Is more data the cure? Measure how much of the test task is LEARNABLE at all,
as a function of training-set size. If the 'decidable' fraction saturates well
below 100%, adding data cannot solve the rest -- the ceiling is information, not size."""
from collections import Counter
from data import load_sents
from fixbuild import build_fair
from vn import strip_all
te=load_sents("test.txt",3000)
print(f"{'ntr':>7s} {'ambig':>7s} {'order1 seen>=2':>15s} {'order2 seen>=2':>15s} {'order1 in test':>15s}")
for ntr in (10000,30000,60000,150000,300000):
    tr=load_sents("train.txt",ntr)
    pair1=Counter(); pair2=Counter()
    for ws in tr:
        prev=prev2=None
        for w in ws:
            wl=w.lower()
            if prev is not None: pair1[(prev,wl)]+=1
            if prev2 is not None: pair2[(prev2,prev,wl)]+=1
            prev2,prev=prev,wl
    n=o1=o2=0
    for ws in te:
        for i,w in enumerate(ws):
            wl=w.lower()
            n+=1
            if i>0 and pair1.get((ws[i-1].lower(),wl),0)>=2: o1+=1
            if i>1 and pair2.get((ws[i-2].lower(),ws[i-1].lower(),wl),0)>=2: o2+=1
    print(f"{ntr:7d} {n:7d} {o1/n:15.3f} {o2/n:15.3f}", flush=True)
