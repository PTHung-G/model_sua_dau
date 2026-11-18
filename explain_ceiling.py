"""Work out the ALL-accuracy ceiling from first principles, on the real numbers."""
n_all      = 32144*1.0        # placeholder, replaced below
from data import load_sents
from fixbuild import build_fair
from collections import Counter
tr=load_sents("train600k.txt",600000); te=load_sents("test.txt",20000)
P=build_fair(tr,te)
names=sorted(P["cand"]); cand=P["cand"]
pair=Counter()
for ws in tr:
    prev=None
    for w in ws:
        wl=w.lower()
        if prev is not None: pair[(prev,wl)]+=1
        prev=wl
n=0; slot0=0; hr=0; hr_easy=0
for row in P["te"]:
    for i,(fi,cs,ci,k) in enumerate(row):
        n+=1
        if ci==0: slot0+=1          # gold IS the most frequent form -> pick0 right
        else:
            hr+=1                    # gold is NOT slot0 -> pick0 always wrong
print(f"test tokens                     : {n:,}")
print(f"gold == slot0 (pick0 correct)   : {slot0:,}  = {slot0/n:.4f}")
print(f"gold != slot0 (pick0 WRONG)     : {hr:,}  = {hr/n:.4f}   <-- HEADROOM")
print()
print(f"pick0 accuracy  = {slot0/n:.4f}   (it is right exactly on the slot0 tokens)")
print(f"PERFECT model   = ({slot0}+{hr})/{n} = 1.0000")
print()
print("So the ceiling for ANY system on ALL is 100% in principle,")
print("but the only tokens where a system can possibly differ from pick0")
print(f"are the {hr:,} headroom tokens = {hr/n:.4f} of all tokens.")
print()
print(f"If a model fixes a fraction f of headroom, then")
print(f"   ALL = {slot0/n:.4f} + f * {hr/n:.4f}")
for f in (0.0,0.5,0.9188,0.9258,1.0):
    print(f"   f={f:.4f} -> ALL={slot0/n + f*hr/n:.4f}")
