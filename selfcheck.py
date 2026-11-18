"""The 'learnable' subset was defined by 'this exact (prev, gold) pair was seen
>=2x in train'. That definition is biased in favour of order-1 lookup, because on
that subset order-1 simply retrieves the pair it was given. Measure how much of the
accuracy on that subset is definitional rather than learned."""
from collections import Counter
from data import load_sents
from fixbuild import build_fair
tr=load_sents("train.txt",60000); te=load_sents("test.txt",3000)
P=build_fair(tr,te); names=sorted(P["cand"]); cand=P["cand"]
pair=Counter(); ctx=Counter()
for ws in tr:
    prev=None
    for w in ws:
        wl=w.lower()
        if prev is not None:
            pair[(prev,wl)]+=1
            ctx[prev]+=1
        prev=wl
n=hit=amb=amb_hit=0
for row in P["te"]:
    for i,(fi,cs,ci,k) in enumerate(row):
        if i==0: continue
        gold=cand[names[fi]][ci]
        prev=cand[names[row[i-1][0]]][row[i-1][2]]
        if pair.get((prev,gold),0)>=2:
            n+=1
            # is `gold` also the single most common continuation of prev?
            # if yes, "learnable" overlaps with the plain unigram prior again.
        if k>1:
            amb+=1
print(f"tokens with (prev,gold) seen>=2x: {n} of {sum(len(r) for r in P['te'])}")
print(f"ambiguous tokens: {amb}")
# how often does order-1 differ from pick0 ON that subset? that is the real headroom
diff=0
for row in P["te"]:
    for i,(fi,cs,ci,k) in enumerate(row):
        if i==0 or k<2: continue
        gold=cand[names[fi]][ci]
        prev=cand[names[row[i-1][0]]][row[i-1][2]]
        if pair.get((prev,gold),0)>=2 and ci!=0:
            diff+=1
print(f"on learnable subset, tokens whose gold is NOT slot0 (real headroom): {diff}")
