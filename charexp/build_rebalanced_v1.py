"""Deterministic train-only corpus rebalance, preserving old artifacts.
Uses previously aligned retained rows, not the legacy candidate inventory contents.
Candidate forms are reconstructed only from observed training labels.
"""
from pathlib import Path
import numpy as np,json,hashlib,collections,time,shutil
r=Path(__file__).resolve().parent;out=r/'rebalance_v1';assert not out.exists();out.mkdir()
d=np.load(r/'syl/train.npz');sent=d['sent'];base=d['base'];gold=d['gold'];features=np.load(r/'train_input_features.npy',mmap_mode='r')
off=np.r_[0,np.cumsum(sent,dtype=np.int64)];ntrain=len(sent)-20000;cut=int(off[ntrain])
oldct=np.load(r/'syl/train_ctab.npy');oldv=json.loads((r/'syl/train_vocab.json').read_text());oldb=json.loads((r/'syl/train_bases.json').read_text())
counts=np.bincount(base[:cut].astype(np.int64)*24+gold[:cut],minlength=len(oldct)*24).reshape(-1,24)
total=counts.sum(1);amb=(counts>0).sum(1)>1
# Eligible minority forms must have >=10 examples; typos/singletons are not promoted.
minor=(counts>=10)&(counts<=200)&(counts*4<=total[:,None])&amb[:,None]
# Exclude exact normalized-text matches to any validation/test source, without labels.
blocked=set()
for name in ('va.txt','te.txt','te_news.txt'):
 with (r.parent/'data'/name).open() as f:
  for line in f:blocked.add(hashlib.blake2b(' '.join(line.lower().split()).encode(),digest_size=16).digest())
# Mapping from retained row IDs to original raw IDs, recovered with the exact old encoding.
lookup={b:(i,{oldv[int(v)]:j for j,v in enumerate(oldct[i]) if v>=0}) for i,b in enumerate(['']+oldb) if i}
from sylprep import TOK,strip_diac
raw_ids=[];row=0
with (r.parent/'data/tr.txt').open() as f:
 for rawid,line in enumerate(f):
  ts=TOK.findall(line)
  if not ts or len(ts)>255:continue
  pairs=[]
  for token in ts:
   low=token.lower();e=lookup.get(strip_diac(low))
   if e is None or low not in e[1]:break
   pairs.append((e[0],e[1][low]))
  else:
   n=len(pairs);assert int(sent[row])==n
   lo=int(off[row]);hi=int(off[row+1]);assert all(int(base[lo+j])==b and int(gold[lo+j])==g for j,(b,g) in enumerate(pairs))
   raw_ids.append(rawid);row+=1
assert row==len(sent)
np.save(out/'source_raw_line_ids.npy',np.array(raw_ids,dtype=np.uint32))
# Add original internal validation sentence text to exclusion set, too.
for i in range(ntrain,len(sent)):
 lo,hi=off[i:i+2];text=' '.join(oldv[int(oldct[int(b),int(g)])] for b,g in zip(base[lo:hi],gold[lo:hi]))
 blocked.add(hashlib.blake2b(text.encode(),digest_size=16).digest())
seen=set();template_counts=collections.Counter();chosen=[];rare_chosen=0;exact=excluded=ordinary_dropped=0
for i in range(ntrain):
 lo,hi=off[i:i+2];bs=base[lo:hi];gs=gold[lo:hi]
 text=' '.join(oldv[int(oldct[int(b),int(g)])] for b,g in zip(bs,gs))
 h=hashlib.blake2b(text.encode(),digest_size=16).digest()
 if h in blocked:excluded+=1;continue
 if h in seen:exact+=1;continue
 seen.add(h)
 rare=bool(minor[bs,gs].any())
 # Same stripped sequence cap: retain distinct minority readings, limit common repeats.
 template=hashlib.blake2b(bs.tobytes(),digest_size=16).digest()
 keep=rare or (int.from_bytes(h[:8],'little')%100<40 and template_counts[template]<3)
 if not keep:ordinary_dropped+=1;continue
 chosen.append(i);rare_chosen+=int(rare);template_counts[template]+=1
# Candidate inventory: observed PRE-SELECTION training forms only, no validation/test.
bases=[];vocab=[];vmap={};ctrows=[[-1]*24];remap=np.zeros_like(oldct,dtype=np.uint8);bmap=np.zeros(len(oldct),dtype=np.uint32)
for b in range(1,len(oldct)):
 js=np.flatnonzero(counts[b])
 if not len(js):continue
 bmap[b]=len(bases)+1;bases.append(oldb[b-1]);arr=[-1]*24
 for newj,j in enumerate(sorted(js,key=lambda j:(-int(counts[b,j]),oldv[int(oldct[b,j])]))):
  form=oldv[int(oldct[b,j])]
  if form not in vmap:vmap[form]=len(vocab);vocab.append(form)
  arr[newj]=vmap[form];remap[b,j]=newj
 ctrows.append(arr)
ct=np.array(ctrows,dtype=np.int32)
for pref,indices in [('train',chosen),('validation',range(ntrain,len(sent)))]:
 kept=[];missing=0
 for i in indices:
  lo,hi=off[i:i+2]
  if pref=='validation' and not np.all(counts[base[lo:hi],gold[lo:hi]]>0):missing+=1;continue
  kept.append(i)
 lens=sent[kept];nt=int(lens.sum(dtype=np.int64));bb=np.empty(nt,np.uint32);gg=np.empty(nt,np.uint8);ff=np.lib.format.open_memmap(out/(pref+'_input_features.npy'),mode='w+',dtype=np.uint8,shape=(nt,2));pos=0
 for i in kept:
  lo,hi=off[i:i+2];n=int(hi-lo);bs=base[lo:hi];gs=gold[lo:hi]
  bb[pos:pos+n]=bmap[bs];gg[pos:pos+n]=remap[bs,gs];ff[pos:pos+n]=features[lo:hi];pos+=n
 ff.flush();np.savez_compressed(out/(pref+'.npz'),sent=lens,base=bb,gold=gg)
 np.save(out/(pref+'_ctab.npy'),ct)
 (out/(pref+'_bases.json')).write_text(json.dumps(bases,ensure_ascii=False));(out/(pref+'_vocab.json')).write_text(json.dumps(vocab,ensure_ascii=False))
 np.save(out/(pref+'_source_rows.npy'),np.array(kept,np.uint32))
 print(pref,len(kept),nt,'uncovered_validation_sentences',missing,flush=True)
 if pref=='validation':val_report={'retained':len(kept),'excluded_uncovered':missing,'note':'Validation sentence subset constrained by train-only inventory; selection score is coverage-filtered, raw evaluation reported separately.'}
report={'seed':0,'source_train_rows':ntrain,'selected_rows':len(chosen),'selected_tokens':int(sent[chosen].sum(dtype=np.int64)),'minority_sentences':rare_chosen,'exact_duplicates_removed':exact,'evaluation_exact_matches_removed':excluded,'ordinary_sentences_downsampled':ordinary_dropped,'inventory':'All forms observed in ORIGINAL training rows only, before resampling; no legacy unused candidates, no validation labels. No top-K truncation added: observed forms are a subset of old top24 inventory.','candidate_forms':len(vocab),'bases':len(bases),'selection':'Keep all unique sentences with an ambiguous variant count 10..200 and minority share <=25%; deterministic 40% common keep rate, max3 distinct ordinary outputs per identical stripped sequence. Not a near-duplicate semantic filter.','validation':val_report,'limitations':['Starts from previously retained/covered rows, so previously excluded sentences are not recovered.','Not a pure data-only ablation: removes unobserved legacy candidate forms and changes sampling/schedule.','Rare eligibility is a heuristic, not proof of label quality.','No paraphrase generation or synthetic duplication.']}
(out/'dataset_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
