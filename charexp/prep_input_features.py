"""Build feature sidecar, asserting identical kept rows and token alignment."""
import json, hashlib, time
from pathlib import Path
import numpy as np
from input_features import features, TOK
from sylprep import strip_diac
root=Path(__file__).resolve().parent
prefix=root/'syl/train'
d=np.load(str(prefix)+'.npz');sent=d['sent'];base=d['base'];gold=d['gold']
bases=json.loads(Path(str(prefix)+'_bases.json').read_text())
vocab=json.loads(Path(str(prefix)+'_vocab.json').read_text())
ct=np.load(str(prefix)+'_ctab.npy')
lookup={b:(i,{vocab[int(v)]:j for j,v in enumerate(ct[i]) if v>=0}) for i,b in enumerate(['']+bases) if i}
out=root/'train_input_features.npy'
a=np.lib.format.open_memmap(out,mode='w+',dtype=np.uint8,shape=(len(base),2))
offset=row=0;t0=time.time()
with (root.parent/'data/tr.txt').open(encoding='utf8') as f:
    for raw_id,line in enumerate(f):
        tokens=TOK.findall(line)
        if not tokens or len(tokens)>255: continue
        selected=[]
        for token in tokens:
            low=token.lower(); entry=lookup.get(strip_diac(low))
            if entry is None or low not in entry[1]: break
            selected.append((entry[0],entry[1][low]))
        else:
            n=len(selected)
            assert int(sent[row])==n,(raw_id,row)
            assert all(int(base[offset+j])==b and int(gold[offset+j])==g for j,(b,g) in enumerate(selected)),raw_id
            ff=features(strip_diac(line));assert len(ff)==n
            a[offset:offset+n]=ff
            offset+=n;row+=1
        if raw_id%250000==0: print(raw_id,row,offset,round(time.time()-t0),flush=True)
assert offset==len(base) and row==len(sent),(row,offset)
a.flush()
report={'rows':row,'tokens':offset,'source_sha256':hashlib.sha256((root.parent/'data/tr.txt').read_bytes()).hexdigest(),'features_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'protocol':'Case and preceding boundary from stripped input only; identical retained training rows and base/gold token IDs asserted.'}
(root/'input_features_manifest.json').write_text(json.dumps(report,indent=2))
print(report,flush=True)
