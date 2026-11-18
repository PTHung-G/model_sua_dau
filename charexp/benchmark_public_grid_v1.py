"""Frozen external-input diagnostic. Not an official leaderboard submission."""
import sys,json,hashlib,re,unicodedata,types,time
from pathlib import Path
import torch
from datasets import load_dataset
r=Path(__file__).resolve().parent
sys.path.insert(0,str(r))
import sylmodel_attention as sm
sys.modules['sylmodel']=sm
import syltrain_features as st
sys.modules['syltrain']=st
from syleval_raw import restore_case
src=(r/'syleval_features.py').read_text()
src=src[:src.index("if __name__ == '__main__':")]
src=src.replace("    raw_rows = []","    raw_rows = []\n    predictions = {}")
a=src.index("    with open(path, encoding='utf-8') as stream:");b=src.index('    def encode(strings):')
block=src[a:b]
block=block.replace("    with open(path, encoding='utf-8') as stream:\n        for line_id, line in enumerate(stream):", "    if True:\n        for line_id, (line, target) in enumerate(PAIRS):")
block=block.replace('gold = TOK.findall(line)','gold = TOK.findall(target)')
block=block.replace('source = [strip_diac(t) for t in gold]','source = TOK.findall(strip_diac(line))\n            assert len(source) == len(gold), (line_id, "nonaligned benchmark pair")')
src=src[:a]+block+src[b:]
src=src.replace('            for j, (_, gold, source, ids) in enumerate(batch):','            for j, (line_id, gold, source, ids) in enumerate(batch):\n                restored_tokens = []')
src=src.replace('                    hit = restored == gold[t]','                    restored_tokens.append(restored)\n                    hit = restored == gold[t]')
src=src.replace("    finally:","                it = iter(restored_tokens)\n                predictions[line_id] = TOK.sub(lambda m: next(it), PAIRS[line_id][0])\n    finally:")
src=src.replace('    return totals','    return totals, predictions')
mod=types.ModuleType('frozen_external_eval');exec(compile(src,'external_eval','exec'),mod.__dict__)
def norm(s):
 s=unicodedata.normalize('NFC',s)
 s=re.sub(r'\s+([,.;:!?\)\]\}\"\'»…])',r'\1',s)
 s=re.sub(r'([\(\[\{\"\'«])\s+',r'\1',s)
 return re.sub(r'\s+',' ',s).strip()
torch.set_num_threads(2)
ck=torch.load(r/'syl_rebalance_output_v1/rebalance_latest.pt.best.pt',map_location='cpu',weights_only=False)
n,b,c,ci,_=st.build_tensors(str(r/'rebalance_v1/train'));cfg=ck['config']
model=sm.SylCRF(n,b,c,ci,d=cfg['d'],hidden=cfg['hidden'],r=cfg['r'],layers=cfg['layers']);model.load_state_dict(ck['model']);model.eval()
out=r/'public_grid_rebalance_v1';out.mkdir(exist_ok=False);results={}
for name in ('business_55','formal_72','conversational_300','literary_800'):
 ds=load_dataset('nrl-ai/vn-diacritic-eval',name,split='train');pairs=[(x['input'],x['target']) for x in ds]
 (out/(name+'.pairs.json')).write_text(json.dumps(pairs,ensure_ascii=False))
 mod.PAIRS=pairs
 stats,preds=mod.evaluate_raw(model,str(r/'rebalance_v1/train'),out/(name+'.pairs.json'),0,bs=16)
 rows=[];correct=total=exact=0;mac=[]
 for i,(inp,gold) in enumerate(pairs):
  pred=preds.get(i,inp);g=norm(gold);p=norm(pred);gt=g.split();pt=p.split();ok=sum(x==y for x,y in zip(gt,pt))
  correct+=ok;total+=len(gt);mac.append(ok/len(gt) if gt else float(not pt));exact+=int(g==p)
  rows.append({'input':inp,'target':gold,'prediction':pred})
 stats.update(normalized_word_micro=correct/total,normalized_word_macro=sum(mac)/len(mac),sentence_exact=exact/len(pairs),normalized_gold_tokens=total,normalized_gold_sha256=hashlib.sha256('\n'.join(norm(g) for _,g in pairs).encode()).hexdigest())
 results[name]=stats
 (out/(name+'.predictions.json')).write_text(json.dumps(rows,ensure_ascii=False))
 (out/'results.json').write_text(json.dumps(results,indent=2));print(name,stats,flush=True)
