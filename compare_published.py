"""Re-score existing predictions using published Nôm punctuation/case metric."""
import json,re,unicodedata,hashlib
from pathlib import Path
import requests,torch
import eval_factorized_grid as fg
import score_all as sa

root=Path('published_comparison');root.mkdir(exist_ok=True)
base='https://raw.githubusercontent.com/nrl-ai/nom-vn/main/'
s=requests.Session()
def fetch(path):
 r=s.get(base+path,timeout=60);r.raise_for_status();(root/Path(path).name).write_text(r.text);return r.text
script=fetch('benchmarks/accuracy/bench_diacritic_hf_udvtb.py')
# Extract only the inspected normalization definitions, not arbitrary remote code.
def norm(t):
 t=unicodedata.normalize('NFC',t)
 t=re.sub(r'\s+([,.;:!?\)\]\}\"\'»…])',r'\1',t)
 t=re.sub(r'([\(\[\{\"\'«])\s+',r'\1',t)
 return re.sub(r'\s+',' ',t).strip()
torch.set_num_threads(4)
c=torch.load('kaggle_factorized_full_output/factorized_full_results/factorized_best.pt',weights_only=False,map_location='cpu');m=fg.model_from_checkpoint(c)
r={'our_model':{},'published':{},'limitations':['Published third-party scores, not independently rerun APIs.','No flagship GPT/Claude/Gemini results established by this source.','HF grid used locally; upstream raw-corpus/version equality not verified.','LLM nonbusiness scores are sentence-macro accuracy; our old 90.43% is token-micro normalized-core and not directly comparable.']}
for tag,path,note in fg.GRID:
 pairs=sa.load_pairs(path);pred=fg.predict(m,[x for x,_ in pairs],c)
 total=correct=exact=0;macro=[]
 for (_,g),p in zip(pairs,pred):
  g=norm(g);p=norm(' '.join(p));gs=g.split();ps=p.split();ok=sum(a==b for a,b in zip(gs,ps))
  total+=len(gs);correct+=ok;macro.append(ok/len(gs) if gs else float(not ps));exact+=int(g==p)
 r['our_model'][tag]={'sentences':len(pairs),'word_micro':correct/total,'word_macro':sum(macro)/len(macro),'sentence_exact':exact/len(pairs),'normalized_target_sha256':hashlib.sha256('\n'.join(norm(g) for _,g in pairs).encode()).hexdigest()}
 print('OURS',tag,r['our_model'][tag],flush=True)
for model,prefix,biz in [('gpt-4o-mini','gpt4omini',None),('gpt-5.4-mini','gpt54mini','gpt_5_4_mini'),('gpt-5.4-nano','gpt54nano','gpt_5_4_nano'),('claude-haiku-4-5','haiku45','claude_haiku_4_5')]:
 r['published'][model]={}
 for tag,suffix in [('conv','tatoeba'),('formal','udhr'),('literary','udvtb')]:
  path=f'benchmarks/results/baseline_diacritic_{prefix}_{suffix}.json';data=json.loads(fetch(path));r['published'][model][tag]={'source':base+path,'data':data};print(model,tag,data,flush=True)
 if biz:
  path=f'benchmarks/results/baseline_diacritic_{biz}.json';r['published'][model]['business']={'source':base+path,'data':json.loads(fetch(path))}
(root/'results.json').write_text(json.dumps(r,ensure_ascii=False,indent=2))
print('COMPLETE',flush=True)
