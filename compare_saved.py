"""Practical saved-checkpoint comparison; not a causal architecture ablation."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import pilot_arch as pa
import pilot_balance as pb
import score_all as sa
import eval_factorized_grid as fg
from attn_model import AttnDiac
from vn import strip_all

ROOT=Path('saved_comparison');ROOT.mkdir(exist_ok=True)
torch.set_num_threads(4)
legacy_path=Path('ckpt_v2_m1m.pt')
new_path=Path('kaggle_factorized_full_output/factorized_full_results/factorized_best.pt')
lc=torch.load(legacy_path,map_location='cpu',weights_only=False)
nc=torch.load(new_path,map_location='cpu',weights_only=False)
assert lc['vocab']=={s:i for i,s in enumerate(nc['forms'])}, 'Stem/form indices differ'
assert json.loads(Path('data/cand.json').read_text())==nc['cand'], 'Legacy corpus candidate inventory differs'
# Legacy checkpoint stores stem IDs, not per-stem candidates; use matching original v2 corpus inventory.

assert lc['char2id']==nc['char2id']
lm=AttnDiac(lc['nchar'],lc['max_c'],window=lc['window'],out_std=lc['out_std'],syl=192,dim=384,blocks=1,packed_chars=False).eval()
lm.load_state_dict(lc['model'],strict=True)
nm=fg.model_from_checkpoint(nc)

@torch.inference_mode()
def predict(model,inputs,bs,legacy=False):
    result=[]
    ids={s:i for i,s in enumerate(nc['forms'])}
    cand=nc['cand']
    for start in range(0,len(inputs),bs):
        toks=[fg.tokenize(s) for s in inputs[start:start+bs]]
        rows=[[(i,strip_all(sa.normalized_core(w))) for i,w in enumerate(ts) if strip_all(sa.normalized_core(w)) in cand] for ts in toks]
        T=max(1,max(map(len,rows)));L=max([1]+[min(64,len(s)) for row in rows for _,s in row])
        B=len(toks)
        x=torch.full((B,T,L),nc['nchar']-1,dtype=torch.long)
        cl=torch.ones((B,T),dtype=torch.long);f=torch.zeros_like(cl);ln=torch.ones_like(cl);mask=torch.zeros((B,T),dtype=torch.bool)
        for b,row in enumerate(rows):
            for j,(_,s) in enumerate(row):
                chars=[nc['char2id'].get(c,0) for c in s[:64]]
                x[b,j,:len(chars)]=torch.tensor(chars);cl[b,j]=len(chars);f[b,j]=ids[s];ln[b,j]=len(cand[s]);mask[b,j]=True
        if legacy:
            logits=pb.logits_of(model,x,cl,ln)
        else:
            logits=pa.candidate_logits(model,'factorized',x,cl,f,ln,mask,(nc['tone_table'],nc['mod_table']))
        picks=logits.argmax(-1).tolist()
        for b,row in enumerate(rows):
            for j,(pos,s) in enumerate(row):toks[b][pos]=sa.restore_token(toks[b][pos],cand[s][picks[b][j]])
        result.extend(toks)
    return result

def per_sentence(gold,preds):
    ans=[]
    for g,p in zip(gold,preds):
        good=n=0
        for j,w in enumerate(g.split()):
            if sa.is_scoring_token(w):
                n+=1;good+=int(j<len(p) and sa.normalized_core(w)==sa.normalized_core(p[j]))
        ans.append((good,n))
    return np.array(ans,dtype=np.int64)

def bootstrap(a,b):
    rng=np.random.default_rng(941);vals=[]
    for _ in range(2000):
        idx=rng.integers(0,len(a),len(a));den=a[idx,1].sum()
        vals.append(float((a[idx,0].sum()-b[idx,0].sum())/max(1,den)))
    return (np.quantile(vals,[.025,.975])*100).tolist()

report={'protocol':{'purpose':'practical comparison of existing weights, not causal factorization claim','primary_batch':1,'sensitivity_batches':[32,128],'local_sampling':'2000 sentences reservoir seed 2718, shared inputs; no train','normalization':fg.PROTOCOL,'legacy':'original unpacked encoding and original unmasked padding via pb.logits_of; never silently corrected','bootstrap':'2000 paired sentence resamples; fixed-checkpoint sampling uncertainty, not training-seed uncertainty'},'checkpoints':{},'sets':{}}
for name,p in [('flat',legacy_path),('factorized',new_path)]:report['checkpoints'][name]={'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'step':120000}
for tag,path,note in sa.TESTSETS:
    if path.startswith('GRID:'):pairs=sa.load_pairs(path);manifest={'source':path,'count':len(pairs)}
    else:
        lines,manifest=pa.reservoir(Path(path),2000,2718)
        pairs=[(strip_all(' '.join(s)), ' '.join(s)) for s in lines]
    inp=[x for x,_ in pairs];gold=[y for _,y in pairs]
    record={'manifest':manifest,'note':note,'models':{}}
    counts={};primary={}
    for name,model in [('flat',lm),('factorized',nm)]:
        rec={}
        for bs in (1,32,128):
            t=time.perf_counter();pred=predict(model,inp,bs,legacy=name=='flat')
            scores=fg.score_predictions(gold,pred,nc['cand'])
            if bs==1:primary[name]=pred;counts[name]=per_sentence(gold,pred)
            disagree=sum(sa.normalized_core(a)!=sa.normalized_core(b) for base,other in zip(primary[name],pred) for a,b in zip(base,other) if sa.is_scoring_token(a))
            rec[str(bs)]={'scores':scores,'seconds':time.perf_counter()-t,'changed_letter_tokens_vs_bs1':disagree}
            print(tag,name,'bs',bs,'ALL',round(100*scores['all_syll_acc'],4),'changed',disagree,flush=True)
        record['models'][name]=rec
    delta=(counts['factorized'][:,0].sum()-counts['flat'][:,0].sum())/counts['flat'][:,1].sum()*100
    record['factorized_minus_flat_pp_bs1']=float(delta)
    record['paired_sentence_bootstrap_95ci_pp']=bootstrap(counts['factorized'],counts['flat'])
    record['n_scoring']=int(counts['flat'][:,1].sum())
    report['sets'][tag]=record
    (ROOT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print('DELTA',tag,delta,record['paired_sentence_bootstrap_95ci_pp'],flush=True)
report['grid_mean']={name:{str(bs):sum(report['sets'][tag]['models'][name][str(bs)]['scores']['all_syll_acc'] for tag in ('business','formal','conv','literary'))/4 for bs in (1,32,128)} for name in ('flat','factorized')}
report['status']='complete'
(ROOT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print('DONE',report['grid_mean'],flush=True)
