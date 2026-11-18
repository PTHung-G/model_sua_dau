"""Lock calibration selection before loading evaluation arrays."""
import hashlib
import json
from pathlib import Path
import numpy as np
from correction_policies import predict,select,metrics,ALPHAS,THRESHOLDS
ROOT=Path(__file__).resolve().parent/'correction_calibration'


def bootstrap(data,a,b,n):
    sid=data['sentence'];gold=data['gold']
    improvement=np.bincount(sid,weights=(b==gold).astype(float)-(a==gold),minlength=n)
    counts=np.bincount(sid,minlength=n)
    rng=np.random.default_rng(2402)
    draws=[]
    for _ in range(100):
        ix=rng.integers(0,n,(100,n))
        draws.extend(improvement[ix].sum(1)/counts[ix].sum(1)*100)
    return dict(delta_pp=float(improvement.sum()/counts.sum()*100),
                ci95_pp=np.quantile(draws,[.025,.975]).tolist())


def main():
    selection_path=ROOT/'selected_policy.json'
    cal=dict(np.load(ROOT/'calibration_scores.npz'))
    if not selection_path.exists():
        selected={kind:select(cal,kind) for kind in ('alpha','threshold')}
        grids={kind:[dict(value=x,**metrics(cal,predict(cal,kind,x),512)) for x in grid]
               for kind,grid in [('alpha',ALPHAS),('threshold',THRESHOLDS)]}
        selection_path.write_text(json.dumps(dict(selected=selected,calibration_grid=grids,
            calibration_sha256=hashlib.file_digest((ROOT/'calibration_scores.npz').open('rb'),'sha256').hexdigest()),indent=2))
    locked=json.loads(selection_path.read_text());selected=locked['selected']
    # Evaluation labels are first opened only after selected_policy.json is written.
    data=dict(np.load(ROOT/'evaluation_scores.npz'));tokens=json.loads((ROOT/'evaluation_tokens.json').read_text())
    policies={'generator':('alpha',0.),'raw_hybrid':('alpha',1.),
              'selected_alpha':('alpha',selected['alpha']),'selected_threshold':('threshold',selected['threshold'])}
    base=predict(data,'alpha',0);raw=predict(data,'alpha',1);gold=data['gold']
    result=dict(selection=selected,selection_sha256=hashlib.file_digest(selection_path.open('rb'),'sha256').hexdigest(),
                evaluation_sentences=1500,scored_tokens=len(gold),
                unknown_tokens=sum(not t['known'] for t in tokens),
                gold_absent_tokens=int((gold<0).sum()),policies={})
    for name,(kind,value) in policies.items():
        p=predict(data,kind,value)
        result['policies'][name]=dict(metrics=metrics(data,p,1500),
                                      vs_generator=bootstrap(data,base,p,1500),
                                      vs_raw_hybrid=bootstrap(data,raw,p,1500))
    probs=np.exp(data['base']-data['base'].max(-1,keepdims=True));probs/=probs.sum(-1,keepdims=True)
    confidence=probs.max(-1)
    result['raw_hybrid_strata']={}
    for field,arr,bins in [('base_confidence',confidence,[0,.5,.8,.95,1.00001]),
                           ('candidate_count',data['counts'],[1,2,6,1000]),
                           ('sentence_length',data['length'],[0,11,21,41,1000000])]:
        result['raw_hybrid_strata'][field]=[]
        for low,high in zip(bins,bins[1:]):
            m=(arr>=low)&(arr<high);changed=(base!=raw)&m
            result['raw_hybrid_strata'][field].append(dict(low=low,high=high,tokens=int(m.sum()),
                changed=int(changed.sum()),fixed=int((m&(base!=gold)&(raw==gold)).sum()),
                broken=int((m&(base==gold)&(raw!=gold)).sum())))
    examples={}
    samples=json.loads((ROOT/'samples.json').read_text())['evaluation']
    for name,condition in [('fixed',(base!=gold)&(raw==gold)),('broken',(base==gold)&(raw!=gold))]:
        examples[name]=[]
        for i in np.flatnonzero(condition)[:15]:
            t=tokens[int(i)]
            examples[name].append(dict(sentence=' '.join(samples[t['sentence']]['words']),
                                       target=t['target'],generator=t['options'][base[i]],
                                       hybrid=t['options'][raw[i]]))
    (ROOT/'case_examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2))
    (ROOT/'results.json').write_text(json.dumps(result,indent=2))
    lines=['# Conservative correction feasibility study','',
           'Policies selected only on 512 calibration sentences, evaluated on 1500 separate sentences. Same-corpus validation; limited statistical power; historical exposure audit is not exhaustive. No weights modified.','',
           f'Selected alpha: {selected["alpha"]}; selected probability-gap threshold: {selected["threshold"]}.','',
           '| Policy | Token accuracy | Sentence exact | Fixed / broken tokens | Δ token pp vs raw (95% descriptive CI) |',
           '|---|---:|---:|---:|---|']
    for name,r in result['policies'].items():
        m=r['metrics'];c=r['vs_raw_hybrid']
        lines.append(f'| {name} | {m["token_accuracy"]*100:.4f}% | {m["sentence_exact"]*100:.3f}% | {m["fixed_tokens"]} / {m["broken_tokens"]} | {c["delta_pp"]:.4f} [{c["ci95_pp"][0]:.4f}, {c["ci95_pp"][1]:.4f}] |')
    lines.extend(['','Confidence intervals use 10000 paired sentence-bootstrap resamples. Unknown and missing-gold tokens remain in denominators. Secondary strata/examples are descriptive, not grounds to retune on this evaluation set.'])
    (ROOT/'CORRECTION_RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
