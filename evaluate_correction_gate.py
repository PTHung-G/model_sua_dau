"""One-shot fresh evaluation of the dev-selected correction gate."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from correction_policies import metrics
from train_correction_gate import gate_predict
from analyze_correction import bootstrap

ROOT=Path(__file__).resolve().parent

def main():
    out=ROOT/'correction_gate'
    assert not (out/'results.json').exists(),'Evaluation already recorded; no retuning'
    d=dict(np.load(out/'evaluation_scores.npz'))
    ckpt=torch.load(out/'gate_best.pt',weights_only=False)
    latest=torch.load(out/'gate_latest.pt',weights_only=False)
    assert latest['epoch']==200
    samples=json.loads((out/'samples.json').read_text())['evaluation']
    old=json.loads((ROOT/'correction_calibration/samples.json').read_text())
    from correction_samples import group
    new_groups={group(e['words']) for e in samples}
    assert len(new_groups)==1500
    assert not new_groups&{group(e['words']) for s in ('calibration','evaluation') for e in old[s]}
    base=d['base'].argmax(-1);raw=d['rerank'].argmax(-1);gated=gate_predict(d,ckpt)
    result=dict(best_epoch=ckpt['epoch'],latest_epoch=latest['epoch'],
                training_examples=ckpt['training_examples'],
                best_sha256=hashlib.file_digest((out/'gate_best.pt').open('rb'),'sha256').hexdigest(),
                gate_rejections=int(((raw!=base)&(gated==base)).sum()),policies={})
    # Shared bootstrap helper has fixed seed2402; gate protocol requires2404.
    sid=d['sentence'];n=1500;counts=np.bincount(sid,minlength=n);gold=d['gold']
    for name,p in [('generator',base),('raw_hybrid',raw),('gate',gated)]:
        gain=np.bincount(sid,weights=(p==gold).astype(float)-(raw==gold),minlength=n)
        rng=np.random.default_rng(2404);draws=[]
        for _ in range(100):
            ix=rng.integers(0,n,(100,n));draws.extend(gain[ix].sum(1)/counts[ix].sum(1)*100)
        result['policies'][name]=dict(metrics=metrics(d,p,n),vs_raw_hybrid=dict(
            delta_pp=float(gain.sum()/counts.sum()*100),ci95_pp=np.quantile(draws,[.025,.975]).tolist()))
    (out/'results.json').write_text(json.dumps(result,indent=2))
    lines=['# Learned correction gate feasibility results','',
           f'Fit:384 sentences, dev:128 sentences, decisive fit changes:{ckpt["training_examples"]}. Best epoch:{ckpt["epoch"]}; latest saved epoch:200. Fresh evaluation:1500 sentences.','',
           '| Policy | Token accuracy | Sentence exact | Fixed/broken vs generator | Δ token pp vs raw (95% descriptive CI) |',
           '|---|---:|---:|---:|---|']
    for name,r in result['policies'].items():
        m=r['metrics'];c=r['vs_raw_hybrid'];lines.append(f'| {name} | {m["token_accuracy"]*100:.4f}% | {m["sentence_exact"]*100:.3f}% | {m["fixed_tokens"]}/{m["broken_tokens"]} | {c["delta_pp"]:.4f} [{c["ci95_pp"][0]:.4f},{c["ci95_pp"][1]:.4f}] |')
    lines+=['',f'Gate rejected {result["gate_rejections"]} raw corrections. Both best/latest strict-loaded; original model weights unchanged.',
            '', 'Small same-corpus feasibility experiment, very few supervised change examples, development-set checkpoint selection. No claim of independent-test generalization. No retuning on this evaluation sample.']
    (out/'GATE_RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
