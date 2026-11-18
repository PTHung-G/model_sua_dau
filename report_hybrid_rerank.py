"""Summarize downloaded CONTROL/HYBRID results; never infer missing runs."""
import argparse
import json
from pathlib import Path
import numpy as np


def compare(a,b):
    a=np.array(a)==0;b=np.array(b)==0
    if a.shape!=b.shape:raise ValueError('Unpaired sentence arrays')
    delta=b.astype(float)-a.astype(float)
    rng=np.random.default_rng(20261023)
    draws=np.concatenate([delta[rng.integers(0,len(delta),(1000,len(delta)))].mean(1) for _ in range(20)])
    return dict(delta_pp=float(delta.mean()*100),ci_pp=(np.quantile(draws,[.025,.975])*100).tolist(),
                rescued=int((~a&b).sum()),broken=int((a&~b).sum()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(Path(__file__).resolve().parent/'kaggle_hybrid_rerank_output'))
    args=p.parse_args();root=Path(args.root)
    rows=[];records={};summary={}
    for arm in ('CONTROL','HYBRID'):
        folder=root/f'hybrid_rerank_{arm.lower()}_results'
        result=json.loads((folder/'results.json').read_text())
        verification=json.loads((folder/'verification.json').read_text())
        record=result['arms'][f'hybrid_rerank_{arm}'];records[arm]=record
        assert result['status']=='complete' and record['budget_complete'], 'Incomplete training run'
        assert verification['latest']['step']==record['consumed_batches']
        assert verification['best']['step']==record['best']['step']
        for kind in ('best','latest'):
            assert verification[kind]['strict_load'] and verification[kind]['frozen_base_equal'] and verification[kind]['finite']
            assert (folder/f'hybrid_rerank_{kind}.pt').is_file()
        metric=record['final_best'];base=result['base_init']
        paired=compare(base['sentence_errors'],metric['sentence_errors'])
        counts=verification['best']['counts']
        summary[arm]=dict(status=result['status'],best_step=record['best']['step'],
                          consumed_batches=record['consumed_batches'],visited_rows=record['visited_source_rows'],
                          best_correct=sum(x==0 for x in metric['sentence_errors']),
                          base_correct=sum(x==0 for x in base['sentence_errors']),
                          token_accuracy=metric['token_accuracy'],counts=counts,paired=paired,
                          final_correct=sum(x==0 for x in record['final']['sentence_errors']),
                          final_token_accuracy=record['final']['token_accuracy'],
                          train_seconds=record['train_seconds'],skipped_batches=record['skipped_batches'])
        rows.append(f"| {arm} | {counts['trainable']:,} | {record['best']['step']} | {summary[arm]['best_correct']} | {paired['rescued']} / {paired['broken']} | {metric['token_accuracy']*100:.3f}% |")
    # Validate pairing using source-sample hashes, not sentence count alone.
    provenance=[json.loads((root/f'hybrid_rerank_{a.lower()}_results/results.json').read_text())['provenance'] for a in ('CONTROL','HYBRID')]
    assert provenance[0]['validation']==provenance[1]['validation'],'Validation manifests differ'
    assert provenance[0]['train']['planned_prefix_sha256']==provenance[1]['train']['planned_prefix_sha256']
    summary['hybrid_vs_control_best']=compare(records['CONTROL']['final_best']['sentence_errors'],records['HYBRID']['final_best']['sentence_errors'])
    assert len(records['CONTROL']['evaluation_log'])==len(records['HYBRID']['evaluation_log'])
    assert summary['CONTROL']['counts']==summary['HYBRID']['counts'], 'Capacity mismatch'
    assert records['CONTROL']['visited_source_rows']==records['HYBRID']['visited_source_rows'], 'Unequal consumed data'
    aligned=[]
    for x,y in zip(records['CONTROL']['evaluation_log'],records['HYBRID']['evaluation_log']):
        assert x['step']==y['step']
        aligned.append(dict(step=x['step'],**compare(x['validation']['sentence_errors'],y['validation']['sentence_errors'])))
    summary['equal_step_comparisons']=aligned
    text='# Higher-capacity hybrid reranker results\n\nExploratory reused-validation results, not an independent test. Both arms have matched capacity and training prefixes.\n\n'
    text+='| Arm | Trainable params | Best step | Correct sentences | Rescued / broken vs generator | Token accuracy |\n|---|---:|---:|---:|---:|---:|\n'+'\n'.join(rows)+'\n\n'
    text+='Best-checkpoint HYBRID vs CONTROL: '+json.dumps(summary['hybrid_vs_control_best'])+'\n\n'
    text+='Confidence intervals are descriptive paired sentence bootstrap intervals (20,000 draws); they do not account for validation reuse, checkpoint selection, or training-seed variation.\n\n'
    text+='Architectural fixes, capacity and budget all differ from older A/B/C; comparisons with those runs cannot isolate the cause of any gain. Unknown stems remain in the scoring denominator but not model context.\n'
    (root/'HYBRID_RERANK_RESULTS.md').write_text(text)
    (root/'hybrid_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
