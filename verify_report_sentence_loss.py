"""Verify loss-pilot artifacts and summarize fixed-budget comparisons."""
import argparse,json
from pathlib import Path
import torch
from verify_factorized_full import verify

def main(directory):
 root=Path(directory);r=json.loads((root/'results.json').read_text());assert r['status']=='complete'
 torch.set_num_threads(4)
 proofs=[]
 for arm in ('token_ce','sentence_hard'):
  for kind in ('latest','best'):
   path=root/f'{arm}_{kind}.pt';verify(path)
   p=torch.load(path,weights_only=False,map_location='cpu')
   assert p['experiment_arm']==arm
   expected=r['config']['steps'] if kind=='latest' else r['arms'][arm]['best']['step']
   assert p['step']==expected
   assert p['params']==1218679
   for key in ('train','validation'):
    assert p['provenance'][key]['sample_sha256']==r['provenance'][key]['sample_sha256']
   proofs.append(json.loads(Path(str(path)+'.verified.json').read_text()))
 lines=['# Sentence-aware loss fine-tuning pilot — verified results','',
 'Same original factorized120k checkpoint; no architecture change. Same100k train /3k validation,3000 updates, batch64, AdamW1e-4 cosine, seed0. Sentence-hard surrogate: 50% original token CE +25% sentence-balanced mean CE +25% sentence-balanced hardest-quarter CE. All token masks match baseline; no discrete sentence-error pseudo-gradient.','',
 'Primary sentence exact ignores case/punctuation after shared normalization. Unknown letter tokens remain in scoring. Validation only; one seed, no public/test selection.','',
 '| Arm | Step | Sentence exact | Token accuracy | Train seconds |','|---|---:|---:|---:|---:|']
 base=r['base_init']
 lines.append(f"| Original | 0 | {base['sentence_exact']*100:.3f}% | {base['token_accuracy']*100:.3f}% | — |")
 for arm,v in r['arms'].items():
  for kind,step,m in [('final',v['step'],v['final']),('best',v['best']['step'],v['best']['validation'])]:
   lines.append(f"| {arm} {kind} | {step} | {m['sentence_exact']*100:.3f}% | {m['token_accuracy']*100:.3f}% | {v['train_seconds']:.1f} |")
 lines+=['','## Learning curves','', '| Arm | Step | Sentence exact | Token accuracy |','|---|---:|---:|---:|']
 for arm,v in r['arms'].items():
  for step,mark in v['marks'].items():
   m=mark['validation'];lines.append(f"| {arm} | {step} | {100*m['sentence_exact']:.3f}% | {100*m['token_accuracy']:.3f}% |")
 lines+=['','## Paired comparisons','']
 for key,v in r.items():
  if key.startswith('paired_'):lines.append(f"{key}: `{json.dumps({label:{k:len(x) if isinstance(x,list) else x for k,x in metric.items()} for label,metric in v.items()})}`")
 lines+=['','## Verification and limits','',
 '- All four downloaded full checkpoints strictly reload; steps/parameters and shared hashes match, optimizer/scheduler/RNG present. Padding-invariance checks pass.',
 '- Best selection includes step0. Compare final fixed-budget and selected best; do not replace original if no reliable gain.',
 '- This is a differentiable sentence-aware surrogate, not exact optimization of whole-sentence binary correctness. It does not expand model context.',
 '- One seed and adaptive reuse of validation require independent confirmation of any gain before a final claim.','']
 (root/'SENTENCE_LOSS_RESULTS.md').write_text('\n'.join(lines))
 (root/'verification.json').write_text(json.dumps(dict(verified=True,checkpoints=proofs),indent=2))
 print('\n'.join(lines))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('directory');main(p.parse_args().directory)
