"""Render verified full-training results without equating historical protocols."""
import argparse
import json
from pathlib import Path


def report(directory):
    root=Path(directory)
    r=json.loads((root/'results.json').read_text())
    assert r['status']=='complete'
    assert not r['final_tests_skipped'], 'Full evaluation required'
    assert r['step']==r['config']['steps']
    proofs=[]
    for name in ('factorized_latest.pt','factorized_best.pt'):
        p=json.loads(Path(str(root/name)+'.verified.json').read_text())
        assert p['verified']
        proofs.append(p)
    assert proofs[0]['step']==r['step']
    assert proofs[1]['step']==r['best']['step']
    lines=['# Full factorized tone/modifier training — verified results','',
           '## Training protocol','',
           f"- Training sentences: {r['n_train_sentences']:,}; steps: {r['step']:,}; seed: {r['config']['seed']}.",
           f"- Parameters: {r['params']:,}; validation tokens: {r['n_val_tokens']:,}.",
           '- Packed character BiGRU, masked local attention, separate lossless positional tone/modifier heads.',
           '- Candidate-constrained joint decoding; candidate-masked CE; no dual attention.',
           '- Fresh initialization; AdamW lr 0.002, weight decay 0, cosine schedule, batch 64.',
           '- Best checkpoint selected only by fixed-sample validation ALL.',
           f"- Selected step: {r['best']['step']:,}; validation ALL: {100*r['best']['validation']['all']:.3f}%.",
           f"- Training time: {r['train_seconds']/3600:.3f} hours; wall time: {r['wall_seconds']/3600:.3f} hours.",
           '', '## Held-out tests','',
           '| Test | Tokens | Pick0 ALL | Checkpoint | Step | ALL | AMB | Fix | Break |',
           '|---|---:|---:|---|---:|---:|---:|---:|---:|']
    for test,record in r['final_tests'].items():
        assert 'pick0' in record,test
        for label in ('final','best'):
            q=record[label];m=q['metrics']
            lines.append(f"| {test} | {record['n_tokens']:,} | {100*record['pick0']['all']:.3f}% | {label} | {q['step']:,} | {100*m['all']:.3f}% | {100*m['amb']:.3f}% | {100*m['fix_rate']:.3f}% | {100*m['break_rate']:.3f}% |")
    lines+=['','## Validation learning curve','', '| Step | ALL | AMB |','|---:|---:|---:|']
    for step,mark in sorted(r['marks'].items(),key=lambda p:int(p[0])):
        m=mark.get('validation')
        if m:
            lines.append(f"| {int(step):,} | {100*m['all']:.3f}% | {100*m['amb']:.3f}% |")
    lines+=['','## Verification and limitations','',
            '- Downloaded latest and best weights reload strictly and pass batch/character-padding invariance checks.',
            '- Checkpoints contain candidate inventory, ordered forms, both label tables, optimizer, scheduler and RNG state.',
            '- Full test files were not inspected until training and checkpoint selection ended.',
            '- ALL uses the retained-token evaluation.py protocol, not all public whitespace tokens.',
            '- One seed. No new full-budget flat-output control: historical flat models use different, padding-dependent encoding.',
            '- Thus any historical score difference cannot be attributed solely to factorization.',
            '- Public register-grid diagnostic metrics, if available, use a separate normalization protocol and are not official leaderboard scores.','']
    out=root/'FULL_RESULTS.md';out.write_text('\n'.join(lines));print(out.resolve())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory');report(p.parse_args().directory)
