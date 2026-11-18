"""Generate a factual English pilot report from verified downloaded artifacts."""
import argparse
import json
from pathlib import Path


def report(directory):
    root = Path(directory)
    data = json.loads((root / 'results.json').read_text())
    proof = json.loads((root / 'verification.json').read_text())
    assert proof['verified']
    arms = {a['arm']: a for a in data['arms']}
    assert set(arms) == {'baseline', 'factorized', 'dual'}
    baseline = arms['baseline']
    cfg = data['config']
    text = [
        '# Controlled architecture pilot — verified results', '',
        '## Protocol', '',
        f"- Training sentences: {cfg['train_sents']:,}; validation sentences: {cfg['val_sents']:,}.",
        f"- Optimizer steps per arm: {cfg['steps']:,}; seed: {cfg['seed']}; batch size: 64.",
        f"- Validation scoring tokens: {data['n_val_tokens']:,}; evaluation batch size: 128.",
        '- All arms are freshly trained with packed character BiGRU states and masked attention padding.',
        '- All arms share reservoir samples, candidate inventory, batch order, learning rate and candidate-masked CE.',
        '- Validation is Wikipedia-domain; no held-out test or public leaderboard scores are used for selection.',
        '- The full v2 candidate inventory is frozen: this is a reduced-gradient-training-data pilot, not a vocabulary rebuilt from 100k sentences.',
        '- The factorized arm uses lossless positional tone/modifier signatures and valid-candidate decoding, not independent argmax/fallback.',
        '- No combined factorization-plus-dual-attention arm has been trained.', '',
        '## Final fixed-budget comparison', '',
        '| Arm | Parameters | ALL | AMB | Δ ALL vs baseline (pp) | Train seconds | Eval seconds |',
        '|---|---:|---:|---:|---:|---:|---:|'
    ]
    for name in ('baseline', 'factorized', 'dual'):
        a = arms[name]
        v = a['validation']
        delta = 100 * (v['all'] - baseline['validation']['all'])
        text.append(f"| {name} | {a['params']:,} | {100*v['all']:.3f}% | {100*v['amb']:.3f}% | {delta:+.3f} | {a['train_seconds']:.1f} | {a['eval_seconds']:.1f} |")
    text += ['', f"Pick-most-frequent baseline ALL: {100*data['pick0_validation']['all']:.3f}%.", '',
             '## Learning curve (validation ALL)', '', '| Arm | Step | ALL |', '|---|---:|---:|']
    for name in ('baseline', 'factorized', 'dual'):
        for step, mark in sorted(arms[name]['marks'].items(), key=lambda x: int(x[0])):
            text.append(f"| {name} | {int(step):,} | {100*mark['validation']['all']:.3f}% |")
    text += ['', '## Artifact verification', '',
             '- All three full checkpoints reload strictly with their saved model configuration.',
             '- Final steps and parameter counts match the results file.',
             '- Train/validation/candidate hashes agree across all arms.',
             '- Reconstructed checkpoint predictions pass character/sentence-padding invariance checks.', '',
             '## Interpretation limits', '',
             '- One seed and one short step budget screen hypotheses; they do not establish a robust final ranking.',
             '- Both experimental arms add parameters. A gain is not evidence that the mechanism alone causes it; parameter-matched controls would be needed next.',
             '- A short-budget loss may reflect slower optimization, not a lower asymptotic ceiling.',
             '- A varying or nonconstant gate is diagnostic, not proof of improved long-range reasoning.',
             '- Do not compare these validation scores directly with historical test scores or official leaderboard metrics.',
             '- No full retraining or combined experiment should be inferred from this pilot.', '']
    path = root / 'PILOT_RESULTS.md'
    path.write_text('\n'.join(text))
    print(path.resolve())
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    report(parser.parse_args().directory)
