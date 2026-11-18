"""Frozen COPY and train-frequency baselines; no fitting on evaluation labels.

build_punct_tensors_v1.build_table orders each train-only candidate row with
Counter.most_common(KMAX). Thus native ctab[:, 0] is the most frequent spelling,
with ties resolved by first occurrence in training. Reuses the neural pipeline's
case restoration, unknown COPY, token spans, and raw scoring, not its weights.
"""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eval_punct_vs_best import TOK, strip, restore_case, replace_spans, read_pairs, score_pairs


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    here = Path(__file__).resolve().parent
    sample = here / 'sample.jsonl'
    prefix = ROOT / 'punct_tensors_audit_v2' / 'train'
    paths = {name: Path(str(prefix) + suffix) for name, suffix in
             [('bases', '_bases.json'), ('vocab', '_vocab.json'), ('ctab', '_ctab.npy')]}
    bases = json.loads(paths['bases'].read_text())
    vocab = json.loads(paths['vocab'].read_text())
    table = np.load(paths['ctab'], mmap_mode='r')
    assert len(table) == len(bases) + 1
    lookup = {}
    for i, base in enumerate(bases, 1):
        assert table[i, 0] >= 0
        form = vocab[int(table[i, 0])]
        assert strip(form).lower() == base
        lookup[base] = form
    pairs = read_pairs(sample)
    predictions = {'COPY': [], 'Train-frequency': []}
    # Prediction accesses source only. Evaluation labels enter only score_pairs.
    for source, _ in pairs:
        predictions['COPY'].append(source)
        words = []
        for m in TOK.finditer(source):
            form = strip(m.group()).lower()
            words.append(restore_case(lookup.get(form, form), m.group()))
        predictions['Train-frequency'].append(replace_spans(source, words))
    results = {name: score_pairs(pairs, pred) for name, pred in predictions.items()}
    frozen = json.loads((here / 'result.json').read_text())
    result = {'sample_file_sha256': sha(sample),
              'inventory_sha256': {name: sha(path) for name, path in paths.items()},
              'protocol': 'same locked 500 raw rows; no label filtering; training-only native frequency order; first training occurrence breaks ties; native case transfer; unknown COPY',
              'baselines': results, 'frozen_model_result_reference': 'result.json'}
    with (here / 'baseline_results.json').open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with (here / 'baseline_predictions.jsonl').open('x') as f:
        for i in range(len(pairs)):
            f.write(json.dumps({'row': i, **{k: v[i] for k, v in predictions.items()}}, ensure_ascii=False) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
