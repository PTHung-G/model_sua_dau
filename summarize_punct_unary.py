"""Paired document-cluster bootstrap for an existing unary-head validation run."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('source_map', type=Path)
    p.add_argument('raw_validation', type=Path)
    args = p.parse_args()
    r = json.loads((args.run / 'result.json').read_text())
    source_map = np.load(args.source_map)
    with args.raw_validation.open(encoding='utf-8') as handle:
        docs = [json.loads(line)['document_id'] for line in handle]
    clusters = defaultdict(lambda: [0, 0])
    for i, n, delta in r['paired_paragraph_deltas']:
        doc = docs[source_map[i]]
        clusters[doc][0] += n
        clusters[doc][1] += delta
    values = np.array(list(clusters.values()))
    totals = values.sum(axis=0)
    assert totals[0] == r['scores']['flat']['tokens']
    structured = r['scores']['letter_tone'] if 'letter_tone' in r['scores'] else r['scores']['factorized']
    assert totals[1] == structured['correct'] - r['scores']['flat']['correct']
    rng = np.random.default_rng(6902)
    replicates = []
    for _ in range(2000):
        n, d = values[rng.integers(0, len(values), size=len(values))].sum(axis=0)
        replicates.append(100*d/n)
    out = dict(document_clusters=len(values), replicates=len(replicates), seed=6902,
               paired_difference_pp=100*totals[1]/totals[0], percentile_95_ci_pp=np.quantile(replicates,[.025,.975]).tolist())
    text = json.dumps(out, indent=2)+'\n'
    (args.run/'bootstrap.json').write_text(text)
    print(text,end='')


if __name__ == '__main__':
    main()
