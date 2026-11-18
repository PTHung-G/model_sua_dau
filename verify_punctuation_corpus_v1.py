"""Independently validate stored pairs and article-disjoint split assignment."""
from pathlib import Path
import json, collections
from build_punctuation_corpus_v1 import OUT, make_pair, split_for

def main():
    docs = {}; counts = {}; punctuation_examples = []
    for split in ('train', 'validation', 'test'):
        ids = set(); count = 0
        with (OUT / (split + '.jsonl')).open() as stream:
            for line in stream:
                x = json.loads(line)
                assert split_for(x['document_id']) == split
                source, target = make_pair(x['target'])
                assert x['input'] == source and x['target'] == target
                assert x['source_url'].startswith('http')
                ids.add(x['document_id']); count += 1
                if len(punctuation_examples) < 5 and '?' in target and ',' in target:
                    punctuation_examples.append(x)
        docs[split] = ids; counts[split] = count
    assert not (docs['train'] & docs['validation'] or docs['train'] & docs['test'] or docs['validation'] & docs['test'])
    result = {'all_pairs_verified': True, 'article_disjoint': True, 'counts': counts, 'note': 'All pairs rechecked; document split verified for every row. Lexical cross-split dedup is implemented by builder; not a semantic near-duplicate proof.'}
    (OUT / 'verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)

if __name__ == '__main__': main()
