"""Verify downloaded source snapshots against the submitted notebook receipt."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(root, receipt):
    expected = receipt['embedded_source_sha256']
    result = {}
    for arm in ('control', 'hybrid'):
        folder = root / f'hybrid_rerank_{arm}_results'
        manifest = json.loads((folder / 'source_sha256.json').read_text())
        if not manifest:
            raise ValueError('Empty source manifest')
        for name, digest in manifest.items():
            if name not in expected:
                raise ValueError(f'Unexpected snapshot source: {name}')
            actual = hashlib.sha256((folder / 'sources' / name).read_bytes()).hexdigest()
            if actual != digest or digest != expected[name]:
                raise ValueError(f'Submission source mismatch: {arm}/{name}')
        for required in ('hybrid_rerank_train.py', 'hybrid_rerank_model.py', 'test_hybrid_rerank.py'):
            if required not in manifest:
                raise ValueError(f'Missing critical source: {arm}/{required}')
        result[arm] = {'matched_sources': len(manifest), 'submitted_version': receipt['version']}
    return result


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=str(root / 'kaggle_hybrid_rerank_output'))
    parser.add_argument('--receipt', default=str(root / 'kp_hybrid_rerank/submission_receipt_v1.json'))
    args = parser.parse_args()
    out = Path(args.root)
    result = verify(out, json.loads(Path(args.receipt).read_text()))
    (out / 'source_verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
