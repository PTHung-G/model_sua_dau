"""Offline verification of downloaded pilot artifacts, including padding invariance.
Usage: python verify_pilot.py OUTPUT_DIRECTORY
Checkpoints are trusted project-produced artifacts (contain optimizer/RNG objects).
"""
import argparse
import json
from pathlib import Path
import torch
from attn_model import AttnDiac
from pilot_arch import FactorizedDiac, candidate_logits


def verify(root):
    root = Path(root)
    results = json.loads((root / 'results.json').read_text())
    assert len(results['arms']) == 3, 'All three arms must finish'
    assert {r['arm'] for r in results['arms']} == {'baseline', 'factorized', 'dual'}
    common = None
    reports = []
    for result in results['arms']:
        arm = result['arm']
        path = root / f"{arm}_s{result['seed']}_latest.pt"
        ck = torch.load(path, map_location='cpu', weights_only=False)
        assert ck['step'] == results['config']['steps'] == result['steps']
        assert ck['params'] == result['params']
        assert all(k in ck for k in ('optimizer', 'scheduler', 'rng', 'cand', 'manifests'))
        hashes = (ck['manifests']['train']['sample_sha256'],
                  ck['manifests']['validation']['sample_sha256'],
                  ck['manifests']['cand_sha256'])
        if common is None:
            common = hashes
        assert hashes == common, 'Arms used different data or candidate inventory'
        cfg = ck['config']
        if arm == 'factorized':
            model = FactorizedDiac(ck['nchar'], ck['max_c'],
                                  len(ck['signatures']['modifier']),
                                  len(ck['signatures']['tone']), **cfg)
            tables = ck['tone_table'], ck['mod_table']
        else:
            model = AttnDiac(ck['nchar'], ck['max_c'], **cfg)
            tables = None
        model.load_state_dict(ck['model'], strict=True)
        model.eval()
        assert sum(p.numel() for p in model.parameters()) == ck['params']
        # Place an identical real sentence in different amounts of word/char padding.
        stems = [s for s in ('toi', 'di', 'hoc') if s in ck['cand']]
        assert stems
        ids = {s: i for i, s in enumerate(ck['forms'])}
        t = len(stems)
        length = max(map(len, stems))
        pad = ck['nchar'] - 1
        x = torch.full((1, t, length), pad, dtype=torch.long)
        cl = torch.tensor([[len(s) for s in stems]])
        f = torch.tensor([[ids[s] for s in stems]])
        ln = torch.tensor([[len(ck['cand'][s]) for s in stems]])
        for j, s in enumerate(stems):
            x[0, j, :len(s)] = torch.tensor([ck['char2id'][c] for c in s])
        big = torch.full((2, t + 4, length + 5), pad, dtype=torch.long)
        big[0, :t, :length] = x[0]
        big[1] = 1
        bcl = torch.ones((2, t + 4), dtype=torch.long)
        bcl[0, :t] = cl[0]
        bcl[1] = length + 5
        bf = torch.zeros((2, t + 4), dtype=torch.long)
        bf[0, :t] = f[0]
        bln = torch.ones_like(bf)
        bln[0, :t] = ln[0]
        mask = torch.zeros_like(bf, dtype=torch.bool)
        mask[0, :t] = True
        mask[1] = True
        with torch.inference_mode():
            small_logits = candidate_logits(model, arm, x, cl, f, ln,
                                             torch.ones_like(f, dtype=torch.bool), tables)[0]
            big_logits = candidate_logits(model, arm, big, bcl, bf, bln, mask, tables)[0, :t]
        error = float((small_logits - big_logits).abs().max())
        assert torch.allclose(small_logits, big_logits, atol=1e-4, rtol=1e-5), (arm, error)
        assert torch.equal(small_logits.argmax(-1), big_logits.argmax(-1))
        reports.append({'arm': arm, 'step': ck['step'], 'params': ck['params'],
                        'padding_max_logit_difference': error,
                        'validation_all': result['validation']['all']})
    output = {'verified': True, 'shared_data_hashes': common, 'arms': reports}
    (root / 'verification.json').write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    verify(parser.parse_args().directory)
