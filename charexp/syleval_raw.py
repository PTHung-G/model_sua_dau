"""Evaluate a candidate-lattice model on RAW lines without gold-based filtering.

Unknown input forms receive a singleton COPY candidate. Casing is restored
position-wise from the stripped input. Report coverage separately from accuracy.
The training candidate table remains fixed; appended copy entries use input only.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch

import sylmodel as sm
import syltrain as st
from sylprep import strip_diac

TOK = re.compile(r'[^\W_]+', re.UNICODE)


def restore_case(candidate, source):
    if len(candidate) != len(source):
        return candidate
    return ''.join(c.upper() if s.isupper() else c for c, s in zip(candidate, source))


@torch.no_grad()
def evaluate_raw(model, prefix, path, limit=20000, bs=32):
    device = next(model.parameters()).device
    vocab = json.loads(Path(prefix + '_vocab.json').read_text())
    bases = json.loads(Path(prefix + '_bases.json').read_text())
    chars = sorted({c for s in vocab for c in s} | {c for s in bases for c in s})
    c2i = {c: i + 1 for i, c in enumerate(chars)}
    bid = {b: i + 1 for i, b in enumerate(bases)}
    ctab = model.cand_ids.cpu().clone()
    width = model.base_ch.shape[1]
    new_bases, new_candidates, rows = [], [], []
    raw_rows = []
    with open(path, encoding='utf-8') as stream:
        for line_id, line in enumerate(stream):
            if limit and line_id >= limit:
                break
            gold = TOK.findall(line)
            source = [strip_diac(t) for t in gold]
            ids = []
            for token in source:
                base = token.lower()
                if base not in bid:
                    bid[base] = len(bases) + 1 + len(new_bases)
                    new_bases.append(base)
                    row = torch.full((ctab.shape[1],), -1, dtype=ctab.dtype)
                    row[0] = len(vocab) + len(new_candidates)
                    new_candidates.append(base)
                    rows.append(row)
                ids.append(bid[base])
            raw_rows.append((line_id, gold, source, ids))

    def encode(strings):
        return torch.tensor([[c2i.get(c, 0) for c in s[:width]] +
                             [0] * max(0, width - len(s)) for s in strings],
                            dtype=torch.long, device=device)

    original = {name: getattr(model, name) for name in
                ('base_ch', 'cand_ch', 'base_mask', 'cand_mask', 'cand_ids')}
    if new_bases:
        model.base_ch = torch.cat((model.base_ch, encode(new_bases)))
        model.cand_ch = torch.cat((model.cand_ch, encode(new_candidates)))
        model.cand_ids = torch.cat((model.cand_ids, torch.stack(rows).to(device)))
        model.base_mask = model.base_ch.ne(0).float()
        model.cand_mask = model.cand_ch.ne(0).float()
    spellings = vocab + new_candidates
    model.eval()
    totals = dict(lines=len(raw_rows), tokens=0, correct=0, candidate_covered=0,
                  unknown_base_tokens=0, correct_ambiguous=0, ambiguous_tokens=0)
    order = sorted((r for r in raw_rows if r[3]), key=lambda r: len(r[3]))
    try:
        for offset in range(0, len(order), bs):
            batch = order[offset:offset + bs]
            lengths = torch.tensor([len(r[3]) for r in batch], device=device)
            x = torch.zeros(len(batch), int(lengths.max()), dtype=torch.long, device=device)
            for j, row in enumerate(batch):
                x[j, :len(row[3])] = torch.tensor(row[3], device=device)
            emit, trans, mask = model(x, lengths)
            pred = sm.viterbi(emit, trans, mask).cpu()
            for j, (_, gold, source, ids) in enumerate(batch):
                for t, base_id in enumerate(ids):
                    candidates = model.cand_ids[base_id].cpu()
                    valid = candidates[candidates >= 0].tolist()
                    candidate = spellings[int(candidates[int(pred[j, t])])]
                    restored = restore_case(candidate, source[t])
                    hit = restored == gold[t]
                    covered = any(restore_case(spellings[v], source[t]) == gold[t] for v in valid)
                    ambiguous = len(valid) > 1
                    totals['tokens'] += 1
                    totals['correct'] += int(hit)
                    totals['candidate_covered'] += int(covered)
                    totals['unknown_base_tokens'] += int(base_id > len(bases))
                    totals['ambiguous_tokens'] += int(ambiguous)
                    totals['correct_ambiguous'] += int(hit and ambiguous)
    finally:
        for name, value in original.items():
            setattr(model, name, value)
    totals['accuracy'] = totals['correct'] / max(1, totals['tokens'])
    totals['coverage'] = totals['candidate_covered'] / max(1, totals['tokens'])
    totals['raw_file_sha256'] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    totals['protocol'] = 'First raw lines; all alphanumeric runs; case-sensitive; input-only OOV copy; no gold filtering'
    return totals


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--raw', required=True)
    parser.add_argument('--limit', type=int, default=20000)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = ck['config']
    nchar, base_ch, cand_ch, cid, _ = st.build_tensors(args.prefix)
    model = sm.SylCRF(nchar, base_ch, cand_ch, cid, d=cfg['d'], r=cfg['r'],
                      hidden=cfg['hidden'], layers=cfg['layers'])
    model.load_state_dict(ck['model'])
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    result = evaluate_raw(model, args.prefix, args.raw, args.limit)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
