"""Measure how many sentences a reranker can rescue, using existing checkpoints only.

No training. For each validation sentence we know the correct candidate's rank
under the frozen generator. This script asks: if a reranker resolved every
disagreement at ranks <= k perfectly, how many sentences become fully correct?

Ranks are restricted to stems with more than one candidate shape, because a
stem with a single shape has nothing to rank. Unknown stems are scored as the
stripped input and cannot be changed by any reranker.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import context_pilot as cp
from score_all import is_scoring_token, normalized_core
from vn import strip_all


def sent_ranks(model, raw, payload, tables, use_logits):
    """Per sentence: list of ranks (1-based) of the correct candidate."""
    out = []
    for offset, inputs, rows, tensors in cp.raw_batches(raw, payload, bs=64):
        xc, cl, f, ln, mask = tensors
        with torch.inference_mode():
            if use_logits:
                scores = model.logits(xc, cl, f, ln, mask)
            else:
                scores = cp.pa.candidate_logits(model, 'factorized', xc, cl, f, ln, mask, tables)
        order = scores.cpu().argsort(dim=-1, descending=True)
        for b, ws in enumerate(inputs):
            gold = raw[offset + b]
            by_pos = {pos: t for t, (pos, _) in enumerate(rows[b])}
            ranks = []
            for pos, word in enumerate(gold):
                if not is_scoring_token(word):
                    continue
                stem = strip_all(normalized_core(ws[pos]))
                cands = payload['cand'].get(stem)
                if cands is None:
                    ranks.append(('unknown', None, 0))
                    continue
                target = normalized_core(word)
                gold_forms = [i for i, c in enumerate(cands) if normalized_core(c) == target]
                if not gold_forms:
                    ranks.append(('missing', None, len(cands)))
                    continue
                ranked = order[b, by_pos[pos], :len(cands)].tolist()
                ranks.append(('known', ranked.index(gold_forms[0]) + 1, len(cands)))
            out.append(ranks)
    return out


def ceiling(records, k, only_ambiguous=True):
    """Count sentences that are fully correct once ranks <= k are guaranteed."""
    exact = 0
    for sent in records:
        ok = True
        for kind, rank, ncand in sent:
            if kind == 'unknown':
                continue
            if kind == 'missing':
                ok = False
                break
            if only_ambiguous and ncand < 2:
                continue
            if rank > k:
                ok = False
                break
        exact += int(ok)
    return exact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-dir', default='kaggle_two_pass_output/two_pass_results')
    parser.add_argument('--checkpoint', default='kaggle_factorized_full_output/factorized_full_results/factorized_best.pt')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--val-sents', type=int, default=3000)
    args = parser.parse_args()
    torch.set_num_threads(8)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    va, _ = cp.pa.reservoir(Path(args.data_dir) / 'va.txt', args.val_sents, cp.VALIDATION_SEED)
    tables = (payload['tone_table'], payload['mod_table'])
    base = cp.model_from_checkpoint(payload, 'baseline').eval()
    records = sent_ranks(base, va, payload, tables, use_logits=False)

    n = len(records)
    known = [r for r in records for kind, _, _ in r if kind == 'known']
    amb = [nc for kind, _, nc in ((k, rk, nc) for r in records for k, rk, nc in r) if kind == 'known' and nc > 1]
    curve = {str(k): dict(sentences=ceiling(records, k),
                          share=ceiling(records, k) / n) for k in (1, 2, 3, 4, 5)}
    # Which single rank matters most: marginal gain of each extra rank level.
    gains = {}
    prev = ceiling(records, 1)
    for k in (2, 3, 4, 5):
        cur = ceiling(records, k)
        gains[str(k)] = cur - prev
        prev = cur
    sent_err = Counter()
    for r in records:
        bad = [k for k, rk, _ in ((a, b, c) for a, b, c in r) if k == 'missing' or (k == 'known' and rk > 1)]
        sent_err['error_sentences'] += int(bool(bad))
        sent_err['sentences_with_rank2_only'] += int(bool(bad) and all(x == 'rank2' for x in bad))
    out = dict(validation_sentences=n,
               known_ambiguous_positions=len(amb),
               single_shape_positions=len(known) - len(amb),
               unknown_positions=sum(1 for r in records for k, _, _ in r if k == 'unknown'),
               exact_at_rank_curve=curve,
               marginal_gain_per_rank_level=gains,
               error_sentence_counts=dict(sent_err))
    dest = Path(args.result_dir) / 'rank_ceiling.json'
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
