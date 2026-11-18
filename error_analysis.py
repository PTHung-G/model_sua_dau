"""Validation error analysis for the frozen generator and two-pass corrector.

Consumes existing checkpoints only; no training. Writes one JSON plus a short
markdown summary. Answer ranks are computed over ALL dictionary candidates of
the true stem, which is the information the reranker is allowed to see.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import context_pilot as cp
import two_pass_pilot as tp
from evaluate import Tokens
from pilot_balance import DEV
from score_all import is_scoring_token, normalized_core
from vn import strip_all


def score_tensors(scores_fn, xc, cl, f, ln, mask, tables):
    """Corrector exposes .logits; the plain generator goes through candidate_logits."""
    if scores_fn is None:
        return cp.pa.candidate_logits(model_holder['model'], 'factorized', xc, cl, f, ln, mask, tables)
    return scores_fn(xc, cl, f, ln, mask)


model_holder = {}


def ranks_for(model, raw, payload, tables, use_logits):
    """Return per-sentence token records: rank of the correct candidate."""
    records = []
    model_holder['model'] = model
    scores_fn = model.logits if use_logits else None
    for offset, inputs, rows, tensors in cp.raw_batches(raw, payload, bs=64):
        xc, cl, f, ln, mask = tensors
        with torch.inference_mode():
            scores = score_tensors(scores_fn, xc, cl, f, ln, mask, tables).cpu()
        order = scores.argsort(dim=-1, descending=True)
        for b, ws in enumerate(inputs):
            gold = raw[offset + b]
            if len(gold) != len(ws):
                raise ValueError('alignment changed')
            by_pos = {pos: (t, stem) for t, (pos, stem) in enumerate(rows[b])}
            sent = []
            for pos, word in enumerate(gold):
                if not is_scoring_token(word):
                    continue
                target = normalized_core(word)
                stem = strip_all(normalized_core(ws[pos]))
                cands = payload['cand'].get(stem)
                if cands is None:
                    sent.append(dict(pos=pos, stem=stem, known=False, correct=False,
                                     rank=None, n_cand=0, target=target, predicted=None))
                    continue
                gold_forms = [i for i, c in enumerate(cands) if normalized_core(c) == target]
                t = by_pos[pos][0]
                if not gold_forms:
                    sent.append(dict(pos=pos, stem=stem, known=True, correct=False,
                                     rank=None, n_cand=len(cands), missing=True,
                                     target=target, predicted=None))
                    continue
                ranked = order[b, t, :len(cands)].tolist()
                rank = min(ranked.index(g) for g in gold_forms)
                pick = normalized_core(cands[ranked[0]])
                wrong = len(cands) > 1
                sent.append(dict(pos=pos, stem=stem, known=True, correct=(rank == 0),
                                 rank=rank + 1, n_cand=len(cands), ambiguous=wrong,
                                 predicted=pick, target=target))
            records.append(sent)
    return records


def summarize(records):
    flat = [r for sent in records for r in sent if r['known'] and not r.get('missing')]
    ranks = np.array([r['rank'] for r in flat])
    c = Counter()
    for r in flat:
        if r['correct']:
            c['correct'] += 1
        elif r['rank'] == 2:
            c['wrong_rank2'] += 1
        elif r['rank'] == 3:
            c['wrong_rank3'] += 1
        elif r['rank'] <= 5:
            c['wrong_rank4_5'] += 1
        elif r['rank'] <= 10:
            c['wrong_rank6_10'] += 1
        else:
            c['wrong_rank11plus'] += 1
    amb = [r for r in flat if r.get('ambiguous')]
    amb_c = Counter()
    for r in amb:
        if r['correct']:
            amb_c['correct'] += 1
        elif r['rank'] <= 3:
            amb_c['wrong_within_top3'] += 1
        elif r['rank'] <= 5:
            amb_c['wrong_within_top5'] += 1
        else:
            amb_c['wrong_beyond_top5'] += 1
    return dict(scored_known_tokens=len(flat), rank_counts=dict(c),
                ambiguous_tokens=len(amb), ambiguous_rank_counts=dict(amb_c),
                rank_quantiles={q: float(np.quantile(ranks, q)) for q in (0.5, 0.75, 0.9, 0.95, 0.99)},
                wrong_rank_histogram={str(k): int(v) for k, v in
                                      sorted(Counter(int(x) for x in ranks[ranks > 1]).items()) if v >= 5})


def clustering(records):
    """Is a wrong sentence dominated by isolated errors or multi-word runs?"""
    hist = Counter()
    examples = []
    for sent in records:
        errs = [i for i, r in enumerate(sent) if not r['correct']]
        if not errs:
            continue
        runs, cur = 0, 1
        for a, b in zip(errs, errs[1:]):
            cur = cur + 1 if b == a + 1 else 1
            runs = max(runs, cur)
        hist[f'max_run_{"1" if runs == 1 else "2" if runs == 2 else "3+"}'] += 1
        if len(examples) < 12:
            examples.append(dict(errors=len(errs), max_run=runs,
                                 tokens=[(sent[i]['target'], sent[i]['predicted'], sent[i]['rank'])
                                         for i in errs]))
    return dict(sentences_with_errors=sum(hist.values()), max_error_run=dict(hist), worst_examples=examples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-dir', default='kaggle_two_pass_output/two_pass_results')
    parser.add_argument('--checkpoint', default='kaggle_factorized_full_output/factorized_full_results/factorized_best.pt')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--val-sents', type=int, default=3000)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    torch.set_num_threads(8)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    tr_dir = Path(args.result_dir)
    va, _ = cp.pa.reservoir(Path(args.data_dir) / 'va.txt', args.val_sents, cp.VALIDATION_SEED)

    base = cp.model_from_checkpoint(payload, 'baseline').eval()
    tables = (payload['tone_table'], payload['mod_table'])
    base_records = ranks_for(base, va, payload, tables, use_logits=False)
    corrector, cpk = tp.load_two_pass_checkpoint(tr_dir / 'two_pass_best.pt', device='cpu')
    corr_records = ranks_for(corrector, va, payload, tables, use_logits=True)

    improved = broken = 0
    for a, b in zip(base_records, corr_records):
        ea = sum(not r['correct'] for r in a)
        eb = sum(not r['correct'] for r in b)
        improved += int(ea > 0 and eb == 0)
        broken += int(ea == 0 and eb > 0)

    out = dict(
        checkpoint=args.checkpoint, corrector=dict(path=str(tr_dir / 'two_pass_best.pt'),
                                                   step=cpk['step']),
        frozen_generator=summarize(base_records),
        corrector_best=summarize(corr_records),
        corrector_vs_base=dict(rescued_sentences=improved, broken_sentences=broken),
        clustering_base=clustering(base_records),
        clustering_corrector=clustering(corr_records),
    )
    dest = Path(args.out or tr_dir / 'error_analysis.json')
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items() if not k.startswith('clustering')},
                     indent=2, ensure_ascii=False))
    print(json.dumps(out['clustering_base'], indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
