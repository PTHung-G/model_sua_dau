"""Measure how many validation sentences sit one candidate-swap from correct.

A sentence that is wrong by a single token stays wrong under every reranker that
only ever moves one decision at a time: fixing it requires one replacement, so
no per-token loss can reach it by gradual improvement. This counts, for each
wrong sentence, whether flipping exactly one stem to another candidate makes the
whole sentence correct, and how deep that candidate sits in the generator's
candidate list.

Read-only: no training, no checkpoint modified.
"""
import argparse
import collections
import json
import time
from pathlib import Path

import torch

import context_pilot as cp
import pilot_arch as pa
import score_all as sa
from vn import strip_all


def ncore(word):
    return sa.normalized_core(word)


@torch.inference_mode()
def analyse(model, payload, raw, tables):
    cand = payload['cand']
    total = one_swap = 0
    depths = collections.Counter()
    wrong_sentences = 0
    per_sentence = []
    for offset, inputs, rows, tensors in cp.raw_batches(raw, payload):
        xc, cl, f, ln, mask = tensors
        picks = pa.candidate_logits(model, 'factorized', xc, cl, f, ln, mask, tables).argmax(-1).cpu().tolist()
        for b, ws in enumerate(inputs):
            gold = raw[offset + b]
            if len(gold) != len(ws):
                raise ValueError('Whitespace alignment changed')
            predicted = [ncore(w) for w in ws]
            for t, (pos, stem) in enumerate(rows[b]):
                predicted[pos] = ncore(cand[stem][picks[b][t]])
            scored = [i for i, w in enumerate(gold) if sa.is_scoring_token(w)]
            n = len(scored)
            if n == 0:
                continue
            total += 1
            wrong = [i for i in scored if ncore(gold[i]) != predicted[i]]
            if not wrong:
                per_sentence.append(0)
                continue
            wrong_sentences += 1
            # Only a sentence with exactly one wrong scoring token can be repaired
            # by a single replacement.
            found = 0
            if len(wrong) == 1:
                pos = wrong[0]
                row = dict(rows[b])
                stem = row.get(pos)
                if stem is not None:
                    for depth, option in enumerate(cand[stem]):
                        if ncore(option) == ncore(gold[pos]):
                            found = 1
                            depths[depth + 1] += 1
                            break
                else:
                    # Unknown stem: scored as stripped input, nothing to swap to.
                    found = 0
            one_swap += found
            per_sentence.append(len(wrong))
    return dict(sentences=total, wrong_sentences=wrong_sentences,
                single_wrong_token=sum(1 for e in per_sentence if e == 1),
                one_swap_from_correct=one_swap,
                gold_depth_of_one_swap=dict(sorted(depths.items())),
                histogram_wrong_tokens=dict(sorted(collections.Counter(per_sentence).items())))


def main(argv=None):
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', default=str(root / 'data'))
    p.add_argument('--checkpoint', default=str(root / 'kaggle_factorized_full_output/factorized_full_results/factorized_best.pt'))
    p.add_argument('--sentences', type=int, default=3000)
    p.add_argument('--out', default=str(root / 'one_swap.json'))
    args = p.parse_args(argv)

    payload = torch.load(str(args.checkpoint), map_location='cpu', weights_only=False)
    raw, manifest = pa.reservoir(Path(args.data_dir) / 'va.txt', args.sentences, cp.VALIDATION_SEED)
    model = cp.model_from_checkpoint(payload, 'baseline')
    tables = (payload['tone_table'], payload['mod_table'])
    start = time.perf_counter()
    out = analyse(model.eval(), payload, raw, tables)
    out['baseline_sentence_exact'] = 1 - out['wrong_sentences'] / max(1, out['sentences'])
    out['seconds'] = time.perf_counter() - start
    out['seed'] = cp.VALIDATION_SEED
    out['sample_sha256'] = manifest['sample_sha256']
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
