"""Fixed covered-training-subset capacity diagnostic (NOT generalization evidence).

Repeat shuffled epochs of 1024 covered sentences; select best by eval-mode train CE.
Uses the original HYBRID architecture, including its default dropout, unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
import torch
from torch.nn import functional as F

import context_pilot as cp
import hybrid_rerank_model as hm
import sentence_loss_epoch as se
import sentence_loss_pilot as sl

pa, pb = cp.pa, cp.pb


def parse_args(argv=None):
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', default=str(root / 'data'))
    p.add_argument('--checkpoint', default=str(root / 'kaggle_factorized_full_output/factorized_full_results/factorized_best.pt'))
    p.add_argument('--out-dir', default=str(root / 'overfit_hybrid_results'))
    p.add_argument('--subset-size', type=int, default=1024)
    p.add_argument('--steps', type=int, default=4000)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--smoke', action='store_true', help='128 covered sentences, two updates, evaluate each update')
    args = p.parse_args(argv)
    if args.smoke:
        args.subset_size, args.steps, args.eval_every = 128, 2, 1
    if args.steps < 0 or min(args.subset_size, args.batch_size, args.eval_every) <= 0 or not math.isfinite(args.lr) or args.lr <= 0:
        p.error('steps must be nonnegative; subset-size, batch-size, eval-every and finite lr must be positive')
    for key in ('data_dir', 'checkpoint', 'out_dir'):
        setattr(args, key, str(Path(getattr(args, key)).resolve()))
    return args


def training_words(words):
    """Same normalized cores as raw scoring, without punctuation/case label misses."""
    return [core for w in words if (core := cp.sa.normalized_core(w))]


def select_subset(rows, payload, size=1024, seed=0):
    """First covered rows in a deterministic permutation of ALL source row indices."""
    selected = []
    for index in se.epoch_order(len(rows), seed):
        words = rows[int(index)]
        scoring = [cp.sa.normalized_core(w) for w in words if cp.sa.is_scoring_token(w)]
        if not scoring:
            continue
        if all((choices := payload['cand'].get(pb.strip_all(w))) and
               w in {cp.sa.normalized_core(c) for c in choices} for w in scoring):
            # The same canonical labels must also be representable by se.collate.
            batch_words = training_words(words)
            if any(w not in payload['cand'].get(pb.strip_all(w), ())
                   for w in batch_words if cp.sa.is_scoring_token(w)):
                continue
            selected.append(int(index))
            if len(selected) == size:
                break
    if len(selected) != size:
        raise ValueError(f'Requested {size} covered sentences; found only {len(selected)}')
    return np.asarray(selected, dtype='<i8')


@torch.inference_mode()
def evaluate_model(model, raw_rows, payload, batch_size=64):
    """Token-weighted supervised CE + ambiguous accuracy + full raw scoring contract."""
    was_training = model.training
    device = next(model.parameters()).device
    model.eval()
    ce_sum = n_supervised = n_ambiguous = ambiguous_correct = 0
    errors, lengths = [], []
    try:
        for start in range(0, len(raw_rows), batch_size):
            batch, _ = se.collate([training_words(w) for w in raw_rows[start:start + batch_size]], payload, device=device)
            if batch is None:
                raise RuntimeError('Covered subset produced an empty batch')
            xc, cl, f, y, ln, mask, ok = batch
            logits = model.logits(xc, cl, f, ln, mask)
            target = sl.training_target(mask, ok, ln)
            ce_sum += F.cross_entropy(logits[target], y[target], reduction='sum').item()
            n_supervised += int(target.sum())
            ambiguous = mask & ok & (ln > 1)
            n_ambiguous += int(ambiguous.sum())
            ambiguous_correct += int(((logits.argmax(-1) == y) & ambiguous).sum())
        for offset, inputs, positions, tensors in cp.raw_batches(raw_rows, payload, bs=batch_size):
            tensors = tuple(t.to(device) for t in tensors)
            picks = model.logits(*tensors).argmax(-1).cpu().tolist()
            for b, words in enumerate(inputs):
                predicted = [cp.sa.normalized_core(w) for w in words]
                for t, (pos, stem) in enumerate(positions[b]):
                    predicted[pos] = cp.sa.normalized_core(payload['cand'][stem][picks[b][t]])
                gold = raw_rows[offset + b]
                if len(gold) != len(predicted):
                    raise ValueError('Whitespace alignment changed')
                scoring = [i for i, w in enumerate(gold) if cp.sa.is_scoring_token(w)]
                lengths.append(len(scoring))
                errors.append(sum(cp.sa.normalized_core(gold[i]) != predicted[i] for i in scoring))
        result = cp.summarize(errors, lengths)
        result.update(supervised_ce=ce_sum / n_supervised, supervised_tokens=n_supervised,
                      ambiguous_token_accuracy=ambiguous_correct / n_ambiguous if n_ambiguous else None,
                      ambiguous_tokens=n_ambiguous, ambiguous_correct=ambiguous_correct,
                      sentence_errors=errors, sentence_lengths=lengths)
        if not math.isfinite(result['supervised_ce']):
            raise RuntimeError('Nonfinite evaluation CE')
        return result
    finally:
        model.train(was_training)


def snapshot_sources(out):
    dest = out / 'sources'
    dest.mkdir(exist_ok=True)
    hashes = {}
    modules = (sys.modules[__name__], hm, se, sl, cp, pa, pb, cp.ev, cp.sa,
               sys.modules['attn_model'], sys.modules['attn'], sys.modules['vn'])
    paths = [Path(m.__file__).resolve() for m in modules]
    paths += [Path(__file__).with_name(name) for name in ('test_overfit_hybrid.py', 'test_hybrid_rerank.py')]
    for path in paths:
        if path.exists():
            hashes[path.name] = se.sha256_file(path)
            if path != (dest / path.name).resolve():
                shutil.copy2(path, dest / path.name)
    pa.atomic_json(out / 'source_sha256.json', hashes)
    return hashes


def run_experiment(args, payload, rows, train_manifest, *, reranker_config=None):
    """In-memory entry point permits CPU toy regression tests; CLI enforces original base."""
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Never silently destroy an earlier diagnostic or its initial state.
    if any((out / name).exists() for name in ('results.json', 'overfit_hybrid_initial.pt',
                                             'overfit_hybrid_best.pt', 'overfit_hybrid_latest.pt')):
        raise FileExistsError(f'Output already contains a run: {out}')
    indices = select_subset(rows, payload, args.subset_size, args.seed)
    fixed = [rows[int(i)] for i in indices]
    np.save(out / 'sample_indices.npy', indices, allow_pickle=False)
    pa.atomic_json(out / 'sample_sentences.json', fixed)
    provenance = dict(train=train_manifest, selection='first covered rows of seeded full-source permutation',
                      seed=args.seed, sample_indices=indices.tolist(), sample_count=len(fixed),
                      indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
                      sample_file_sha256=se.sha256_file(out / 'sample_sentences.json'),
                      indices_file_sha256=se.sha256_file(out / 'sample_indices.npy'),
                      source_sha256=snapshot_sources(out))
    pa.atomic_json(out / 'sample_manifests.json', provenance)
    cp.reset_seed(args.seed)
    model = hm.HybridReranker(payload, 'HYBRID', reranker_config).to(pb.DEV)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1.0)
    rng = np.random.default_rng(args.seed)
    initial = evaluate_model(model, fixed, payload, args.batch_size)
    results = dict(status='training', diagnostic_only=True, generalization_evidence=False,
                   selection='minimum eval-mode fixed-training-subset supervised_ce; NOT validation',
                   config=vars(args), provenance=provenance, initial=initial, final=initial,
                   best=dict(step=0, metrics=initial), evaluation_log=[dict(step=0, metrics=initial)],
                   loss_log=[], optimizer_steps=0, completed_epochs=0, epoch_cursor=0)
    metadata = {k: payload[k] for k in ('config', 'cand', 'forms', 'char2id', 'max_c', 'nchar',
                                        'signatures', 'tone_table', 'mod_table')}
    metadata.update(arm=hm.arm_name('HYBRID'), experiment_arm=hm.arm_name('HYBRID'),
                    config_name='HYBRID', reranker_config=model.cfg, run_config=vars(args),
                    provenance=provenance, diagnostic_only=True, selection=results['selection'],
                    frozen_base_metadata={k: v for k, v in payload.items()
                                          if k not in ('model', 'optimizer', 'scheduler', 'rng', 'marks')},
                    optimizer_config=dict(name='AdamW', lr=args.lr, weight_decay=0.0),
                    scheduler_config=dict(name='LambdaLR', multiplier=1.0, schedule='constant'),
                    normalization='NFC punctuation-free lowercase letter tokens; raw scoring positions retained')
    order = np.empty(0, dtype='<i8')
    cursor = 0
    epoch = 0

    def save(kind, step):
        state_metadata = dict(metadata, best=results['best'], epoch_index=epoch,
                              epoch_cursor=cursor, epoch_order=order.copy(), optimizer_steps=step)
        pa.checkpoint(out / f'overfit_hybrid_{kind}.pt', model, opt, scheduler, rng, step,
                      state_metadata, {entry['step']: entry['metrics'] for entry in results['evaluation_log']})
        pa.atomic_json(out / 'results.json', results)

    if args.steps == 0:
        results['status'] = 'complete'
    # Immutable initial full checkpoint survives all later best/latest updates.
    save('initial', 0)
    save('best', 0)
    save('latest', 0)
    print(json.dumps(dict(step=0, metrics=initial)), flush=True)
    for step in range(1, args.steps + 1):
        if cursor == len(order):
            order = rng.permutation(len(fixed)).astype('<i8', copy=False)
            cursor = 0
            epoch += 1
        chosen = order[cursor:cursor + args.batch_size]
        cursor += len(chosen)
        batch, _ = se.collate([training_words(fixed[int(i)]) for i in chosen], payload)
        if batch is None:
            raise RuntimeError('Covered subset produced an empty training batch')
        xc, cl, f, y, ln, mask, ok = batch
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model.logits(xc, cl, f, ln, mask)
        target = sl.training_target(mask, ok, ln)
        loss = F.cross_entropy(logits[target], y[target])
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        opt.step()
        scheduler.step()
        results['optimizer_steps'] = step
        results['completed_epochs'] = epoch - int(cursor < len(order))
        results['epoch_cursor'] = cursor
        results['loss_log'].append(dict(step=step, loss=loss.item(), lr=opt.param_groups[0]['lr']))
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate_model(model, fixed, payload, args.batch_size)
            results['final'] = metrics
            results['evaluation_log'].append(dict(step=step, metrics=metrics))
            improved = metrics['supervised_ce'] < results['best']['metrics']['supervised_ce']
            if improved:
                results['best'] = dict(step=step, metrics=metrics)
            if step == args.steps:
                results['status'] = 'complete'
            if improved:
                save('best', step)
            save('latest', step)
            print(json.dumps(dict(step=step, metrics=metrics)), flush=True)
    return results


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        torch.set_num_threads(4)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if (payload.get('arm') != 'factorized' or payload.get('experiment_arm') or
            payload['config']['dim'] != 384 or payload.get('step', 0) > 120000):
        raise ValueError('Expected original dim384 factorized120k checkpoint, not continuation')
    rows, manifest = se.load_training(Path(args.data_dir) / 'tr.txt')
    manifest['checkpoint'] = dict(path=args.checkpoint, sha256=se.sha256_file(args.checkpoint), step=payload.get('step'))
    return run_experiment(args, payload, rows, manifest)


if __name__ == '__main__':
    main()
