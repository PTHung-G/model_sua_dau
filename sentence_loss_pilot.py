"""Controlled loss-only pilot; validation only, no submissions.

sentence_hard is a sentence-aware differentiable surrogate, NOT exact sentence
loss. It does not guarantee any particular improvement mechanism or outcome.
Both arms use the unmodified original FactorizedDiac and a fresh identical
optimizer. Trusted local checkpoints are loaded with weights_only=False.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import context_pilot as cp

pa, pb, ev = cp.pa, cp.pb, cp.ev
ARMS = ('token_ce', 'sentence_hard')
VALIDATION_SEED = cp.VALIDATION_SEED
LOSS_DESCRIPTION = ('0.5*token_mean + 0.5*mean_active_sentences('
                    '0.5*sentence_mean + 0.5*mean_top_ceil_0.25_n)')


def training_target(mask, ok, ln):
    """Original training mask, including its batch-wide singleton fallback."""
    target = mask & ok & (ln > 1)
    if not target.any():
        target = mask & ok
    if not target.any():
        raise RuntimeError('No covered training labels')
    return target


def loss_components(logits, y, target, experiment_arm):
    """CE only touches active labels; scatter unreduced CE to sentence layout.

    Masked padding/out-of-dictionary labels may be -1 and are never sent to CE.
    Sentences with no active targets are omitted, not assigned a zero loss.
    torch.topk propagates gradients through its selected CE values.
    """
    if experiment_arm not in ARMS:
        raise ValueError(experiment_arm)
    if not target.any():
        raise RuntimeError('No covered training labels')
    selected_logits, selected_y = logits[target], y[target]
    ce = F.cross_entropy(selected_logits, selected_y, reduction='none')
    # Keep the control EXACTLY the original reduced cross_entropy operation.
    token_mean = F.cross_entropy(selected_logits, selected_y)
    grid = torch.zeros_like(y, dtype=ce.dtype).masked_scatter(target, ce)
    sentence_means, sentence_tops = [], []
    for b in range(target.shape[0]):
        values = grid[b][target[b]]
        if values.numel():
            sentence_means.append(values.mean())
            sentence_tops.append(values.topk(math.ceil(0.25 * values.numel())).values.mean())
    sentence_mean = torch.stack(sentence_means).mean()
    top_mean = torch.stack(sentence_tops).mean()
    surrogate = 0.5 * sentence_mean + 0.5 * top_mean
    loss = token_mean if experiment_arm == 'token_ce' else 0.5 * token_mean + 0.5 * surrogate
    return loss, dict(token_mean=token_mean, sentence_mean=sentence_mean,
                      sentence_top_quartile_mean=top_mean, sentence_surrogate=surrogate,
                      loss=loss)


def detached_components(parts):
    return {name: value.detach().item() for name, value in parts.items()}


def train_arm(arm, args, payload, tr, va, tokens, tables, provenance, base_init, results):
    cp.reset_seed(args.seed)
    model = cp.model_from_checkpoint(payload, 'baseline')
    if type(model) is not pa.FactorizedDiac:
        raise TypeError('Both arms must use unmodified FactorizedDiac')
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    rng = np.random.default_rng(args.seed)
    batches = pb.make_batches(tr, payload['cand'], payload['char2id'], payload['max_c'], 64,
                              rng, nchar=payload['nchar'], forms={s:i for i,s in enumerate(payload['forms'])})
    cp.reset_seed(args.seed)
    metadata = {k: payload[k] for k in ('config', 'cand', 'forms', 'char2id', 'max_c', 'nchar',
                                       'signatures', 'tone_table', 'mod_table')}
    metadata.update(arm='factorized', experiment_arm=arm, seed=args.seed,
                    validation_seed=VALIDATION_SEED, params=sum(p.numel() for p in model.parameters()),
                    run_config=vars(args), provenance=provenance, manifests=provenance,
                    architecture='unmodified pa.FactorizedDiac; all weights trainable',
                    initialization='strict original base; fresh optimizer',
                    optimizer_config=dict(name='AdamW', lr=args.lr, weight_decay=0.0),
                    scheduler_config=dict(name='CosineAnnealingLR', T_max=args.steps),
                    selection='full raw validation sentence_exact then token_accuracy; includes unknowns',
                    loss_description='original token CE' if arm == 'token_ce' else LOSS_DESCRIPTION,
                    train_batch_size=64, eval_batch_size=128)
    outdir = Path(args.out_dir)
    initial = cp.evaluate_model(model, va, payload, tokens, tables)
    if initial['sentence_errors'] != base_init['sentence_errors']:
        raise RuntimeError('Initial predictions differ from base')
    record = dict(arm='factorized', experiment_arm=arm, params=metadata['params'], initial=initial,
                  marks={0: dict(validation=initial)}, best=dict(step=0, validation=initial),
                  train_seconds=0.0, loss_log=[])
    results['arms'][arm] = record
    marks = record['marks']
    metadata.update(best=record['best'], train_seconds=0.0)
    pa.checkpoint(outdir / f'{arm}_best.pt', model, opt, scheduler, rng, 0, metadata, marks)
    pa.atomic_json(outdir / 'results.json', results)
    model.train()
    for step in range(1, args.steps + 1):
        pa.synchronize()
        start = time.perf_counter()
        xc, cl, f, y, ln, mask, ok = next(batches)
        logits = pa.candidate_logits(model, 'factorized', xc, cl, f, ln, mask, tables)
        target = training_target(mask, ok, ln)
        loss, parts = loss_components(logits, y, target, arm)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at {arm} {step}')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        pa.synchronize()
        record['train_seconds'] += time.perf_counter() - start
        components = detached_components(parts)
        if step == 1 or step % 100 == 0 or step == args.steps:
            logged = dict(step=step, lr=scheduler.get_last_lr()[0], **components)
            record['loss_log'].append(logged)
            print(f'{arm} {json.dumps(logged, sort_keys=True)}', flush=True)
        improved = False
        if step % 1000 == 0 or step == args.steps:
            metric = cp.evaluate_model(model, va, payload, tokens, tables)
            metric['paired_vs_base_init'] = cp.paired(base_init, metric)
            marks[step] = dict(validation=metric, loss_components=components)
            record['final'] = metric
            improved = cp.selection_key(metric) > cp.selection_key(record['best']['validation'])
            if improved:
                record['best'] = dict(step=step, validation=metric)
            print(f"{arm} validation step={step} sentence_exact={metric['sentence_exact']:.6f} token_accuracy={metric['token_accuracy']:.6f}", flush=True)
        if step % 500 == 0 or step % 1000 == 0 or step == args.steps:
            metadata.update(best=record['best'], train_seconds=record['train_seconds'], loss_components=components)
            if improved:
                pa.checkpoint(outdir / f'{arm}_best.pt', model, opt, scheduler, rng, step, metadata, marks)
            pa.checkpoint(outdir / f'{arm}_latest.pt', model, opt, scheduler, rng, step, metadata, marks)
            record['step'] = step
            pa.atomic_json(outdir / 'results.json', results)
    return record


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument('--data-dir', default=str(root / 'data'))
    parser.add_argument('--checkpoint', default=str(root / 'kaggle_factorized_full_output/factorized_full_results/factorized_best.pt'))
    parser.add_argument('--out-dir', default='sentence_loss_pilot_results')
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--train-sents', type=int, default=100000)
    parser.add_argument('--val-sents', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(argv)
    if min(args.steps, args.train_sents, args.val_sents, args.lr) <= 0:
        parser.error('steps, sample sizes, and lr must be positive')
    if args.smoke:
        args.steps, args.train_sents, args.val_sents = 2, 128, 20
    for name in ('data_dir', 'checkpoint', 'out_dir'):
        setattr(args, name, str(Path(getattr(args, name)).resolve()))
    return args


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        torch.set_num_threads(4)
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if payload['config']['dim'] != 384:
        raise ValueError('Expected full dim384 checkpoint')
    source_dir = outdir / 'sources'
    source_dir.mkdir(exist_ok=True)
    sources = {}
    for module in (sys.modules[__name__], cp, pa, pb, ev, cp.sa,
                   sys.modules['attn_model'], sys.modules['attn'], sys.modules['vn']):
        path = Path(module.__file__).resolve()
        sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if path != (source_dir / path.name).resolve():
            shutil.copy2(path, source_dir / path.name)
    tr, tr_manifest = pa.reservoir(Path(args.data_dir) / 'tr.txt', args.train_sents, args.seed)
    va, va_manifest = pa.reservoir(Path(args.data_dir) / 'va.txt', args.val_sents, VALIDATION_SEED)
    provenance = dict(train=tr_manifest, validation=va_manifest, source_sha256=sources,
                      checkpoint=dict(path=args.checkpoint, sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                                      step=payload['step']),
                      vocabulary_sha256=hashlib.sha256(json.dumps({k:payload[k] for k in ('cand','forms','char2id')},
                                                                 sort_keys=True, ensure_ascii=False).encode()).hexdigest())
    provenance['config_sha256'] = hashlib.sha256(json.dumps(vars(args), sort_keys=True).encode()).hexdigest()
    pa.atomic_json(outdir / 'sample_manifests.json', provenance)
    tokens = ev.Tokens(va, payload['cand'], payload['char2id'])
    tables = tuple(payload[k].to(pb.DEV) for k in ('tone_table', 'mod_table'))
    cp.reset_seed(args.seed)
    base = cp.model_from_checkpoint(payload, 'baseline')
    base_init = cp.evaluate_model(base, va, payload, tokens, tables)
    del base
    results = dict(config=vars(args), device=pb.DEV, status='training', base_init=base_init,
                   provenance=provenance, arms={}, validation_only=True,
                   interpretation='sentence-aware surrogate, not exact sentence loss; no guaranteed mechanism')
    pa.atomic_json(outdir / 'results.json', results)
    for arm in ARMS:
        train_arm(arm, args, payload, tr, va, tokens, tables, provenance, base_init, results)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    results['paired_sentence_hard_vs_token_ce'] = {
        label: cp.paired(results['arms']['token_ce'][label] if label == 'final' else results['arms']['token_ce']['best']['validation'],
                         results['arms']['sentence_hard'][label] if label == 'final' else results['arms']['sentence_hard']['best']['validation'])
        for label in ('final', 'best')}
    results['status'] = 'complete'
    pa.atomic_json(outdir / 'results.json', results)
    print(f'complete: {outdir / "results.json"}', flush=True)


if __name__ == '__main__':
    main()
