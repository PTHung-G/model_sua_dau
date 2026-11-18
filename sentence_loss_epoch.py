"""Single-arm sentence-hard continuation from original factorized120k best.

Default: 10,000 source batches from a seed-0 permutation of ALL corpus-v2
training rows, without replacement (a PARTIAL epoch, not 120k optimizer steps).
--steps 0 requests the complete epoch, including its final remainder. Validation
only; no public/test selection. Trusted checkpoints use weights_only=False.
Scheduler advances AFTER every consumed source batch, including skipped batches;
its cosine horizon is the planned number of consumed batches, not optimizer steps.
"""
import argparse
import hashlib
from itertools import islice
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from torch import nn

import context_pilot as cp
import sentence_loss_pilot as sl

pa, pb, ev = cp.pa, cp.pb, cp.ev
ARM = 'sentence_hard_full_epoch'
SEED = 0


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_training(path, limit=None):
    """Hash/count ALL raw rows; smoke only materializes an islice prefix."""
    path = Path(path)
    digest, prefix = hashlib.sha256(), hashlib.sha256()
    total = 0
    with path.open('rb') as stream:
        for index, raw in enumerate(stream):
            digest.update(raw)
            if limit is None or index < limit:
                prefix.update(raw)
            total += 1
    if limit is None:
        rows = pb.load_sents(path)
    else:
        with path.open(encoding='utf-8') as stream:
            rows = [line.split() for line in islice(stream, limit)]
    if not rows:
        raise ValueError('Training source is empty')
    return rows, dict(source=str(path.resolve()), source_lines=total,
                      source_sha256=digest.hexdigest(), prefix_count=len(rows),
                      prefix_sha256=prefix.hexdigest(), sampled=len(rows),
                      requested=limit, selection='all rows' if limit is None else 'prefix (debug only)')


def epoch_order(n, seed=SEED):
    """Permutation guarantees unique visits without a corpus-sized visited set."""
    return np.random.default_rng(seed).permutation(n).astype('<i8', copy=False)


def source_batches(order, batch_size=64, budget=None):
    for batch, start in enumerate(range(0, len(order), batch_size)):
        if budget is not None and batch >= budget:
            break
        yield order[start:start + batch_size]


def collate(rows, payload, device=None, forms=None):
    """pb.make_batches known-form encoding, but retain even one-known-token rows.

    Unknown forms are omitted exactly as in pb. Zero-known source rows are counted
    and excluded from tensors. Known forms with invalid gold are kept for context
    but not loss. Preserve pb's label fallback, character truncation/padding, and
    batch-wide ambiguous-target fallback via sl.training_target at training time.
    """
    device = pb.DEV if device is None else device
    cand, char2id = payload['cand'], payload['char2id']
    if forms is None:
        forms = {s: i for i, s in enumerate(payload['forms'])}
    encoded = []
    stats = dict(source_rows=len(rows), zero_known_rows=0, one_known_rows=0,
                 invalid_gold_tokens=0, known_tokens=0)
    for words in rows:
        row = []
        for word in words:
            lower = word.lower()
            form = pb.strip_all(lower)
            choices = cand.get(form)
            if not choices:
                continue
            ci = choices.index(lower) if lower in choices else -1
            row.append((form, len(choices), ci))
            stats['invalid_gold_tokens'] += int(ci < 0)
        stats['known_tokens'] += len(row)
        if not row:
            stats['zero_known_rows'] += 1
        else:
            stats['one_known_rows'] += int(len(row) == 1)
            encoded.append(row)
    if not encoded:
        return None, stats
    B, T = len(encoded), max(map(len, encoded))
    L = max(len(s) for row in encoded for s, _, _ in row)
    xc = torch.full((B, T, L), payload['nchar'] - 1, dtype=torch.long)
    cl = torch.ones((B, T), dtype=torch.long)
    f = torch.zeros((B, T), dtype=torch.long)
    y = torch.zeros((B, T), dtype=torch.long)
    ln = torch.ones((B, T), dtype=torch.long)
    mask = torch.zeros((B, T), dtype=torch.bool)
    ok = torch.zeros_like(mask)
    for b, row in enumerate(encoded):
        for t, (form, k, ci) in enumerate(row):
            ids = [char2id.get(c, 0) for c in form[:64]]
            xc[b, t, :len(ids)] = torch.tensor(ids)
            cl[b, t], f[b, t], y[b, t], ln[b, t] = max(len(ids), 1), forms[form], max(ci, 0), k
            mask[b, t], ok[b, t] = True, ci >= 0
    return tuple(x.to(device) for x in (xc, cl, f, y, ln, mask, ok)), stats


def parse_args(argv=None):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default=str(root / 'data'))
    parser.add_argument('--checkpoint', default=str(root / 'kaggle_factorized_full_output/factorized_full_results/factorized_best.pt'))
    parser.add_argument('--out-dir', default='sentence_loss_epoch_results')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--val-sents', type=int, default=3000)
    parser.add_argument('--steps', type=int, default=10000, help='source batch budget; 0 means full epoch')
    parser.add_argument('--train-limit', type=int, default=None, help='debug prefix only; absent means ALL training rows')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(argv)
    if args.lr <= 0 or args.val_sents <= 0 or args.steps < 0 or (args.train_limit is not None and args.train_limit <= 0):
        parser.error('lr, val-sents, train-limit must be positive; steps must be nonnegative')
    if args.smoke:
        args.train_limit, args.val_sents, args.steps = 128, 20, 2
    for name in ('data_dir', 'checkpoint', 'out_dir'):
        setattr(args, name, str(Path(getattr(args, name)).resolve()))
    return args


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        torch.set_num_threads(4)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if payload['config']['dim'] != 384 or payload.get('arm') != 'factorized' or payload.get('experiment_arm'):
        raise ValueError('Expected original dim384 factorized best, not a pilot continuation')
    if payload.get('step', 0) > 120000:
        raise ValueError('Expected original factorized120k checkpoint')
    tr, train_manifest = load_training(Path(args.data_dir) / 'tr.txt', args.train_limit)
    va, val_manifest = pa.reservoir(Path(args.data_dir) / 'va.txt', args.val_sents, cp.VALIDATION_SEED)
    order = epoch_order(len(tr))
    np.save(out / 'epoch_order.npy', order, allow_pickle=False)
    full_batches = math.ceil(len(tr) / 64)
    planned = min(args.steps, full_batches) if args.steps else full_batches
    planned_visits = min(planned * 64, len(tr))
    train_manifest.update(seed=SEED, permutation_sha256=hashlib.sha256(order.tobytes()).hexdigest(),
                          permutation_file_sha256=sha256_file(out / 'epoch_order.npy'),
                          permutation_count=len(order), planned_visits=planned_visits,
                          planned_prefix_sha256=hashlib.sha256(order[:planned_visits].tobytes()).hexdigest())
    source_dir = out / 'sources'
    source_dir.mkdir(exist_ok=True)
    sources = {}
    for module in (sys.modules[__name__], sl, cp, pa, pb, ev, cp.sa,
                   sys.modules['attn_model'], sys.modules['attn'], sys.modules['vn']):
        path = Path(module.__file__).resolve()
        sources[path.name] = sha256_file(path)
        if path != (source_dir / path.name).resolve():
            shutil.copy2(path, source_dir / path.name)
    provenance = dict(train=train_manifest, validation=val_manifest, source_sha256=sources,
                      checkpoint=dict(path=args.checkpoint, sha256=sha256_file(args.checkpoint), step=payload['step']),
                      vocabulary_sha256=hashlib.sha256(json.dumps({k: payload[k] for k in ('cand', 'forms', 'char2id')}, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                      config_sha256=hashlib.sha256(json.dumps(vars(args), sort_keys=True).encode()).hexdigest())
    pa.atomic_json(out / 'sample_manifests.json', provenance)
    cp.reset_seed(SEED)
    model = cp.model_from_checkpoint(payload, 'baseline')
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=planned)
    rng = np.random.default_rng(SEED)
    tokens = ev.Tokens(va, payload['cand'], payload['char2id'])
    tables = tuple(payload[k].to(pb.DEV) for k in ('tone_table', 'mod_table'))
    initial = cp.evaluate_model(model, va, payload, tokens, tables)
    record = dict(arm='factorized', experiment_arm=ARM, initial=initial, final=initial,
                  best=dict(step=0, validation=initial), marks={0: dict(validation=initial)},
                  loss_log=[], evaluation_log=[dict(step=0, validation=initial)],
                  planned_batches=planned, full_epoch_batches=full_batches, consumed_batches=0,
                  optimizer_steps=0, skipped_batches=0, visited_source_rows=0, epoch_complete=False,
                  budget_complete=False, train_seconds=0.0,
                  coverage=dict(zero_known_rows=0, one_known_rows=0, invalid_gold_tokens=0, known_tokens=0))
    results = dict(config=dict(vars(args), planned_batches=planned), device=pb.DEV, status='training',
                   base_init=initial, provenance=provenance, arms={ARM: record}, validation_only=True,
                   interpretation='sentence-aware surrogate; default 10000-batch partial epoch, no replacement')
    metadata = {k: payload[k] for k in ('config', 'cand', 'forms', 'char2id', 'max_c', 'nchar', 'signatures', 'tone_table', 'mod_table')}
    metadata.update(arm='factorized', experiment_arm=ARM, seed=SEED, validation_seed=cp.VALIDATION_SEED,
                    params=sum(p.numel() for p in model.parameters()), run_config=vars(args),
                    provenance=provenance, manifests=provenance, architecture='unmodified pa.FactorizedDiac; all weights trainable',
                    initialization='strict original base; fresh optimizer', loss_description=sl.LOSS_DESCRIPTION,
                    optimizer_config=dict(name='AdamW', lr=args.lr, weight_decay=0.0),
                    scheduler_config=dict(name='CosineAnnealingLR', T_max=planned, cadence='after each consumed source batch, including skips'),
                    selection='full raw validation sentence_exact then token_accuracy; includes unknowns',
                    train_batch_size=64, eval_batch_size=128)

    def save(kind):
        metadata.update(best=record['best'], train_seconds=record['train_seconds'],
                        epoch_cursor=record['visited_source_rows'], epoch_index=0,
                        permutation_sha256=train_manifest['permutation_sha256'],
                        planned_batches=planned, consumed_batches=record['consumed_batches'],
                        optimizer_steps=record['optimizer_steps'], skipped_batches=record['skipped_batches'],
                        epoch_complete=record['epoch_complete'], coverage=record['coverage'])
        pa.checkpoint(out / f'sentence_hard_epoch_{kind}.pt', model, opt, scheduler, rng,
                      record['consumed_batches'], metadata, record['marks'])
        pa.atomic_json(out / 'results.json', results)

    save('best')
    save('latest')
    forms = {s: i for i, s in enumerate(payload['forms'])}
    for indices in source_batches(order, budget=planned):
        pa.synchronize()
        start = time.perf_counter()
        model.train()
        batch, stats = collate([tr[int(i)] for i in indices], payload, forms=forms)
        for key in record['coverage']:
            record['coverage'][key] += stats[key]
        components = None
        opt.zero_grad(set_to_none=True)
        if batch is None or not (batch[5] & batch[6]).any():
            record['skipped_batches'] += 1
        else:
            xc, cl, f, y, ln, mask, ok = batch
            logits = pa.candidate_logits(model, 'factorized', xc, cl, f, ln, mask, tables)
            loss, parts = sl.loss_components(logits, y, sl.training_target(mask, ok, ln), 'sentence_hard')
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            record['optimizer_steps'] += 1
            components = sl.detached_components(parts)
        record['consumed_batches'] += 1
        record['visited_source_rows'] += len(indices)
        step = record['consumed_batches']
        # Explicit epoch argument uses closed-form source-batch time, even if no
        # optimizer step has occurred. Avoid scheduler-before-optimizer warnings
        # and deprecated epoch arguments by setting its exact cosine state.
        scheduler.last_epoch = step
        scheduler._step_count = step + 1
        lr = args.lr * (1 + math.cos(math.pi * step / planned)) / 2
        for group in opt.param_groups:
            group['lr'] = lr
        scheduler._last_lr = [lr for _ in opt.param_groups]
        pa.synchronize()
        record['train_seconds'] += time.perf_counter() - start
        if step == 1 or step % 100 == 0 or step == planned:
            logged = dict(step=step, optimizer_steps=record['optimizer_steps'], lr=lr,
                          skipped=components is None, **(components or {}))
            record['loss_log'].append(logged)
            print(json.dumps(logged), flush=True)
        improved = False
        if step % 2000 == 0 or step == planned:
            metric = cp.evaluate_model(model, va, payload, tokens, tables)
            metric['paired_vs_base_init'] = cp.paired(initial, metric)
            record['final'] = metric
            record['marks'][step] = dict(validation=metric, loss_components=components)
            record['evaluation_log'].append(dict(step=step, validation=metric))
            improved = cp.selection_key(metric) > cp.selection_key(record['best']['validation'])
            if improved:
                record['best'] = dict(step=step, validation=metric)
        record['epoch_complete'] = record['visited_source_rows'] == len(tr)
        record['budget_complete'] = step == planned
        if step == planned:
            assert record['visited_source_rows'] == planned_visits
            assert record['optimizer_steps'] + record['skipped_batches'] == planned
            assert record['consumed_batches'] == planned
            # Unique by construction: all consumed indices are one permutation
            # prefix, never sampled with replacement or cycled to a second epoch.
            results['status'] = 'complete'
        if improved:
            save('best')
        if step % 1000 == 0 or step == planned:
            save('latest')
    print(f'complete: {out / "results.json"}', flush=True)
    return results


if __name__ == '__main__':
    main()
