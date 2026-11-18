"""Full-corpus v2 factorized training, from scratch; never select on test data.

Default: seed 0, 120000 steps, bs64, AdamW(lr=.002, wd=0), cosine decay.
Smoke: --smoke --skip-final-tests --out-dir /tmp/factorized_full_smoke
Checkpoints are trusted local pickle files (torch.load(..., weights_only=False)).
"""
import argparse
import hashlib
from itertools import islice
import json
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import evaluate as ev
import pilot_arch as pa
import pilot_balance as pb


VALIDATION_SEED = 1730
CONFIG = dict(syl=192, dim=384, blocks=1, heads=8, ch_dim=16,
              window=8, out_std=0.1, drop=0.0, ff_mult=1, packed_chars=True)


def file_manifest(path, limit=0):
    """Hash the complete raw source in O(1) memory, plus the loaded prefix."""
    source_hash, loaded_hash = hashlib.sha256(), hashlib.sha256()
    lines = loaded = 0
    with Path(path).open('rb') as fh:
        for raw in fh:
            lines += 1
            source_hash.update(raw)
            if not limit or lines <= limit:
                loaded_hash.update(raw)
                loaded += 1
    return dict(source=str(Path(path).resolve()), source_lines=lines,
                source_sha256=source_hash.hexdigest(), sampled=loaded,
                sample_sha256=loaded_hash.hexdigest(), limit=limit,
                sampling='all' if not limit else 'streaming prefix')


def load_train(path, limit):
    # Crucial: do NOT load_sents(path)[:limit] in a local smoke run.
    if limit:
        with Path(path).open(encoding='utf-8') as fh:
            return [line.split() for line in islice(fh, limit)]
    return pb.load_sents(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default=str(Path(__file__).resolve().parent / 'data'))
    parser.add_argument('--out-dir', default='factorized_full_results')
    parser.add_argument('--steps', type=int, default=120000)
    parser.add_argument('--train-limit', type=int, default=0,
                        help='0 loads the full training corpus; positive values stream a prefix')
    parser.add_argument('--val-sents', type=int, default=10000)
    parser.add_argument('--eval-every', type=int, default=10000)
    parser.add_argument('--save-every', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--skip-final-tests', action='store_true')
    args = parser.parse_args()
    if min(args.steps, args.val_sents, args.eval_every, args.save_every) < 1 or args.train_limit < 0:
        parser.error('steps, val-sents and intervals must be positive; train-limit must be >= 0')
    if args.smoke:
        args.steps = min(args.steps, 2)
        args.train_limit = min(args.train_limit or 128, 128)
        args.val_sents = min(args.val_sents, 32)
    args.data_dir = str(Path(args.data_dir).resolve())
    args.out_dir = str(Path(args.out_dir).resolve())
    return args


def main():
    args = parse_args()
    outdir, data_dir = Path(args.out_dir), Path(args.data_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    source_dir = outdir / 'sources'
    source_dir.mkdir(exist_ok=True)
    sources = {}
    for module in (sys.modules[__name__], pa, pb, ev, sys.modules['attn_model'], sys.modules['attn'], sys.modules['vn']):
        path = Path(module.__file__).resolve()
        target = source_dir / path.name
        if path != target.resolve():
            shutil.copy2(path, target)
        sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    pa.atomic_json(outdir / 'run_config.json', dict(vars(args), validation_seed=VALIDATION_SEED,
                   model_config=CONFIG, optimizer='AdamW', lr=0.002, weight_decay=0.0,
                   scheduler='CosineAnnealingLR', train_batch_size=64, eval_batch_size=128,
                   initialization='from scratch', source_sha256=sources))
    cand_raw = (data_dir / 'cand.json').read_bytes()
    cand = json.loads(cand_raw)
    if not cand or any(not isinstance(cs, list) or not cs for cs in cand.values()):
        raise ValueError('Candidate dictionary must contain nonempty ordered lists')
    forms = sorted(cand)
    char2id = {c: i for i, c in enumerate(sorted({c for s in forms for c in s}), 1)}
    signatures, tone_table, mod_table = pa.factor_tables(
        cand, forms, outdir / 'factorization_collisions.json')
    tables = (tone_table.to(pb.DEV), mod_table.to(pb.DEV))
    train_manifest = file_manifest(data_dir / 'tr.txt', args.train_limit)
    tr = load_train(data_dir / 'tr.txt', args.train_limit)
    if not tr or len(tr) != train_manifest['sampled']:
        raise ValueError('Empty or changed training corpus')
    va, val_manifest = pa.reservoir(data_dir / 'va.txt', args.val_sents, VALIDATION_SEED)
    manifests = dict(train=train_manifest, validation=val_manifest,
                     cand_sha256=hashlib.sha256(cand_raw).hexdigest())
    pa.atomic_json(outdir / 'sample_manifests.json', manifests)
    tokens = ev.Tokens(va, cand, char2id)
    if not tokens.n_tok:
        raise ValueError('Validation has no evaluable tokens')
    max_c, nchar = max(map(len, cand.values())), len(char2id) + 2
    model = pa.FactorizedDiac(nchar, max_c, len(signatures['modifier']),
                              len(signatures['tone']), **CONFIG).to(pb.DEV)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    rng = np.random.default_rng(args.seed)
    batches = pb.make_batches(tr, cand, char2id, max_c, 64, rng,
                              nchar=nchar, forms={s: i for i, s in enumerate(forms)})
    metadata = dict(arm='factorized', seed=args.seed, validation_seed=VALIDATION_SEED,
                    config=CONFIG, run_config=vars(args), params=params,
                    cand=cand, forms=forms, char2id=char2id, max_c=max_c, nchar=nchar,
                    signatures=signatures, tones=pa.TONES, tone_table=tone_table,
                    mod_table=mod_table, manifests=manifests, source_sha256=sources,
                    optimizer_config=dict(name='AdamW', lr=0.002, weight_decay=0.0),
                    scheduler_config=dict(name='CosineAnnealingLR', T_max=args.steps),
                    objective='candidate-masked CE: ambiguous covered tokens, fallback covered',
                    train_batch_size=64, eval_batch_size=128,
                    initialization='from scratch', selection='validation ALL only')
    pa.atomic_json(outdir / 'metadata.json', {k: v for k, v in metadata.items()
                   if k not in ('cand', 'forms', 'char2id', 'signatures', 'tone_table', 'mod_table')})
    results = dict(config=vars(args), model_config=CONFIG, params=params, device=pb.DEV,
                   validation_seed=VALIDATION_SEED, n_train_sentences=len(tr),
                   n_val_tokens=tokens.n_tok, marks={}, best=None, final_tests={},
                   status='training', train_seconds=0.0, eval_seconds=0.0, save_seconds=0.0)
    with torch.inference_mode():
        results['pick0_validation'] = ev.evaluate(
            lambda xc, cl, f, ln: torch.zeros_like(f), tokens, device=pb.DEV, bs=128)
    marks = results['marks']
    pa.atomic_json(outdir / 'results.json', results)
    print(f"factorized params={params:,} train={len(tr):,} val_tokens={tokens.n_tok:,} "
          f"seed={args.seed} validation_seed={VALIDATION_SEED} device={pb.DEV}", flush=True)
    wall_start = time.perf_counter()
    latest = outdir / 'factorized_latest.pt'
    best = outdir / 'factorized_best.pt'
    model.train()
    for step in range(1, args.steps + 1):
        pa.synchronize()
        start = time.perf_counter()
        xc, cl, f, y, ln, mk, ok = next(batches)
        logits = pa.candidate_logits(model, 'factorized', xc, cl, f, ln, mk, tables)
        ambiguous = mk & (ln > 1) & ok
        target = ambiguous if ambiguous.any() else mk & ok
        if not target.any():
            raise RuntimeError(f'No covered labels at step {step}')
        loss = F.cross_entropy(logits[target], y[target])
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at step {step}')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        pa.synchronize()
        results['train_seconds'] += time.perf_counter() - start
        if step % 1000 == 0:
            print(f"step={step}/{args.steps} loss={float(loss.detach()):.6f} "
                  f"lr={opt.param_groups[0]['lr']:.8f} train_s={results['train_seconds']:.1f}", flush=True)
        evaluate_now = step % args.eval_every == 0 or step == args.steps
        save_now = evaluate_now or step % args.save_every == 0 or step == 5000
        improved = False
        validation = None
        if evaluate_now:
            validation, seconds = pa.evaluate_model(model, 'factorized', tokens, tables)
            results['eval_seconds'] += seconds
            improved = results['best'] is None or validation['all'] > results['best']['validation']['all']
            if improved:
                results['best'] = dict(step=step, validation=validation)
            results['validation'] = validation
            print(f"step={step} loss={float(loss.detach()):.6f} "
                  f"{ev.fmt('validation', validation)} best_step={results['best']['step']}", flush=True)
        if save_now:
            marks[step] = dict(step=step, loss=float(loss.detach()), validation=validation,
                               lr=opt.param_groups[0]['lr'], train_seconds=results['train_seconds'])
            results['step'] = step
            metadata.update(validation=validation, best=results['best'],
                            timing={k: results[k] for k in ('train_seconds', 'eval_seconds', 'save_seconds')})
            start = time.perf_counter()
            # The best file is written at its actual selected step, never relabeled.
            if improved:
                pa.checkpoint(best, model, opt, scheduler, rng, step, metadata, marks)
            pa.checkpoint(latest, model, opt, scheduler, rng, step, metadata, marks)
            results['save_seconds'] += time.perf_counter() - start
            results['wall_seconds'] = time.perf_counter() - wall_start
            pa.atomic_json(outdir / 'results.json', results)
    # Test files are neither opened nor inspected until training/selection end.
    results['status'] = 'training_complete'
    pa.atomic_json(outdir / 'results.json', results)
    if not args.skip_final_tests:
        for filename in ('te.txt', 'te_news.txt'):
            path = data_dir / filename
            if not path.is_file():
                results['final_tests'][filename] = dict(status='missing')
                pa.atomic_json(outdir / 'results.json', results)
                continue
            test_tokens = ev.Tokens(pb.load_sents(path), cand, char2id)
            report = dict(manifest=file_manifest(path), n_tokens=test_tokens.n_tok)
            with torch.inference_mode():
                report['pick0'] = ev.evaluate(
                    lambda xc, cl, f, ln: torch.zeros_like(f), test_tokens, device=pb.DEV, bs=128)
            # Reload each checkpoint: evaluate the persisted final and selected weights.
            for label, checkpoint_path in (('final', latest), ('best', best)):
                payload = torch.load(checkpoint_path, map_location=pb.DEV, weights_only=False)
                model.load_state_dict(payload['model'])
                metrics, seconds = pa.evaluate_model(model, 'factorized', test_tokens, tables)
                report[label] = dict(step=payload['step'], metrics=metrics, eval_seconds=seconds)
                print(f"{filename} {label} step={payload['step']} {ev.fmt('test', metrics)}", flush=True)
                del payload
            results['final_tests'][filename] = report
            pa.atomic_json(outdir / 'results.json', results)
            del test_tokens
    results['status'] = 'complete'
    results['final_tests_skipped'] = args.skip_final_tests
    results['wall_seconds'] = time.perf_counter() - wall_start
    pa.atomic_json(outdir / 'results.json', results)
    print(f"complete latest_step={args.steps} best_step={results['best']['step']} "
          f"params={params:,} results={outdir / 'results.json'}", flush=True)


if __name__ == '__main__':
    main()
