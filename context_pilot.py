"""Controlled two-arm fine-tuning pilot; validation only, no submissions.

Trusted local checkpoints use torch.load(weights_only=False). Frozen vocabulary,
encoder configuration and factor tables come exclusively from that checkpoint.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time
import unicodedata

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import evaluate as ev
import pilot_arch as pa
import pilot_balance as pb
import score_all as sa
from attn_model import AttnDiac
from vn import strip_all

ARMS = ('baseline', 'context')
VALIDATION_SEED = 1730


class ContextDiac(pa.FactorizedDiac):
    """Normalized backbone -> packed sentence BiGRU -> zero-init residual.

    Load the complete unextended base STRICTLY before registering new modules.
    Only right padding is supported, matching both batch producers below.
    """
    def __init__(self, nchar, max_c, n_mod, n_tone, *, base_state=None, **config):
        super().__init__(nchar, max_c, n_mod, n_tone, **config)
        if base_state is not None:
            self.load_state_dict(base_state, strict=True)
        if config['dim'] != 384 or not config.get('packed_chars'):
            raise ValueError('Context pilot requires dim384 and packed_chars=True')
        self.context_gru = nn.GRU(384, 64, batch_first=True, bidirectional=True)
        self.context_proj = nn.Linear(128, 384)
        nn.init.zeros_(self.context_proj.weight)
        nn.init.zeros_(self.context_proj.bias)

    def forward(self, xc, cl, mask=None):
        if mask is None:
            mask = torch.ones(xc.shape[:2], dtype=torch.bool, device=xc.device)
        lengths = mask.sum(-1)
        expected = torch.arange(mask.shape[1], device=mask.device)[None] < lengths[:, None]
        if not torch.equal(mask, expected):
            raise ValueError('Sentence masks must be right-padded contiguous prefixes')
        # AttnDiac.out is Identity; this is its EXISTING normalized hidden.
        hidden = AttnDiac.forward(self, xc, cl, mask=mask)
        hidden = hidden.masked_fill(~mask[..., None], 0)
        packed = nn.utils.rnn.pack_padded_sequence(
            hidden, lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.context_gru(packed)
        context, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=hidden.shape[1])
        hidden = (hidden + self.context_proj(context)).masked_fill(~mask[..., None], 0)
        return (self.tone_head(hidden).masked_fill(~mask[..., None], 0),
                self.mod_head(hidden).masked_fill(~mask[..., None], 0))


def reset_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_from_checkpoint(payload, arm):
    if payload.get('arm') != 'factorized' or not payload['config'].get('packed_chars'):
        raise ValueError('Expected original packed factorized checkpoint')
    args = (payload['nchar'], payload['max_c'], len(payload['signatures']['modifier']),
            len(payload['signatures']['tone']))
    if arm == 'context':
        model = ContextDiac(*args, base_state=payload['model'], **payload['config'])
    elif arm == 'baseline':
        model = pa.FactorizedDiac(*args, **payload['config'])
        model.load_state_dict(payload['model'], strict=True)
    else:
        raise ValueError(arm)
    return model.to(pb.DEV)


def load_pilot_checkpoint(path, device=None):
    """Reload a saved baseline/context full checkpoint without discarding extension.

    pa.candidate_logits must be called with arm='factorized' for BOTH models;
    saved arm is the experimental arm, not the output-head type.
    """
    payload = torch.load(path, map_location='cpu', weights_only=False)
    args = (payload['nchar'], payload['max_c'], len(payload['signatures']['modifier']),
            len(payload['signatures']['tone']))
    if payload['arm'] == 'context':
        model = ContextDiac(*args, **payload['config'])
    elif payload['arm'] == 'baseline':
        model = pa.FactorizedDiac(*args, **payload['config'])
    else:
        raise ValueError('Expected baseline/context pilot checkpoint')
    model.load_state_dict(payload['model'], strict=True)
    return model.to(device or pb.DEV).eval(), payload


def raw_batches(raw_sents, payload, bs=128):
    """Input-only encoding; retain raw whitespace positions, even unknowns.

    Unknown tokens are not model context (same known-token stream as training),
    but remain stripped predictions and are NEVER omitted from scoring.
    """
    cand, chars = payload['cand'], payload['char2id']
    forms = {s: i for i, s in enumerate(payload['forms'])}
    for start in range(0, len(raw_sents), bs):
        inputs = [unicodedata.normalize('NFC', strip_all(' '.join(ws))).split()
                  for ws in raw_sents[start:start + bs]]
        rows = [[(i, strip_all(sa.normalized_core(w))) for i, w in enumerate(ws)
                 if strip_all(sa.normalized_core(w)) in cand] for ws in inputs]
        B, T = len(rows), max(1, max(map(len, rows)))
        L = max([1] + [min(64, len(s)) for row in rows for _, s in row])
        xc = torch.full((B, T, L), payload['nchar'] - 1, dtype=torch.long)
        cl = torch.ones((B, T), dtype=torch.long)
        f, ln = torch.zeros_like(cl), torch.ones_like(cl)
        mask = torch.zeros((B, T), dtype=torch.bool)
        for b, row in enumerate(rows):
            for t, (_, stem) in enumerate(row):
                ids = [chars.get(c, 0) for c in stem[:64]]
                xc[b, t, :len(ids)] = torch.tensor(ids)
                cl[b, t], f[b, t], ln[b, t], mask[b, t] = len(ids), forms[stem], len(cand[stem]), True
        yield start, inputs, rows, tuple(x.to(pb.DEV) for x in (xc, cl, f, ln, mask))


def summarize(errors, lengths):
    n = len(errors)
    return dict(sentence_exact=sum(e == 0 for e in errors) / max(1, n),
                token_accuracy=1 - sum(errors) / max(1, sum(lengths)),
                n_sentences=n, n_tokens=sum(lengths), n_errors=sum(errors))


@torch.inference_mode()
def evaluate_model(model, raw_sents, payload, tokens, tables):
    was_training = model.training
    model.eval()
    start = time.perf_counter()
    try:
        def predict(xc, cl, f, ln):
            mask = xc[..., 0] != tokens.pad_c
            return pa.candidate_logits(model, 'factorized', xc, cl, f, ln, mask, tables).argmax(-1)
        known = ev.evaluate(predict, tokens, device=pb.DEV, bs=128)
        errors, lengths, oracle_errors = [], [], []
        unknown_count = 0
        for offset, inputs, rows, tensors in raw_batches(raw_sents, payload):
            xc, cl, f, ln, mask = tensors
            picks = pa.candidate_logits(model, 'factorized', xc, cl, f, ln, mask, tables).argmax(-1).cpu().tolist()
            for b, ws in enumerate(inputs):
                # Scoring compares normalized cores, so restoration of punctuation/case is unnecessary.
                predicted = [sa.normalized_core(w) for w in ws]
                for t, (pos, stem) in enumerate(rows[b]):
                    predicted[pos] = sa.normalized_core(payload['cand'][stem][picks[b][t]])
                gold = raw_sents[offset + b]
                if len(gold) != len(predicted):
                    raise ValueError('Whitespace alignment changed')
                err = n = oracle_err = 0
                for pos, word in enumerate(gold):
                    if not sa.is_scoring_token(word):
                        continue
                    n += 1
                    target = sa.normalized_core(word)
                    stem = strip_all(sa.normalized_core(ws[pos]))
                    cs = payload['cand'].get(stem)
                    unknown_count += int(cs is None)
                    oracle_ok = (target in {sa.normalized_core(c) for c in cs}
                                 if cs else target == sa.normalized_core(ws[pos]))
                    oracle_err += int(not oracle_ok)
                    err += int(target != predicted[pos])
                errors.append(err)
                lengths.append(n)
                oracle_errors.append(oracle_err)
        out = summarize(errors, lengths)
        out.update(known_tokens=known, sentence_errors=errors, sentence_lengths=lengths,
                   oracle_dictionary_ceiling=summarize(oracle_errors, lengths),
                   unknown_letter_tokens=unknown_count,
                   normalization='NFC punctuation-free lowercase letter tokens; unknowns remain stripped input')
        by_length = {}
        for label, low, high in [('0', 0, 0), ('1-10', 1, 10), ('11-20', 11, 20),
                                 ('21-40', 21, 40), ('41+', 41, float('inf'))]:
            indices = [i for i, n in enumerate(lengths) if low <= n <= high]
            by_length[label] = summarize([errors[i] for i in indices], [lengths[i] for i in indices])
        out['by_length'] = by_length
        pa.synchronize()
        out['eval_seconds'] = time.perf_counter() - start
        return out
    finally:
        model.train(was_training)


def paired(reference, other):
    a, b = reference['sentence_errors'], other['sentence_errors']
    if len(a) != len(b):
        raise ValueError('Unpaired validation arrays')
    return dict(rescued_sentences=[i for i, (x, y) in enumerate(zip(a, b)) if x > 0 and y == 0],
                broken_sentences=[i for i, (x, y) in enumerate(zip(a, b)) if x == 0 and y > 0],
                token_errors_reduced=sum(max(0, x-y) for x, y in zip(a, b)),
                token_errors_added=sum(max(0, y-x) for x, y in zip(a, b)))


def selection_key(metrics):
    return metrics['sentence_exact'], metrics['token_accuracy']


def train_arm(arm, args, payload, tr, va, tokens, tables, provenance, base_init, results):
    reset_seed(args.seed)
    model = model_from_checkpoint(payload, arm)
    # Optimizer and cosine schedule are deliberately NEW: do not resume the exhausted 120k schedule.
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    rng = np.random.default_rng(args.seed)
    batches = pb.make_batches(tr, payload['cand'], payload['char2id'], payload['max_c'], 64,
                              rng, nchar=payload['nchar'], forms={s:i for i,s in enumerate(payload['forms'])})
    # Reset after construction too, so any stochastic training uses the same stream.
    reset_seed(args.seed)
    metadata = {k: payload[k] for k in ('config', 'cand', 'forms', 'char2id', 'max_c', 'nchar',
                                       'signatures', 'tone_table', 'mod_table')}
    metadata.update(arm=arm, seed=args.seed, validation_seed=VALIDATION_SEED,
                    params=sum(p.numel() for p in model.parameters()), run_config=vars(args),
                    architecture=dict(backbone='FactorizedDiac', context_gru=arm == 'context',
                                      input_dim=384, hidden_dim=64, bidirectional=True,
                                      residual_projection='Linear128->384 zero weight and bias',
                                      location='after existing normalized AttnDiac hidden',
                                      packed_sentence_lengths=True, all_weights_trainable=True),
                    provenance=provenance, initialization='strict original base then extension; fresh optimizer',
                    optimizer_config=dict(name='AdamW', lr=args.lr, weight_decay=0.0),
                    scheduler_config=dict(name='CosineAnnealingLR', T_max=args.steps),
                    selection='full raw validation sentence_exact then token_accuracy',
                    train_batch_size=64, eval_batch_size=128)
    outdir = Path(args.out_dir)
    initial = evaluate_model(model, va, payload, tokens, tables)
    if initial['sentence_errors'] != base_init['sentence_errors']:
        raise RuntimeError('Initial predictions differ from base')
    record = dict(arm=arm, params=metadata['params'], initial=initial, marks={},
                  best=dict(step=0, validation=initial), train_seconds=0.0)
    results['arms'][arm] = record
    marks = record['marks']
    pa.checkpoint(outdir / f'{arm}_best.pt', model, opt, scheduler, rng, 0, metadata, marks)
    model.train()
    for step in range(1, args.steps + 1):
        pa.synchronize()
        start = time.perf_counter()
        xc, cl, f, y, ln, mask, ok = next(batches)
        logits = pa.candidate_logits(model, 'factorized', xc, cl, f, ln, mask, tables)
        target = mask & ok & (ln > 1)
        if not target.any():
            target = mask & ok
        if not target.any():
            raise RuntimeError('No covered training labels')
        loss = F.cross_entropy(logits[target], y[target])
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at {arm} {step}')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        pa.synchronize()
        record['train_seconds'] += time.perf_counter() - start
        if step % 100 == 0 or step == args.steps:
            print(f'{arm} step={step}/{args.steps} loss={loss.item():.6f} lr={scheduler.get_last_lr()[0]:.8f}', flush=True)
        improved = False
        if step % 1000 == 0 or step == args.steps:
            metric = evaluate_model(model, va, payload, tokens, tables)
            metric['paired_vs_base_init'] = paired(base_init, metric)
            marks[step] = dict(validation=metric, loss=loss.item())
            record['final'] = metric
            improved = selection_key(metric) > selection_key(record['best']['validation'])
            if improved:
                record['best'] = dict(step=step, validation=metric)
            print(f"{arm} validation step={step} sentence_exact={metric['sentence_exact']:.6f} token_accuracy={metric['token_accuracy']:.6f}", flush=True)
        if step % 500 == 0 or step % 1000 == 0 or step == args.steps:
            metadata.update(best=record['best'], train_seconds=record['train_seconds'])
            if improved:
                pa.checkpoint(outdir / f'{arm}_best.pt', model, opt, scheduler, rng, step, metadata, marks)
            pa.checkpoint(outdir / f'{arm}_latest.pt', model, opt, scheduler, rng, step, metadata, marks)
            record['step'] = step
            pa.atomic_json(outdir / 'results.json', results)
    return record


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument('--data-dir', default=str(root / 'data'))
    parser.add_argument('--checkpoint', default=str(root / 'kaggle_factorized_full_output/factorized_full_results/factorized_best.pt'))
    parser.add_argument('--out-dir', default='context_pilot_results')
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--train-sents', type=int, default=100000)
    parser.add_argument('--val-sents', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if min(args.steps, args.train_sents, args.val_sents, args.lr) <= 0:
        parser.error('steps, sample sizes, and lr must be positive')
    if args.smoke:
        args.steps, args.train_sents, args.val_sents = min(args.steps, 2), min(args.train_sents, 128), min(args.val_sents, 20)
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
    for module in (sys.modules[__name__], pa, pb, ev, sa, sys.modules['attn_model'], sys.modules['attn'], sys.modules['vn']):
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
    pa.atomic_json(outdir / 'sample_manifests.json', provenance)
    tokens = ev.Tokens(va, payload['cand'], payload['char2id'])
    tables = tuple(payload[k].to(pb.DEV) for k in ('tone_table', 'mod_table'))
    reset_seed(args.seed)
    base = model_from_checkpoint(payload, 'baseline')
    base_init = evaluate_model(base, va, payload, tokens, tables)
    del base
    results = dict(config=vars(args), device=pb.DEV, status='training', base_init=base_init,
                   provenance=provenance, arms={}, validation_only=True)
    pa.atomic_json(outdir / 'results.json', results)
    for arm in ARMS:
        train_arm(arm, args, payload, tr, va, tokens, tables, provenance, base_init, results)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    results['paired_context_vs_baseline'] = {
        label: paired(results['arms']['baseline'][label] if label == 'final' else results['arms']['baseline']['best']['validation'],
                      results['arms']['context'][label] if label == 'final' else results['arms']['context']['best']['validation'])
        for label in ('final', 'best')}
    results['status'] = 'complete'
    pa.atomic_json(outdir / 'results.json', results)
    print(f'complete: {outdir / "results.json"}', flush=True)


if __name__ == '__main__':
    main()
