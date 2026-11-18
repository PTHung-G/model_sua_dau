"""Frozen factorized120k generator + full-sentence candidate-aware corrector.

Validation only: no submissions/test reads. Original candidate CE, known-only
context stream (unknowns still scored), no teacher forcing or top-k restriction.
This architecture comparison has NO matched-parameter control. Trusted local
checkpoints are loaded with weights_only=False. Default is a 10k-batch partial
no-replacement epoch, NOT 10k random draws nor a complete corpus epoch.
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

import sentence_loss_epoch as se

cp, sl = se.cp, se.sl
pa, pb, ev = cp.pa, cp.pb, cp.ev
ARM = 'two_pass'
CORRECTOR_CONFIG = dict(dim=128, heads=4, layers=2, feedforward=512, dropout=0.1)


class TwoPassDiac(nn.Module):
    def __init__(self, payload, corrector_config=None, load_base=True):
        super().__init__()
        self.corrector_config = dict(CORRECTOR_CONFIG if corrector_config is None else corrector_config)
        cfg = self.corrector_config
        if payload['config']['dim'] != 384 or not payload['config'].get('packed_chars'):
            raise ValueError('Requires packed dim384 original generator')
        self.base = pa.FactorizedDiac(payload['nchar'], payload['max_c'],
                                     len(payload['signatures']['modifier']),
                                     len(payload['signatures']['tone']), **payload['config'])
        if load_base:
            self.base.load_state_dict(payload['model'], strict=True)
        self.base.requires_grad_(False)
        self.base.eval()
        self.register_buffer('tone_table', payload['tone_table'].clone())
        self.register_buffer('mod_table', payload['mod_table'].clone())
        d = cfg['dim']
        self.tone_embedding = nn.Embedding(len(payload['signatures']['tone']), d)
        self.mod_embedding = nn.Embedding(len(payload['signatures']['modifier']), d)
        self.hidden_projection = nn.Linear(384, d)
        layer = nn.TransformerEncoderLayer(d, cfg['heads'], cfg['feedforward'],
                                            cfg['dropout'], batch_first=True)
        self.context = nn.TransformerEncoder(layer, cfg['layers'], enable_nested_tensor=False)
        self.query = nn.Linear(d, d)
        nn.init.zeros_(self.query.weight)
        nn.init.zeros_(self.query.bias)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def frozen_scores(self, xc, cl, f, ln, mask):
        with torch.no_grad():
            # Bypass FactorizedDiac.forward to use the ORIGINAL normalized hidden.
            hidden = cp.AttnDiac.forward(self.base, xc, cl, mask=mask)
            scores = (self.base.tone_head(hidden).gather(-1, self.tone_table[f]) +
                      self.base.mod_head(hidden).gather(-1, self.mod_table[f]))
            invalid = torch.arange(scores.shape[-1], device=scores.device) >= ln[..., None]
            return hidden, scores.masked_fill(invalid, ev.NEG)

    def logits(self, xc, cl, f, ln, mask):
        if xc.shape[0] == 0 or xc.shape[1] == 0:
            return self.query.weight.new_empty((*f.shape, self.tone_table.shape[-1]))
        hidden, base_scores = self.frozen_scores(xc, cl, f, ln, mask)
        candidates = self.tone_embedding(self.tone_table[f]) + self.mod_embedding(self.mod_table[f])
        probabilities = base_scores.softmax(-1)
        expected = torch.einsum('btc,btcd->btd', probabilities, candidates)
        top1 = candidates.gather(2, base_scores.argmax(-1)[..., None, None].expand(
            -1, -1, 1, candidates.shape[-1])).squeeze(2)
        inputs = expected + top1 + self.hidden_projection(hidden)
        t, d = inputs.shape[1:]
        positions = torch.arange(t, device=inputs.device, dtype=inputs.dtype)[:, None]
        rates = torch.exp(torch.arange(0, d, 2, device=inputs.device, dtype=inputs.dtype) * (-math.log(10000.) / d))
        positional = torch.zeros((t, d), device=inputs.device, dtype=inputs.dtype)
        positional[:, 0::2] = torch.sin(positions * rates)
        positional[:, 1::2] = torch.cos(positions * rates)
        inputs = (inputs + positional).masked_fill(~mask[..., None], 0)
        safe_mask = mask.clone()
        safe_mask[~safe_mask.any(-1), 0] = True
        contextual = self.context(inputs, src_key_padding_mask=~safe_mask)
        contextual = contextual.masked_fill(~mask[..., None], 0)
        query = self.query(contextual).masked_fill(~mask[..., None], 0)
        correction = torch.einsum('btd,btcd->btc', query, candidates) / math.sqrt(d)
        scores = base_scores + correction
        invalid = torch.arange(scores.shape[-1], device=scores.device) >= ln[..., None]
        return scores.masked_fill(invalid, ev.NEG)


def load_two_pass_checkpoint(path, device=None):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('arm') != ARM:
        raise ValueError('Expected two_pass checkpoint')
    model = TwoPassDiac(payload, payload['corrector_config'], load_base=False)
    model.load_state_dict(payload['model'], strict=True)
    return model.to(device or pb.DEV).eval(), payload


@torch.inference_mode()
def evaluate_model(model, raw_sents, payload, tokens=None):
    """cp raw normalized scoring, plus full-raw sentence/token top-1/3/5 oracle.

    Unknowns retain stripped predictions even in oracle scores. Candidate-gold
    misses count known tokens whose normalized gold is absent from dictionary.
    """
    was_training = model.training
    model.eval()
    start = time.perf_counter()
    try:
        if tokens is None:
            tokens = ev.Tokens(raw_sents, payload['cand'], payload['char2id'])
        def predict(xc, cl, f, ln):
            return model.logits(xc, cl, f, ln, xc[..., 0] != tokens.pad_c).argmax(-1)
        known = ev.evaluate(predict, tokens, device=pb.DEV, bs=64)
        errors, lengths, dictionary_errors = [], [], []
        top_errors = {k: [] for k in (1, 3, 5)}
        unknown_count = missing = 0
        for offset, inputs, rows, tensors in cp.raw_batches(raw_sents, payload, bs=64):
            xc, cl, f, ln, mask = tensors
            scores = model.logits(xc, cl, f, ln, mask)
            picks = scores.argmax(-1).cpu().tolist()
            ranked = scores.topk(min(5, scores.shape[-1]), dim=-1).indices.cpu().tolist()
            for b, ws in enumerate(inputs):
                predicted = [cp.sa.normalized_core(w) for w in ws]
                options, ranked_options = {}, {}
                for t, (pos, stem) in enumerate(rows[b]):
                    cs = payload['cand'][stem]
                    predicted[pos] = cp.sa.normalized_core(cs[picks[b][t]])
                    options[pos] = {cp.sa.normalized_core(c) for c in cs}
                    ranked_options[pos] = [cp.sa.normalized_core(cs[i]) for i in ranked[b][t] if i < len(cs)]
                gold = raw_sents[offset + b]
                if len(gold) != len(predicted):
                    raise ValueError('Whitespace alignment changed')
                err = n = dictionary_err = 0
                oracle_err = {k: 0 for k in top_errors}
                for pos, word in enumerate(gold):
                    if not cp.sa.is_scoring_token(word):
                        continue
                    n += 1
                    target = cp.sa.normalized_core(word)
                    unknown_count += int(pos not in options)
                    missing += int(pos in options and target not in options[pos])
                    dictionary_err += int(target not in options.get(pos, {predicted[pos]}))
                    err += int(target != predicted[pos])
                    for k in oracle_err:
                        # argmax and topk can break exact ties differently.
                        choices = [predicted[pos]] if k == 1 else ranked_options.get(pos, [predicted[pos]])[:k]
                        oracle_err[k] += int(target not in choices)
                errors.append(err)
                lengths.append(n)
                dictionary_errors.append(dictionary_err)
                for k in top_errors:
                    top_errors[k].append(oracle_err[k])
        out = cp.summarize(errors, lengths)
        out.update(known_tokens=known, sentence_errors=errors, sentence_lengths=lengths,
                   oracle_dictionary_ceiling=cp.summarize(dictionary_errors, lengths),
                   topk_oracle={str(k): cp.summarize(v, lengths) for k, v in top_errors.items()},
                   candidate_gold_missing_count=missing, unknown_letter_tokens=unknown_count,
                   normalization='NFC punctuation-free lowercase letter tokens; unknowns remain stripped input')
        out['by_length'] = {}
        for label, low, high in [('0', 0, 0), ('1-10', 1, 10), ('11-20', 11, 20),
                                 ('21-40', 21, 40), ('41+', 41, float('inf'))]:
            indices = [i for i, n in enumerate(lengths) if low <= n <= high]
            out['by_length'][label] = cp.summarize([errors[i] for i in indices], [lengths[i] for i in indices])
        pa.synchronize()
        out['eval_seconds'] = time.perf_counter() - start
        return out
    finally:
        model.train(was_training)


def parse_args(argv=None):
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', default=str(root / 'data'))
    p.add_argument('--checkpoint', default=str(root / 'kaggle_factorized_full_output/factorized_full_results/factorized_best.pt'))
    p.add_argument('--out-dir', default='two_pass_results')
    p.add_argument('--steps', type=int, default=10000)
    p.add_argument('--val-sents', type=int, default=3000)
    p.add_argument('--lr', type=float, default=0.0003)
    p.add_argument('--train-limit', type=int, default=None, help='debug prefix only')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args(argv)
    if args.steps < 0 or args.val_sents <= 0 or args.lr <= 0 or (args.train_limit is not None and args.train_limit <= 0):
        p.error('steps must be nonnegative; lr, val-sents, train-limit positive')
    if args.smoke:
        args.train_limit, args.val_sents, args.steps = 128, 20, 2
    for key in ('data_dir', 'checkpoint', 'out_dir'):
        setattr(args, key, str(Path(getattr(args, key)).resolve()))
    return args


def snapshot_sources(out):
    dest = out / 'sources'
    dest.mkdir(exist_ok=True)
    hashes = {}
    for module in (sys.modules[__name__], se, sl, cp, pa, pb, ev, cp.sa,
                   sys.modules['attn_model'], sys.modules['attn'], sys.modules['vn']):
        path = Path(module.__file__).resolve()
        hashes[path.name] = se.sha256_file(path)
        shutil.copy2(path, dest / path.name)
    test = Path(__file__).with_name('test_two_pass.py')
    if test.exists():
        hashes[test.name] = se.sha256_file(test)
        shutil.copy2(test, dest / test.name)
    pa.atomic_json(out / 'source_sha256.json', hashes)
    return hashes


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        torch.set_num_threads(4)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Freeze code BEFORE checkpoint/corpus loading or any training.
    sources = snapshot_sources(out)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if (payload.get('arm') != 'factorized' or payload.get('experiment_arm') or
            payload['config']['dim'] != 384 or payload.get('step', 0) > 120000):
        raise ValueError('Expected original factorized120k checkpoint, not continuation')
    tr, train_manifest = se.load_training(Path(args.data_dir) / 'tr.txt', args.train_limit)
    va, val_manifest = pa.reservoir(Path(args.data_dir) / 'va.txt', args.val_sents, cp.VALIDATION_SEED)
    order = se.epoch_order(len(tr))
    np.save(out / 'epoch_order.npy', order, allow_pickle=False)
    full_batches = math.ceil(len(tr) / 64)
    planned = min(args.steps, full_batches) if args.steps else full_batches
    planned_visits = min(planned * 64, len(tr))
    train_manifest.update(seed=0, permutation_sha256=hashlib.sha256(order.tobytes()).hexdigest(),
                          permutation_file_sha256=se.sha256_file(out / 'epoch_order.npy'),
                          permutation_count=len(order), planned_visits=planned_visits,
                          planned_prefix_sha256=hashlib.sha256(order[:planned_visits].tobytes()).hexdigest())
    provenance = dict(train=train_manifest, validation=val_manifest, source_sha256=sources,
                      checkpoint=dict(path=args.checkpoint, sha256=se.sha256_file(args.checkpoint), step=payload['step']),
                      vocabulary_sha256=hashlib.sha256(json.dumps({k: payload[k] for k in ('cand', 'forms', 'char2id')}, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                      config_sha256=hashlib.sha256(json.dumps(vars(args), sort_keys=True).encode()).hexdigest())
    pa.atomic_json(out / 'sample_manifests.json', provenance)
    cp.reset_seed(0)
    model = TwoPassDiac(payload).to(pb.DEV)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=planned)
    rng = np.random.default_rng(0)
    tokens = ev.Tokens(va, payload['cand'], payload['char2id'])
    initial = evaluate_model(model, va, payload, tokens)
    baseline = cp.model_from_checkpoint(payload, 'baseline')
    base_initial = cp.evaluate_model(baseline, va, payload, tokens,
                                     (payload['tone_table'].to(pb.DEV), payload['mod_table'].to(pb.DEV)))
    if initial['sentence_errors'] != base_initial['sentence_errors']:
        raise RuntimeError('Initial raw predictions differ from original frozen base')
    del baseline
    record = dict(arm=ARM, initial=initial, final=initial, best=dict(step=0, validation=initial),
                  marks={0: dict(validation=initial)}, loss_log=[], evaluation_log=[dict(step=0, validation=initial)],
                  planned_batches=planned, full_epoch_batches=full_batches, consumed_batches=0,
                  optimizer_steps=0, skipped_batches=0, visited_source_rows=0, epoch_complete=False,
                  budget_complete=False, train_seconds=0.0,
                  coverage=dict(zero_known_rows=0, one_known_rows=0, invalid_gold_tokens=0, known_tokens=0))
    results = dict(config=dict(vars(args), planned_batches=planned), device=pb.DEV, status='training',
                   base_init=initial, provenance=provenance, arms={ARM: record}, validation_only=True,
                   interpretation='Frozen original generator vs added corrector; no matched-parameter control. Known-only context, unknowns scored unchanged. Original candidate CE, no gold context.')
    metadata = {k: payload[k] for k in ('config', 'cand', 'forms', 'char2id', 'max_c', 'nchar', 'signatures', 'tone_table', 'mod_table')}
    metadata.update(arm=ARM, experiment_arm=ARM, corrector_config=model.corrector_config,
                    frozen_base_metadata={k: v for k, v in payload.items() if k not in ('model', 'optimizer', 'scheduler', 'rng', 'marks')},
                    seed=0, validation_seed=cp.VALIDATION_SEED, run_config=vars(args),
                    params=sum(p.numel() for p in model.parameters()), trainable_params=sum(p.numel() for p in trainable),
                    provenance=provenance, manifests=provenance,
                    architecture='Frozen AttnDiac hidden + full candidate expected/top1 embeddings + projected hidden; sinusoidal full-sentence Transformer; zero-init candidate query; shared signature embeddings',
                    initialization='strict original frozen generator; fresh corrector and optimizer',
                    loss_description='original candidate token CE using sl.training_target, including batch-wide singleton fallback',
                    optimizer_config=dict(name='AdamW', lr=args.lr, weight_decay=0.01),
                    scheduler_config=dict(name='CosineAnnealingLR', T_max=planned, cadence='consumed source batches including skips'),
                    selection='full raw validation sentence_exact then token_accuracy; includes unknowns',
                    train_batch_size=64, eval_batch_size=64)

    def save(kind):
        metadata.update(best=record['best'], train_seconds=record['train_seconds'],
                        epoch_cursor=record['visited_source_rows'], epoch_index=0,
                        permutation_sha256=train_manifest['permutation_sha256'], planned_batches=planned,
                        consumed_batches=record['consumed_batches'], optimizer_steps=record['optimizer_steps'],
                        skipped_batches=record['skipped_batches'], epoch_complete=record['epoch_complete'], coverage=record['coverage'])
        pa.checkpoint(out / f'two_pass_{kind}.pt', model, opt, scheduler, rng,
                      record['consumed_batches'], metadata, record['marks'])
        pa.atomic_json(out / 'results.json', results)

    save('best')
    save('latest')
    print(json.dumps(dict(step=0, topk_oracle=initial['topk_oracle'], sentence_exact=initial['sentence_exact'])), flush=True)
    forms = {s: i for i, s in enumerate(payload['forms'])}
    for indices in se.source_batches(order, budget=planned):
        pa.synchronize()
        start = time.perf_counter()
        model.train()
        batch, stats = se.collate([tr[int(i)] for i in indices], payload, forms=forms)
        for key in record['coverage']:
            record['coverage'][key] += stats[key]
        opt.zero_grad(set_to_none=True)
        loss_value = None
        if batch is None or not (batch[5] & batch[6]).any():
            record['skipped_batches'] += 1
        else:
            xc, cl, f, y, ln, mask, ok = batch
            logits = model.logits(xc, cl, f, ln, mask)
            target = sl.training_target(mask, ok, ln)
            loss = F.cross_entropy(logits[target], y[target])
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            opt.step()
            record['optimizer_steps'] += 1
            loss_value = loss.item()
        record['consumed_batches'] += 1
        record['visited_source_rows'] += len(indices)
        step = record['consumed_batches']
        scheduler.last_epoch, scheduler._step_count = step, step + 1
        lr = args.lr * (1 + math.cos(math.pi * step / planned)) / 2
        for group in opt.param_groups:
            group['lr'] = lr
        scheduler._last_lr = [lr for _ in opt.param_groups]
        pa.synchronize()
        record['train_seconds'] += time.perf_counter() - start
        if step == 1 or step % 100 == 0 or step == planned:
            logged = dict(step=step, optimizer_steps=record['optimizer_steps'], lr=lr, loss=loss_value, skipped=loss_value is None)
            record['loss_log'].append(logged)
            print(json.dumps(logged), flush=True)
        improved = False
        if step % 2000 == 0 or step == planned:
            metric = evaluate_model(model, va, payload, tokens)
            metric['paired_vs_base_init'] = cp.paired(initial, metric)
            paired = metric['paired_vs_base_init']
            print(json.dumps(dict(step=step, sentence_exact=metric['sentence_exact'],
                                  token_accuracy=metric['token_accuracy'],
                                  n_correct_sentences=sum(e == 0 for e in metric['sentence_errors']),
                                  n_sentences=metric['n_sentences'],
                                  rescued_sentences=len(paired['rescued_sentences']),
                                  broken_sentences=len(paired['broken_sentences']),
                                  topk_oracle=metric['topk_oracle'])), flush=True)
            record['final'] = metric
            record['marks'][step] = dict(validation=metric, loss=loss_value)
            record['evaluation_log'].append(dict(step=step, validation=metric))
            improved = cp.selection_key(metric) > cp.selection_key(record['best']['validation'])
            if improved:
                record['best'] = dict(step=step, validation=metric)
        record['epoch_complete'] = record['visited_source_rows'] == len(tr)
        record['budget_complete'] = step == planned
        if step == planned:
            assert record['visited_source_rows'] == planned_visits
            assert record['optimizer_steps'] + record['skipped_batches'] == planned
            results['status'] = 'complete'
        if improved:
            save('best')
        if step % 1000 == 0 or step == planned:
            save('latest')
    best_model, _ = load_two_pass_checkpoint(out / 'two_pass_best.pt')
    record['final_best'] = evaluate_model(best_model, va, payload, tokens)
    record['final_best']['paired_vs_base_init'] = cp.paired(initial, record['final_best'])
    pa.atomic_json(out / 'results.json', results)
    print(json.dumps(dict(best_step=record['best']['step'], final_best_topk_oracle=record['final_best']['topk_oracle'],
                         sentence_exact=record['final_best']['sentence_exact'])), flush=True)
    return results


if __name__ == '__main__':
    main()
