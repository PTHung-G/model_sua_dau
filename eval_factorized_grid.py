"""Checkpoint-frozen GRID diagnostic, NOT the official seq2seq leaderboard.

Example (after training/validation selection finishes):
  python3 eval_factorized_grid.py --ckpt factorized_full_results/factorized_best.pt \
      --out factorized_grid_diagnostic.json

Only load trusted full checkpoints: torch.load uses weights_only=False. No mutable
candidate files or training/test corpora are read. GRID targets are scoring-only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import unicodedata

import torch

import pilot_arch as pa
import score_all as sa
from vn import strip_all


METRICS = ('all_syll_acc', 'syll_acc', 'stem_cov', 'exact_cov', 'raw_acc')
GRID = tuple(row for row in sa.TESTSETS if row[1].startswith('GRID:'))
PROTOCOL = {
    'status': 'diagnostic only; NOT official seq2seq leaderboard',
    'inputs': 'score_all.load_pairs public GRID input; target used only by score_all.score',
    'vocabulary': 'cand/forms/char2id/config/signatures/tables exclusively from checkpoint',
    'tokenization': 'NFC, strip_all(input), whitespace split; normalized_core + strip_all lookup',
    'inference': 'CPU; packed_chars=True; singleton sentences; all-true covered-token attention mask',
    'candidates': 'pa.candidate_logits with checkpoint form ids, lengths, tone/mod tables',
    'alignment': 'retain whitespace token positions; uncovered tokens remain stripped input',
    'restoration': 'score_all.restore_token preserves input punctuation and simple case',
    'baseline': 'pick0 on identical input tokenization and restoration, no target access',
    'scoring': 'score_all.score: positional, punctuation-free lowercased NFC cores; raw positional tokens',
    'mean': 'unweighted arithmetic mean over the four GRID registers (not pooled tokens)',
}


def model_from_checkpoint(payload):
    """Validate frozen metadata and strictly load the actual factorized architecture."""
    required = ('model', 'step', 'params', 'cand', 'forms', 'char2id', 'nchar',
                'max_c', 'config', 'signatures', 'tone_table', 'mod_table')
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f'Missing checkpoint fields: {missing}')
    if payload.get('arm') != 'factorized':
        raise ValueError('Expected a factorized checkpoint')
    cand, forms = payload['cand'], payload['forms']
    if (not isinstance(cand, dict) or not cand or
            any(not isinstance(cs, list) or not cs or
                any(not isinstance(c, str) for c in cs) for cs in cand.values())):
        raise ValueError('Checkpoint candidates must be nonempty ordered lists')
    if len(forms) != len(cand) or set(forms) != set(cand):
        raise ValueError('Checkpoint forms do not match candidates')
    if payload['max_c'] != max(map(len, cand.values())):
        raise ValueError('Checkpoint max_c disagrees with candidates')
    if payload['config'].get('packed_chars') is not True:
        raise ValueError('This evaluator requires checkpoint packed_chars=True')
    char2id, nchar = payload['char2id'], payload['nchar']
    if (nchar != len(char2id) + 2 or
            set(char2id.values()) != set(range(1, nchar - 1))):
        raise ValueError('Invalid checkpoint character mapping')
    signatures = payload['signatures']
    for name, key in (('tone', 'tone_table'), ('modifier', 'mod_table')):
        table = payload[key]
        if (not torch.is_tensor(table) or table.dtype != torch.long or
                tuple(table.shape) != (len(forms), payload['max_c']) or
                not signatures[name] or table.min().item() < 0 or
                table.max().item() >= len(signatures[name])):
            raise ValueError(f'Invalid checkpoint {key}')
    model = pa.FactorizedDiac(nchar, payload['max_c'], len(signatures['modifier']),
                              len(signatures['tone']), **payload['config']).cpu()
    model.load_state_dict(payload['model'], strict=True)
    if sum(p.numel() for p in model.parameters()) != payload['params']:
        raise ValueError('Checkpoint parameter count disagrees with model')
    model.eval()
    return model


def tokenize(sentence):
    return unicodedata.normalize('NFC', strip_all(sentence)).split()


@torch.inference_mode()
def predict(model, sentences, checkpoint, *, pick0=False):
    """Infer from inputs alone, preserving every whitespace position.

    Singleton sentence batches deliberately avoid fake sentence padding; every
    encoded token is covered, including single-candidate and form-id-zero tokens.
    Characters are packed by the model, with the trainer's 64-character limit.
    """
    cand, char2id = checkpoint['cand'], checkpoint['char2id']
    form_ids = {form: i for i, form in enumerate(checkpoint['forms'])}
    tables = (checkpoint['tone_table'].cpu(), checkpoint['mod_table'].cpu())
    predictions = []
    for sentence in sentences:
        tokens = tokenize(sentence)
        rows = [(i, strip_all(sa.normalized_core(token)))
                for i, token in enumerate(tokens)]
        rows = [(i, stem) for i, stem in rows if stem in cand]
        if not rows:
            predictions.append(tokens)
            continue
        if pick0:
            picks = [0] * len(rows)
        else:
            lengths = [max(1, min(64, len(stem))) for _, stem in rows]
            xc = torch.full((1, len(rows), max(lengths)), checkpoint['nchar'] - 1,
                            dtype=torch.long)
            cl = torch.tensor([lengths], dtype=torch.long)
            f = torch.tensor([[form_ids[stem] for _, stem in rows]], dtype=torch.long)
            ln = torch.tensor([[len(cand[stem]) for _, stem in rows]], dtype=torch.long)
            mask = torch.ones_like(f, dtype=torch.bool)
            for j, (_, stem) in enumerate(rows):
                ids = [char2id.get(c, 0) for c in stem[:64]]
                if ids:
                    xc[0, j, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            logits = pa.candidate_logits(model, 'factorized', xc, cl, f, ln, mask, tables)
            picks = logits.argmax(-1)[0].tolist()
        for (position, stem), pick in zip(rows, picks):
            tokens[position] = sa.restore_token(tokens[position], cand[stem][pick])
        predictions.append(tokens)
    return predictions


def score_predictions(gold, predictions, cand):
    """Use the shared scorer with checkpoint candidates, without global leakage."""
    previous = sa.CAND
    try:
        sa.CAND = cand
        return sa.score(gold, predictions)
    finally:
        sa.CAND = previous


def mean_scores(scores):
    if not scores:
        raise ValueError('Cannot average no registers')
    return {key: sum(row[key] for row in scores) / len(scores) for key in METRICS}


def evaluate_grid(model, checkpoint, limit=0):
    if limit < 0:
        raise ValueError('limit must be nonnegative')
    registers = {}
    for tag, path, note in GRID:
        pairs = sa.load_pairs(path)
        if limit:
            pairs = pairs[:limit]
        inputs = [inp for inp, _ in pairs]
        gold = [target for _, target in pairs]
        predictions = predict(model, inputs, checkpoint)
        baseline = predict(model, inputs, checkpoint, pick0=True)
        scores = score_predictions(gold, predictions, checkpoint['cand'])
        pick0 = score_predictions(gold, baseline, checkpoint['cand'])
        registers[tag] = dict(dataset=path, note=note, n_sentences=len(pairs),
                              scores=scores, pick0=pick0)
        print(tag, ' '.join(f'{key}={scores[key]:.6f}' for key in METRICS), flush=True)
    return dict(registers=registers, MEAN=mean_scores([r['scores'] for r in registers.values()]),
                pick0_MEAN=mean_scores([r['pick0'] for r in registers.values()]))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt', required=True, help='trusted full factorized_best.pt checkpoint')
    parser.add_argument('--out', default='factorized_grid_diagnostic.json')
    parser.add_argument('--limit', type=int, default=0, help='prefix sentences per register; 0 = all')
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error('--limit must be >= 0')
    output = Path(args.out)
    if output.exists():
        parser.error('--out already exists; choose a new diagnostic output path')
    torch.set_num_threads(min(4, torch.get_num_threads()))
    path = Path(args.ckpt).resolve()
    checkpoint_hash = file_sha256(path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model = model_from_checkpoint(checkpoint)
    report = dict(checkpoint=dict(path=str(path), sha256=checkpoint_hash,
                                  step=checkpoint['step'], params=checkpoint['params'],
                                  config=checkpoint['config'], selection=checkpoint.get('selection'),
                                  best=checkpoint.get('best'),
                                  training_manifests=checkpoint.get('manifests')),
                  protocol=PROTOCOL, limit=args.limit)
    start = time.perf_counter()
    report.update(evaluate_grid(model, checkpoint, args.limit))
    report['eval_seconds'] = time.perf_counter() - start
    with output.open('x', encoding='utf-8') as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
        fh.write('\n')
    print('MEAN (unweighted diagnostic)', report['MEAN'], flush=True)
    print(f'wrote {output}', flush=True)


if __name__ == '__main__':
    main()
