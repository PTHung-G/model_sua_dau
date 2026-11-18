"""True-epoch feature-aware CRF training. No test-based checkpoint selection.

On Kaggle, embed sylmodel_attention.py as sylmodel.py and the feature trainer
as syltrain.py. Validation uses complete sentences from a separate prefix.
"""
import argparse
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import sylmodel as sm
import syltrain as st

MAXL = 64


def segments(sent, off, max_length=MAXL):
    """Return absolute start/length arrays covering every retained token once."""
    if max_length < 1 or np.any(np.asarray(sent) <= 0):
        raise ValueError('Sentence and segment lengths must be positive')
    starts, lengths = [], []
    for n, start in zip(sent, off):
        for delta in range(0, int(n), max_length):
            starts.append(int(start) + delta)
            lengths.append(min(max_length, int(n) - delta))
    return np.asarray(starts, dtype=np.int64), np.asarray(lengths, dtype=np.int64)


def epoch_batches(lengths, bs, rng, block_batches=32):
    """Shuffle all segments; length-sort only local shuffled blocks. Never drop."""
    if bs < 1 or block_batches < 1:
        raise ValueError('Batch sizes must be positive')
    order = rng.permutation(len(lengths))
    block = bs * block_batches
    for pos in range(0, len(order), block):
        sub = order[pos:pos + block]
        sub = sub[np.argsort(lengths[sub], kind='stable')]
        for j in range(0, len(sub), bs):
            yield sub[j:j + bs]


def make_batch(data, starts, lengths, idx, device):
    base, gold, _, _, features = data
    ln = lengths[idx]
    width = int(ln.max())
    x = np.zeros((len(idx), width), dtype=np.int64)
    y = np.zeros_like(x)
    ff = np.zeros((*x.shape, 2), dtype=np.int64)
    for row, (start, n) in enumerate(zip(starts[idx], ln)):
        x[row, :n] = base[start:start + n]
        y[row, :n] = gold[start:start + n]
        ff[row, :n] = features[start:start + n]
    return tuple(torch.as_tensor(a, device=device) for a in (x, y, ln, ff))


def load_data(prefix, smoke_limit=None):
    base, gold, sent, off = st.load_split(str(prefix))
    features = np.load(str(prefix) + '_input_features.npy', mmap_mode='r')
    if len(base) != len(gold) or int(sent.sum()) != len(base) or np.any(sent <= 0):
        raise ValueError('Invalid sentence/token layout: ' + str(prefix))
    if features.shape != (len(base), 2):
        raise ValueError('Misaligned feature sidecar: ' + str(prefix))
    if smoke_limit:
        sent, off = sent[:smoke_limit], off[:smoke_limit]
        end = int(sent.sum())
        base, gold, features = base[:end], gold[:end], features[:end]
    return base, gold, sent, off, features


def inventory_prefix(prefix):
    """Allow split-local tables or shared train tables in an ancestor directory."""
    prefix = Path(prefix)
    candidates = [prefix] + [p / 'train' for p in prefix.parents]
    for candidate in candidates:
        if all(Path(str(candidate) + s).is_file() for s in
               ('_vocab.json', '_bases.json', '_ctab.npy')):
            return candidate
    raise FileNotFoundError('No complete inventory for ' + str(prefix))


def lr_factor(update, total):
    """Multiplier for a zero-based optimizer update; last update reaches floor."""
    warm = max(1, total // 20)
    if update < warm:
        return (update + 1) / warm
    progress = min(1.0, (update - warm) / max(1, total - warm - 1))
    return 0.05 + 0.95 * (1 + math.cos(math.pi * progress)) / 2


@torch.no_grad()
def evaluate(model, data, bs):
    """All validation sentences, complete lengths, separate feature sidecar."""
    was_training = model.training
    model.eval()
    _, _, sent, off, _ = data
    order = np.argsort(sent, kind='stable')
    correct = tokens = 0
    device = next(model.parameters()).device
    try:
        for pos in range(0, len(order), bs):
            idx = order[pos:pos + bs]
            x, y, ln, ff = make_batch(data, off, sent, idx, device)
            e, t, mask = model(x, ln, ff)
            pred = sm.viterbi(e, t, mask)
            valid = torch.arange(x.shape[1], device=device)[None, :] < ln[:, None]
            correct += int(((pred == y) & valid).sum())
            tokens += int(ln.sum())
    finally:
        model.train(was_training)
    if not tokens:
        raise ValueError('Validation is empty')
    return correct / tokens, tokens


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prefix', default='train')
    ap.add_argument('--val-prefix', required=True)
    for flag, default in [('epochs', 8), ('seed', 0), ('bs', 64), ('d', 128),
                          ('hidden', 136), ('r', 32), ('layers', 2), ('eval-every', 5000)]:
        ap.add_argument('--' + flag, type=int, default=default)
    ap.add_argument('--lr', type=float, default=.0015)
    ap.add_argument('--max-minutes', type=float, default=600)
    ap.add_argument('--out', default='rebalance_latest.pt')
    ap.add_argument('--smoke', action='store_true', help='Use first 32 train and 16 validation sentences')
    a = ap.parse_args()
    if a.epochs < 1 or a.bs < 1 or a.eval_every < 0 or a.max_minutes < 0 or a.lr <= 0:
        ap.error('Invalid epochs, batch size, evaluation interval, time cap, or LR')
    if Path(a.prefix).resolve() == Path(a.val_prefix).resolve():
        ap.error('Training and validation prefixes must differ')
    torch.set_num_threads(2)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    rng = np.random.default_rng(a.seed)
    train = load_data(a.prefix, 32 if a.smoke else None)
    val = load_data(a.val_prefix, 16 if a.smoke else None)
    ip, vp = inventory_prefix(a.prefix), inventory_prefix(a.val_prefix)
    for suffix in ('_vocab.json', '_bases.json', '_ctab.npy'):
        if Path(str(ip) + suffix).read_bytes() != Path(str(vp) + suffix).read_bytes():
            raise ValueError('Training/validation inventory mismatch: ' + suffix)
    nchar, base_ch, cand_ch, cid, _ = st.build_tensors(str(ip))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = sm.SylCRF(nchar, base_ch, cand_ch, cid, d=a.d, hidden=a.hidden,
                      r=a.r, layers=a.layers).to(device)
    starts, lengths = segments(train[2], train[3])
    if not len(starts):
        raise ValueError('Training is empty')
    steps_per_epoch = math.ceil(len(starts) / a.bs)
    total = steps_per_epoch * a.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, total))
    step = tokens_seen = epochs_completed = epoch_tokens = epoch_batches_done = 0
    current_epoch = 1
    best = -1.0
    status = 'running'
    epoch_order = None
    started = time.monotonic()
    print(f'Train sentences={len(train[2])} tokens={len(train[0])} segments={len(starts)} '
          f'steps/epoch={steps_per_epoch} planned_steps={total} device={device}', flush=True)

    def validate_save(reason):
        nonlocal best
        acc, count = evaluate(model, val, a.bs)
        improved = acc > best
        best = max(best, acc)
        state = dict(model=model.state_dict(), optimizer=opt.state_dict(),
                     scheduler=scheduler.state_dict(), config=vars(a), step=step,
                     epoch=current_epoch, epochs_completed=epochs_completed,
                     tokens_seen=tokens_seen, epoch_tokens_seen=epoch_tokens,
                     epoch_batches_completed=epoch_batches_done,
                     epoch_segment_order=epoch_order, status=status,
                     planned_steps=total, steps_per_epoch=steps_per_epoch,
                     val_acc=acc, best_val_acc=best, validation_tokens=count,
                     elapsed_seconds=time.monotonic() - started,
                     numpy_rng=rng.bit_generator.state, numpy_global_rng=np.random.get_state(),
                     python_rng=random.getstate(), torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        for path in [a.out] + ([a.out + '.best.pt'] if improved else []):
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix + '.tmp')
            torch.save(state, tmp)
            os.replace(tmp, target)
        print(f'[{reason}] step={step} completed_epochs={epochs_completed} '
              f'epoch_tokens={epoch_tokens} tokens_seen={tokens_seen} val={acc:.6f} status={status}', flush=True)

    failure = None
    model.train()
    try:
        for current_epoch in range(1, a.epochs + 1):
            epoch_tokens = epoch_batches_done = 0
            batches = list(epoch_batches(lengths, a.bs, rng))
            epoch_order = np.concatenate(batches)
            for idx in batches:
                if a.max_minutes and time.monotonic() - started >= 60 * a.max_minutes:
                    status = 'time_cap'
                    break
                x, y, ln, ff = make_batch(train, starts, lengths, idx, device)
                e, t, mask = model(x, ln, ff)
                loss = sm.nll(e, t, mask, y, ln)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss before update {step + 1}')
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                opt.step()
                scheduler.step()
                step += 1
                seen = int(ln.sum())
                tokens_seen += seen
                epoch_tokens += seen
                epoch_batches_done += 1
                if a.eval_every and step % a.eval_every == 0:
                    validate_save('interval')
            if status == 'time_cap':
                break
            if epoch_tokens != len(train[0]) or epoch_batches_done != steps_per_epoch:
                raise RuntimeError('Epoch coverage accounting mismatch')
            epochs_completed += 1
            status = 'completed' if epochs_completed == a.epochs else 'running'
            validate_save('epoch')
    except (KeyboardInterrupt, FloatingPointError, RuntimeError) as exc:
        status = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
        failure = exc
    finally:
        validate_save('final')
    if failure is not None:
        raise failure


if __name__ == '__main__':
    main()
