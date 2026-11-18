"""True-epoch punctuation-aware CRF training.

Reads tensors produced by build_punct_tensors_v1.py. Every retained token is
covered exactly once per epoch; validation uses complete sentences.
"""
import argparse
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import sylmodel_punct as sm

MAXL = 64


def segments(sent, off, max_length=MAXL):
    if max_length < 1 or np.any(np.asarray(sent) <= 0):
        raise ValueError('Sentence and segment lengths must be positive')
    starts, lengths = [], []
    for n, start in zip(sent, off):
        for delta in range(0, int(n), max_length):
            starts.append(int(start) + delta)
            lengths.append(min(max_length, int(n) - delta))
    return np.asarray(starts, dtype=np.int64), np.asarray(lengths, dtype=np.int64)


def epoch_batches(lengths, bs, rng, block_batches=32):
    if bs < 1 or block_batches < 1:
        raise ValueError('Batch sizes must be positive')
    order = rng.permutation(len(lengths))
    block = bs * block_batches
    for pos in range(0, len(order), block):
        sub = order[pos:pos + block]
        sub = sub[np.argsort(lengths[sub], kind='stable')]
        for j in range(0, len(sub), bs):
            yield sub[j:j + bs]


def load_split(prefix):
    import json
    manifest_path = Path(prefix).parent / 'manifest.json'
    if not manifest_path.exists() or json.loads(manifest_path.read_text()).get('pipeline_schema') != 2:
        raise ValueError('Legacy or unverified tensor schema; rebuild schema2 before training')
    d = np.load(str(prefix) + '.npz')
    sent = d['sent'].astype(np.int64)
    off = np.zeros(len(sent), dtype=np.int64)
    np.cumsum(sent[:-1], out=off[1:])
    return (d['base'].astype(np.int64), d['gold'].astype(np.int64), sent, off,
            np.load(str(prefix) + '_input_features.npy', mmap_mode='r'),
            np.load(str(prefix) + '_gaps.npy', mmap_mode='r'))


def make_batch(data, starts, lengths, idx, device):
    base, gold, sent, off, features, gaps = data
    idx = np.asarray(idx, dtype=np.int64)
    ln = lengths[idx]
    width = int(ln.max())
    B = len(idx)
    x = np.zeros((B, width), dtype=np.int64)
    y = np.zeros_like(x)
    ff = np.zeros((B, width), dtype=np.int64)
    gg = np.zeros((B, width + 1), dtype=np.int64)
    for row, (start, n) in enumerate(zip(starts[idx], ln)):
        x[row, :n] = base[start:start + n]
        y[row, :n] = gold[start:start + n]
        ff[row, :n] = features[start:start + n]
        # Every preceding paragraph contributes one extra trailing gap.
        paragraph = int(np.searchsorted(off, start, side='right') - 1)
        if paragraph < 0 or start + n > off[paragraph] + sent[paragraph]:
            raise ValueError('Segment crosses a paragraph boundary')
        gap_start = int(start) + paragraph
        gg[row, :n + 1] = gaps[gap_start:gap_start + n + 1]
    return tuple(torch.as_tensor(a, device=device) for a in (x, y, ln, ff, gg))


def load_data(prefix, smoke_limit=None):
    base, gold, sent, off, features, gaps = load_split(prefix)
    if not len(sent) or len(base) != len(gold) or int(sent.sum()) != len(base) or np.any(sent <= 0):
        raise ValueError('Invalid layout: ' + str(prefix))
    if features.shape != (len(base),) or gaps.shape != (len(base) + len(sent),):
        raise ValueError('Misaligned sidecars: ' + str(prefix))
    if smoke_limit:
        k = min(smoke_limit, len(sent))
        sent = sent[:k]
        off = np.zeros(k, dtype=np.int64)
        np.cumsum(sent[:-1], out=off[1:])
        end = int(sent.sum())
        base, gold, features = base[:end], gold[:end], features[:end]
        gaps = gaps[:end + k]
    return base, gold, sent, off, features, gaps


def lr_factor(update, total):
    warm = max(1, total // 20)
    if update < warm:
        return (update + 1) / warm
    progress = min(1.0, (update - warm) / max(1, total - warm - 1))
    return 0.05 + 0.95 * (1 + math.cos(math.pi * progress)) / 2


@torch.no_grad()
def evaluate(model, data, bs):
    was_training = model.training
    model.eval()
    _, _, sent, off, _, _ = data
    order = np.argsort(sent, kind='stable')
    correct = tokens = 0
    device = next(model.parameters()).device
    try:
        for pos in range(0, len(order), bs):
            idx = order[pos:pos + bs]
            x, y, ln, ff, gg = make_batch(data, off, sent, idx, device)
            e, t, mask = model(x, ln, ff, gg)
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
    ap.add_argument('--inventory-prefix', default=None,
                    help='prefix holding _ctab/_bases/_vocab/_punct (defaults to prefix)')
    for flag, default in [('epochs', 8), ('seed', 0), ('bs', 32), ('d', 128),
                          ('hidden', 136), ('r', 32), ('layers', 2), ('eval-every', 5000)]:
        ap.add_argument('--' + flag, type=int, default=default)
    ap.add_argument('--lr', type=float, default=.0015)
    ap.add_argument('--max-minutes', type=float, default=600)
    ap.add_argument('--out', default='punct_latest.pt')
    ap.add_argument('--resume', default=None)
    ap.add_argument('--smoke', action='store_true', help='Load a small subset; does NOT authorize CPU training')
    ap.add_argument('--execute-gpu-training', action='store_true', help='Explicit GPU execution gate after audit')
    a = ap.parse_args()
    if a.epochs < 1 or a.bs < 1 or a.eval_every < 0 or a.max_minutes < 0 or a.lr <= 0:
        ap.error('Invalid epochs, batch size, evaluation interval, time cap, or LR')
    if Path(a.prefix).resolve() == Path(a.val_prefix).resolve():
        ap.error('Training and validation prefixes must differ')
    if not a.execute_gpu_training:
        ap.error('Preflight only: GPU training requires explicit --execute-gpu-training')
    if not torch.cuda.is_available():
        ap.error('GPU is required; CPU training is disabled')
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    rng = np.random.default_rng(a.seed)
    train = load_data(a.prefix, 32 if a.smoke else None)
    val = load_data(a.val_prefix, 16 if a.smoke else None)
    ip = Path(a.inventory_prefix or a.prefix)
    nchar, base_ch, cand_ch, cid, _ = build_tensors(str(ip))
    npunct = len(__import__('json').loads(Path(str(ip) + '_punct.json').read_text()))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = sm.PunctSylCRF(nchar, base_ch, cand_ch, cid, npunct, d=a.d,
                           hidden=a.hidden, r=a.r, layers=a.layers).to(device)
    starts, lengths = segments(train[2], train[3])
    if not len(starts):
        raise ValueError('Training is empty')
    steps_per_epoch = math.ceil(len(starts) / a.bs)
    total = max(1, steps_per_epoch * a.epochs)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, total))
    step = tokens_seen = epochs_completed = epoch_tokens = epoch_batches_done = 0
    current_epoch = 1
    best = -1.0
    status = 'running'
    epoch_order = None
    started = time.monotonic()
    resume_pending = False
    if a.resume:
        saved = torch.load(a.resume, map_location='cpu', weights_only=False)
        if saved.get('pipeline_schema') != 2:
            raise ValueError('Resume requires corrected schema2 checkpoint')
        for key in ('epochs', 'bs', 'd', 'hidden', 'r', 'layers', 'lr', 'seed'):
            if saved['config'].get(key) != getattr(a, key):
                raise ValueError('Resume configuration mismatch: ' + key)
        if saved['planned_steps'] != total or saved['steps_per_epoch'] != steps_per_epoch:
            raise ValueError('Resume data/schedule mismatch')
        for key in ('base_ch','cand_ch','cand_ids'):
            if not torch.equal(saved['model'][key].cpu(), model.state_dict()[key].cpu()):
                raise ValueError('Resume inventory mismatch: ' + key)
        model.load_state_dict(saved['model'])
        opt.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        step = saved['step']; tokens_seen = saved['tokens_seen']
        epochs_completed = saved['epochs_completed']; current_epoch = saved['epoch']
        epoch_tokens = saved['epoch_tokens_seen']; epoch_batches_done = saved['epoch_batches_completed']
        epoch_order = saved['epoch_segment_order']; best = saved['best_val_acc']
        rng.bit_generator.state = saved['numpy_rng']
        torch.set_rng_state(saved['torch_rng'])
        if saved.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        resume_pending = current_epoch > epochs_completed
        print(f'Resume step={step} completed_epochs={epochs_completed} partial_epoch={resume_pending}', flush=True)
    print(f'params={sum(p.numel() for p in model.parameters() if p.requires_grad)} '
          f'sentences={len(train[2])} tokens={len(train[0])} segments={len(starts)} '
          f'steps/epoch={steps_per_epoch} planned={total} punct={npunct} device={device}',
          flush=True)

    def validate_save(reason):
        nonlocal best
        acc, count = evaluate(model, val, a.bs)
        improved = acc > best
        best = max(best, acc)
        state = dict(pipeline_schema=2, model=model.state_dict(), optimizer=opt.state_dict(),
                     scheduler=scheduler.state_dict(), config=vars(a), step=step,
                     epoch=current_epoch, epochs_completed=epochs_completed,
                     tokens_seen=tokens_seen, epoch_tokens_seen=epoch_tokens,
                     epoch_batches_completed=epoch_batches_done,
                     epoch_segment_order=epoch_order, status=status,
                     planned_steps=total, steps_per_epoch=steps_per_epoch,
                     val_acc=acc, best_val_acc=best, validation_tokens=count,
                     elapsed_seconds=time.monotonic() - started,
                     numpy_rng=rng.bit_generator.state,
                     torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        for path in [a.out] + ([a.out + '.best.pt'] if improved else []):
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix + '.tmp')
            torch.save(state, tmp)
            os.replace(tmp, target)
        print(f'validate({reason}) epoch={current_epoch} acc={acc:.6f} best={best:.6f} '
              f'step={step} elapsed={time.monotonic() - started:.0f}s', flush=True)

    try:
        first_epoch = current_epoch if resume_pending else epochs_completed + 1
        for epoch in range(first_epoch, a.epochs + 1):
            current_epoch = epoch
            if not resume_pending:
                epoch_tokens = epoch_batches_done = 0
                epoch_order = np.concatenate(list(epoch_batches(lengths, a.bs, rng)))
            resume_pending = False
            for pos in range(epoch_batches_done * a.bs, len(epoch_order), a.bs):
                if a.max_minutes and time.monotonic() - started >= a.max_minutes * 60:
                    status = 'time_cap'
                    raise SystemExit
                idx = epoch_order[pos:pos + a.bs]
                x, y, ln, ff, gg = make_batch(train, starts, lengths, idx, device)
                emit, trans, mask = model(x, ln, ff, gg)
                loss = sm.nll(emit, trans, mask, y, ln)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite loss; optimizer not updated')
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                opt.step()
                scheduler.step()
                step += 1
                tokens_seen += int(ln.sum())
                epoch_tokens += int(ln.sum())
                epoch_batches_done += 1
                if step % 100 == 0:
                    print(f'step {step}/{total} epoch {epoch} loss {loss.item():.4f} '
                          f'lr {scheduler.get_last_lr()[0]:.6f}', flush=True)
                if a.eval_every and step % a.eval_every == 0:
                    validate_save('interval')
                if a.max_minutes and time.monotonic() - started > a.max_minutes * 60:
                    status = 'time_cap'
                    raise SystemExit
            epochs_completed = epoch
            validate_save('epoch')
        status = 'complete'
    except SystemExit:
        status = status if status != 'running' else 'stopped'
    except Exception:
        status = 'error'
        raise
    finally:
        if status != 'error':
            validate_save(status)
    print('DONE', status, flush=True)


def build_tensors(prefix, min_char_count=100):
    """Character table shared by training and inference.

    Characters occurring in fewer than `min_char_count` inventory strings are
    mapped to dedicated UNK index 1 (padding is 0). This keeps the character embedding small
    instead of paying for thousands of one-off foreign characters.
    """
    import json
    from collections import Counter
    vocab = json.loads(Path(prefix + '_vocab.json').read_text())
    bases = json.loads(Path(prefix + '_bases.json').read_text())
    ctab = np.load(prefix + '_ctab.npy')
    counts = Counter()
    for s in bases:
        counts.update(s)
    for s in vocab:
        counts.update(s)
    chars = sorted(c for c, n in counts.items() if n >= min_char_count)
    c2i = {c: i + 2 for i, c in enumerate(chars)}
    width = 16

    def enc(strings):
        a = np.zeros((len(strings), width), dtype=np.int64)
        for i, s in enumerate(strings):
            for j in range(min(len(s), width)):
                a[i, j] = c2i.get(s[j], 1)
        return torch.from_numpy(a).to(torch.int16)

    cid = ctab.copy()
    cid[0, :] = -1
    return len(chars) + 2, enc([''] + bases), enc(vocab), torch.from_numpy(cid), ctab


if __name__ == '__main__':
    main()
