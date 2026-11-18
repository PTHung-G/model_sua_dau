"""Huan luyen CRF muc am tiet tren du lieu da dung boi sylprep.py.

Gom cau theo do dai de giam dem: cau trung binh 21 am tiet nen khong can dem
den 64. Them gioi han thoi gian de kernel tu dung va van danh gia, luu lai.
"""
import argparse, json, math, time
import numpy as np
import torch
import torch.nn as nn

import sylmodel as sm

CHW = 16
MAXL = 64


def build_tensors(prefix):
    vocab = json.load(open(prefix + '_vocab.json'))
    bases = json.load(open(prefix + '_bases.json'))
    ctab = np.load(prefix + '_ctab.npy')
    chars = sorted({c for s in vocab for c in s} | {c for s in bases for c in s})
    c2i = {c: i + 1 for i, c in enumerate(chars)}

    def enc(strings):
        a = np.zeros((len(strings), CHW), dtype=np.int64)
        for i, s in enumerate(strings):
            for j in range(min(len(s), CHW)):
                a[i, j] = c2i[s[j]]
        return torch.from_numpy(a)

    cid = ctab.copy()
    cid[0, :] = -1
    return len(chars) + 1, enc([''] + bases), enc(vocab), torch.from_numpy(cid), ctab


def load_split(prefix):
    d = np.load(prefix + '.npz')
    sent = d['sent'].astype(np.int64)
    off = np.zeros(len(sent), dtype=np.int64)
    np.cumsum(sent[:-1], out=off[1:])
    return d['base'].astype(np.int64), d['gold'].astype(np.int64), sent, off


def make_batch(base, gold, sent, off, idx, L, rng):
    B = len(idx)
    x = np.zeros((B, L), dtype=np.int64)
    y = np.zeros((B, L), dtype=np.int64)
    ln = np.zeros(B, dtype=np.int64)
    for j, i in enumerate(idx):
        n = int(sent[i])
        st = int(rng.integers(0, n - L + 1)) if n > L else 0
        m = min(n - st, L)
        x[j, :m] = base[off[i] + st: off[i] + st + m]
        y[j, :m] = gold[off[i] + st: off[i] + st + m]
        ln[j] = m
    return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(ln)


@torch.no_grad()
def evaluate(model, base, gold, sent, off, idx, ctab, n_cap=20000, bs=128):
    model.eval()
    device = next(model.parameters()).device
    idx = idx[:n_cap]
    order = idx[np.argsort(sent[idx], kind='stable')]
    ok = tot = aok = atot = 0
    for k in range(0, len(order), bs):
        sub = order[k:k + bs]
        L = int(sent[sub].max())
        x, y, ln = make_batch(base, gold, sent, off, sub, L, np.random.default_rng(0))
        emit, trans, mask = model(x.to(device), ln.to(device))
        pred = sm.viterbi(emit, trans, mask).cpu()
        m = torch.arange(L).unsqueeze(0) < ln.unsqueeze(1)
        hit = (pred == y) & m
        ok += int(hit.sum()); tot += int(m.sum())
        n_cand = (torch.from_numpy(ctab)[x] >= 0).sum(-1)
        amb = m & (n_cand > 1)
        aok += int((hit & amb).sum()); atot += int(amb.sum())
    model.train()
    return ok / max(tot, 1), aok / max(atot, 1), tot, atot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', default='syl/train')
    ap.add_argument('--steps', type=int, default=200000)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1.5e-3)
    ap.add_argument('--d', type=int, default=192)
    ap.add_argument('--r', type=int, default=32)
    ap.add_argument('--hidden', type=int, default=192)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--eval-every', type=int, default=5000)
    ap.add_argument('--val-sents', type=int, default=20000)
    ap.add_argument('--max-minutes', type=float, default=0)
    ap.add_argument('--out', default='sylcrf_ordered_latest.pt')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if dev == 'cuda':
        torch.backends.cudnn.benchmark = True

    nchar, base_ch, cand_ch, cid, ctab = build_tensors(a.prefix)
    base, gold, sent, off = load_split(a.prefix)
    nv = a.val_sents
    tr_idx = np.arange(0, len(sent) - nv)
    va_idx = np.arange(len(sent) - nv, len(sent))
    print('cau train %d | cau val %d | am tiet %d | ky tu %d | ung vien %d | thiet bi %s' %
          (len(tr_idx), len(va_idx), len(base), nchar, len(cand_ch), dev), flush=True)
    if a.smoke:
        tr_idx, va_idx = tr_idx[:2000], va_idx[:300]

    model = sm.SylCRF(nchar, base_ch, cand_ch, cid, d=a.d, r=a.r,
                      hidden=a.hidden, layers=a.layers).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    print('tham so', nparam, flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    warm = max(1, a.steps // 20)
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else
        0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, a.steps - warm))))

    rng = np.random.default_rng(0)
    lorder = np.argsort(np.minimum(sent[tr_idx], MAXL), kind='stable')
    t0 = time.time()
    run = 0.0
    done = 0
    best_acc = -1.0
    def save_checkpoint(path, step, val_acc=None):
        from pathlib import Path
        import os
        target = Path(path)
        temporary = target.with_suffix(target.suffix + '.tmp')
        torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(),
                        scheduler=sch.state_dict(), step=step, config=vars(a),
                        val_acc=val_acc, numpy_rng=rng.bit_generator.state,
                        torch_rng=torch.get_rng_state()), temporary)
        os.replace(temporary, target)
    model.train()
    for step in range(1, a.steps + 1):
        p = int(rng.integers(0, len(lorder) - a.bs))
        sel = tr_idx[lorder[p:p + a.bs]]
        L = int(min(MAXL, sent[sel].max()))
        x, y, ln = make_batch(base, gold, sent, off, sel, L, rng)
        x, y, ln = x.to(dev), y.to(dev), ln.to(dev)
        emit, trans, mask = model(x, ln)
        loss = sm.nll(emit, trans, mask, y, ln)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()
        run += float(loss.detach()); done = step
        if step % 200 == 0:
            el = time.time() - t0
            print('  step %d loss %.4f | %.0f s | %.0f am tiet/s' %
                  (step, run / 200, el, step * a.bs * 20 / el), flush=True)
            run = 0.0
        if step % a.eval_every == 0:
            acc, amb, tot, atot = evaluate(model, base, gold, sent, off, va_idx, ctab)
            print('  [val] step %d | am tiet %.2f%% | nhieu ung vien %.2f%% (n=%d)' %
                  (step, 100 * acc, 100 * amb, atot), flush=True)
            save_checkpoint(a.out, step, acc)
            if acc > best_acc:
                best_acc = acc
                save_checkpoint(a.out + '.best.pt', step, acc)
        if a.max_minutes and (time.time() - t0) / 60 > a.max_minutes:
            print('  dung theo gioi han thoi gian o step', step, flush=True)
            break

    print('CUOI (step %d):' % done, flush=True)
    for name, pref in [('val', a.prefix)]:
        b2, g2, s2, o2 = (base, gold, sent, off) if pref == a.prefix else load_split(pref)
        idx = va_idx if name == 'val' else np.arange(len(s2))
        acc, amb, tot, atot = evaluate(model, b2, g2, s2, o2, idx, ctab)
        print('  %-9s am tiet %.2f%% | nhieu ung vien %.2f%% (n=%d)' %
              (name, 100 * acc, 100 * amb, atot), flush=True)
        save_checkpoint(a.out, done, acc)
        if acc > best_acc:
            best_acc = acc
            save_checkpoint(a.out + '.best.pt', done, acc)
    print('tong %.1f phut | %d tham so | %d buoc' % ((time.time() - t0) / 60, nparam, done))


if __name__ == '__main__':
    main()
