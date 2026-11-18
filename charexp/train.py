import argparse, json, math, os, random, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import CharTransformer, BiLSTMCRF

p = argparse.ArgumentParser()
p.add_argument('--model', required=True, choices=['transformer','bilstmcrf'])
p.add_argument('--steps', type=int, default=6000)
p.add_argument('--bs', type=int, default=24)
p.add_argument('--lr', type=float, default=1e-3)
p.add_argument('--maxlen', type=int, default=160)
p.add_argument('--eval-lines', type=int, default=2000)
p.add_argument('--eval-every', type=int, default=1000)
p.add_argument('--threads', type=int, default=3)
p.add_argument('--out', required=True)
p.add_argument('--smoke', type=int, default=0)
a = p.parse_args()
torch.set_num_threads(a.threads)
torch.manual_seed(0); random.seed(0); np.random.seed(0)

V = len(json.load(open('vocab.json')))

def load(name):
    d = np.load(f'{name}.npz')
    return d['src'], d['tgt'], d['lens']

tr_src, tr_tgt, tr_len = load('train')
va_src, va_tgt, va_len = load('val')
print(f'V={V} train_lines={len(tr_len)}', flush=True)

def make_batches(src, tgt, lens, bs, maxlen, shuffle=True, limit=None):
    idx = [i for i, l in enumerate(lens) if 8 <= l <= maxlen]
    idx.sort(key=lambda i: lens[i])
    batches = [idx[i:i+bs] for i in range(0, len(idx), bs)]
    if limit: batches = batches[:limit]
    if shuffle: random.shuffle(batches)
    off = np.concatenate([[0], np.cumsum(lens)])
    for b in batches:
        L = max(int(lens[i]) for i in b)
        xs = np.zeros((len(b), L), np.int64); ys = np.zeros((len(b), L), np.int64)
        ls = np.zeros(len(b), np.int64)
        for k, i in enumerate(b):
            s, e = off[i], off[i] + lens[i]
            xs[k, :lens[i]] = src[s:e]; ys[k, :lens[i]] = tgt[s:e]; ls[k] = lens[i]
        yield (torch.from_numpy(xs), torch.from_numpy(ys), torch.from_numpy(ls))

def build_model():
    if a.model == 'transformer':
        return CharTransformer(V, d=128, layers=6, heads=8, maxlen=256)
    return BiLSTMCRF(V, emb=256, hidden=128, layers=2)

model = build_model()
nparam = sum(x.numel() for x in model.parameters())
print(f'{a.model} params={nparam:,}', flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
steps = a.smoke if a.smoke else a.steps
warm = max(1, steps // 20)
def lr_at(t):
    if t < warm: return a.lr * (t + 1) / warm
    prog = (t - warm) / max(1, steps - warm)
    return a.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * prog)))

@torch.no_grad()
def evaluate(src, tgt, lens, n_lines, maxlen):
    model.eval()
    idx = [i for i, l in enumerate(lens) if 8 <= l <= maxlen][:n_lines]
    off = np.concatenate([[0], np.cumsum(lens)])
    cor = tot = 0; sent_ok = sent_tot = 0
    B = 32
    for k in range(0, len(idx), B):
        b = idx[k:k+B]
        L = max(int(lens[i]) for i in b)
        xs = np.zeros((len(b), L), np.int64); ys = np.zeros((len(b), L), np.int64)
        ls = np.zeros(len(b), np.int64)
        for j, i in enumerate(b):
            s, e = off[i], off[i]+lens[i]
            xs[j,:lens[i]] = src[s:e]; ys[j,:lens[i]] = tgt[s:e]; ls[j] = lens[i]
        x, y, l = torch.from_numpy(xs), torch.from_numpy(ys), torch.from_numpy(ls)
        pred = model(x, lens=l) if a.model == 'bilstmcrf' else model(x).argmax(-1)
        m = torch.arange(L).unsqueeze(0) < l.unsqueeze(1)
        cor += ((pred == y) & m).sum().item(); tot += m.sum().item()
        ok = (((pred == y) | ~m).all(1))
        sent_ok += ok.sum().item(); sent_tot += len(b)
    model.train()
    return 100*cor/tot, 100*sent_ok/sent_tot

it = make_batches(tr_src, tr_tgt, tr_len, a.bs, a.maxlen)
t0 = time.time(); run = 0.0; seen = 0
for step in range(steps):
    try: x, y, l = next(it)
    except StopIteration:
        it = make_batches(tr_src, tr_tgt, tr_len, a.bs, a.maxlen); x, y, l = next(it)
    for g in opt.param_groups: g['lr'] = lr_at(step)
    if a.model == 'bilstmcrf':
        mask = (torch.arange(x.size(1)).unsqueeze(0) < l.unsqueeze(1)).float()
        e, loss = model(x, y, l, mask)
    else:
        out = model(x)
        loss = torch.nn.functional.cross_entropy(
            out.reshape(-1, V), y.reshape(-1), ignore_index=0)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    run += loss.item(); seen += 1
    if (step+1) % 50 == 0:
        el = time.time()-t0
        print(f'step {step+1}/{steps} loss {run/seen:.4f} '
              f'{el/(step+1):.3f}s/step eta {(steps-step-1)*el/(step+1)/60:.1f}min', flush=True)
        run = 0.0; seen = 0
        if a.smoke: break
    if not a.smoke and (step+1) % a.eval_every == 0:
        acc, sent = evaluate(va_src, va_tgt, va_len, 1000, a.maxlen)
        print(f'  [val @ {step+1}] token {acc:.2f}%  sentence {sent:.2f}%', flush=True)

va_acc, va_sent = evaluate(va_src, va_tgt, va_len, a.eval_lines, a.maxlen)
print(f'FINAL val token {va_acc:.2f}%  sentence {va_sent:.2f}%', flush=True)
if not a.smoke:
    torch.save({'model': model.state_dict(), 'model_type': a.model, 'V': V,
                'params': nparam, 'steps': steps, 'bs': a.bs, 'lr': a.lr,
                'maxlen': a.maxlen, 'val_token': va_acc, 'val_sent': va_sent},
               a.out)
    print('saved', a.out, flush=True)
