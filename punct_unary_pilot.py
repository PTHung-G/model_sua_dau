"""Matched unary residual pilot on frozen PunctSylCRF; no test access."""
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import unicodedata as ud
import zlib

import numpy as np
import torch
from torch import nn

from eval_punct_vs_best import FrozenRestorer
from syltrain_punct import load_split, make_batch, segments
from sylmodel_punct import NEG, nll, viterbi

TONE = {c: i for i, c in enumerate('\u0300\u0301\u0303\u0309\u0323', 1)}


def analyze_spelling(word):
    text = ud.normalize('NFD', word.lower())
    marks = [TONE[c] for c in text if c in TONE]
    letter = ud.normalize('NFC', ''.join(c for c in text if c not in TONE))
    # Supported Vietnamese-like syllables only; never discard any candidate.
    valid = (len(marks) <= 1 and all(c in 'abcdefghijklmnopqrstuvwxyzăâêôơưđ' for c in letter)
             and any(c in 'aeiouyăâêôơư' for c in letter))
    if marks:
        host = None
        for c in text:
            if c in TONE and (host is None or host not in 'aeiouy'):
                valid = False
            if not ud.combining(c):
                host = c
    return letter, (marks[0] if marks else 0) if valid else None


class Spec:
    def __init__(self, vocab):
        self.vocab = vocab
        self.ids = {}

    @lru_cache(maxsize=262144)
    def get(self, i):
        if i < 0:
            return -1, 0, 0
        word = self.vocab[i]
        letter, tone = analyze_spelling(word)
        key = ('letter', letter) if tone is not None else ('fallback', i)
        group = self.ids.setdefault(key, len(self.ids))
        flat = zlib.crc32(word.encode('utf-8')) % 6
        return group, tone if tone is not None else flat, flat

    def batch(self, cid):
        ids, inverse = torch.unique(cid, return_inverse=True)
        return torch.tensor([self.get(i) for i in ids.tolist()], device=cid.device)[inverse].unbind(-1)


class MatchedHead(nn.Module):
    def __init__(self, d, rank=24):
        super().__init__()
        self.context = nn.Linear(d, 2 * rank, bias=False)
        self.letter = nn.Linear(d, rank, bias=False)
        self.interaction = nn.Linear(d, rank, bias=False)
        self.category = nn.Embedding(6, rank)
        self.gain = nn.Parameter(torch.zeros(2 * rank))

    def forward(self, ctx, feat, bucket):
        a, b = (self.context(ctx) * self.gain).unsqueeze(2).chunk(2, dim=-1)
        return (a * self.letter(feat) + b * self.interaction(feat) * self.category(bucket)).sum(-1)


def group_mean(ce, group, cid):
    same = (group.unsqueeze(-2) == group.unsqueeze(-1)) & (cid.unsqueeze(-2) >= 0)
    w = same.to(ce.dtype)
    return (w @ ce) / w.sum(-1, keepdim=True).clamp_min(1)


def paired_scores(original, mask, ctx, ce, cid, heads, spec):
    group, tone, flat = spec.batch(cid)
    structured = group_mean(ce, group, cid)
    features = ((ce, flat), (structured, tone))
    # Where one spelling family contains multiple tones, letter-point differences
    # cancel in the structured correction. Original frozen scores remain a residual.
    return [(original + head(ctx, z, bucket)).masked_fill(~mask, NEG)
            for head, (z, bucket) in zip(heads, features)]


def sampled_train(path, count, rng):
    # Keep compressed arrays in native dtype; make_batch converts only selected batches.
    with np.load(str(path) + '.npz') as d:
        base, gold, lengths = d['base'], d['gold'], d['sent'].astype(np.int64)
    starts = np.r_[0, lengths.cumsum()[:-1]]
    rows = np.sort(rng.choice(len(lengths), count, replace=False))
    data = (base, gold, lengths, starts, np.load(str(path)+'_input_features.npy', mmap_mode='r'),
            np.load(str(path)+'_gaps.npy', mmap_mode='r'))
    x, l = segments(lengths[rows], starts[rows])
    return data, x, l, rows


@torch.no_grad()
def evaluate(model, heads, spec, data, dev_rows, device, batch=12):
    result = [Counter() for _ in range(3)]
    comparison = [Counter() for _ in range(3)]
    conf = []
    _, _, lengths, starts, *_ = data
    compact_dev = dev_rows[np.argsort(lengths[dev_rows], kind='stable')]
    for i in range(0, len(compact_dev), batch):
        ix = compact_dev[i:i+batch]
        x, y, ln, f, g = make_batch(data, starts, lengths, ix, device)
        e, t, mask, ctx, ce, cid = model(x, ln, f, g, return_unary_features=True)
        es = [e, *paired_scores(e, mask, ctx, ce, cid, heads, spec)]
        real = mask.any(-1)
        assert mask.gather(-1, y.unsqueeze(-1)).squeeze(-1)[real].all()
        preds = [viterbi(em, t, mask) for em in es]
        ok = [p == y for p in preds]
        latent = mask.sum(-1) > 1
        gold_ids = cid.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        for c, p, z in zip(result, preds, ok):
            c.update(tokens=int(real.sum()), correct=int((z & real).sum()),
                     paragraphs=len(ix), exact=int((z | ~real).all(-1).sum()),
                     ambiguous=int(latent.sum()), ambiguous_correct=int((z & latent).sum()))
            # Compare real Unicode spellings, including unsupported cases.
            errors = real & ~z
            a = gold_ids[errors].tolist()
            b = cid.gather(-1, p.unsqueeze(-1)).squeeze(-1)[errors].tolist()
            c['tone_only_errors'] += sum(analyze_spelling(spec.vocab[a_])[0] == analyze_spelling(spec.vocab[b_])[0]
                                         for a_, b_ in zip(a, b))
        for c, ref, arm in zip(comparison, (ok[0],ok[0],ok[1]), (ok[1],ok[2],ok[2])):
            c.update(fixed=int((~ref & arm & real).sum()), harmed=int((ref & ~arm & real).sum()))
        conf.extend(zip(ix.tolist(), ln.tolist(), ((ok[2] & real).sum(-1)-(ok[1] & real).sum(-1)).tolist()))
    return {'scores': dict(zip(('frozen','flat','factorized'), result)),
            'edits_vs_frozen': dict(zip(('flat','factorized'),comparison[:2])),
            'factorized_vs_flat': comparison[2], 'paired_paragraph_deltas': conf}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory-dir', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=160)
    p.add_argument('--batch', type=int, default=12)
    p.add_argument('--train-rows', type=int, default=3500)
    p.add_argument('--dev-rows', type=int, default=1000)
    p.add_argument('--seed', type=int, default=506)
    p.add_argument('--threads', type=int, default=2)
    a = p.parse_args()
    if min(a.steps,a.batch,a.train_rows,a.dev_rows,a.threads) <= 0:
        p.error('positive sizes required')
    a.out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    device = torch.device('cpu')
    rest = FrozenRestorer('punct', a.checkpoint, a.inventory_dir/'train')
    model = rest.model.requires_grad_(False).eval()
    spec = Spec(rest.vocab)
    heads = [MatchedHead(model.d), MatchedHead(model.d)]
    heads[1].load_state_dict(heads[0].state_dict())
    params = [sum(p.numel() for p in h.parameters()) for h in heads]
    assert params[0] == params[1]
    opts = [torch.optim.AdamW(h.parameters(), lr=2e-3, weight_decay=.01) for h in heads]
    tr, starts, lens, train_rows = sampled_train(a.inventory_dir/'train', a.train_rows, rng)
    val = load_split(a.inventory_dir/'validation')
    dev_rows = np.sort(rng.choice(len(val[2]), a.dev_rows, replace=False))
    order = rng.permutation(len(lens))
    if len(order) < a.steps*a.batch:
        raise ValueError('Need more training segments for sampling without replacement')
    order = order[:a.steps*a.batch].reshape(-1,a.batch)
    with a.checkpoint.open('rb') as ck:
        base_sha = hashlib.file_digest(ck,'sha256').hexdigest()
    (a.out/'manifest.json').write_text(json.dumps(dict(seed=a.seed, torch=str(torch.__version__), base_sha256=base_sha,
        train_rows=train_rows.tolist(), validation_rows=dev_rows.tolist(), batch=a.batch, steps=a.steps,
        lr=.002, head_parameters=params, frozen_parameters=sum(p.numel() for p in model.parameters())), indent=2)+'\n')
    history = []
    for i, idx in enumerate(order):
        x,y,ln,f,g = make_batch(tr, starts, lens, idx, device)
        with torch.no_grad():
            original, t, mask, ctx, ce, cid = model(x, ln, f, g, return_unary_features=True)
        scores = paired_scores(original,mask,ctx,ce,cid,heads,spec)
        if i == 0:
            assert all(torch.equal(s[mask], original[mask]) for s in scores)
            assert all(torch.equal(viterbi(s,t,mask),viterbi(original,t,mask)) for s in scores)
        losses = []
        for opt,s in zip(opts,scores):
            opt.zero_grad(set_to_none=True)
            loss = nll(s,t,mask,y,ln)
            assert torch.isfinite(loss).item()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        history.append(losses)
        if (i+1) % 40 == 0:
            print(f'{i+1}/{a.steps}: rolling CRF NLL (flat, factorized): {np.mean(history[-40:], axis=0)}',flush=True)
    result = evaluate(model, heads, spec, val, dev_rows, device)
    for name,head in zip(('flat','factorized'),heads):
        # Latest is also the sole evaluation point; no dev-based checkpoint selection.
        ck = dict(state=head.state_dict(), base_sha256=base_sha, arm=name, step=a.steps, seed=a.seed)
        torch.save(ck,a.out/f'{name}_latest.pt')
        torch.save(ck,a.out/f'{name}_best.pt')
    result['train_losses'] = history
    (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('train_losses','paired_paragraph_deltas')},indent=2),flush=True)


if __name__ == '__main__':
    main()
