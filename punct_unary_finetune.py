"""Paired short full-network fine-tuning from the same ViDiCRF checkpoint."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from punct_unary_pilot import FrozenRestorer, MatchedHead, Spec, analyze_spelling, group_mean, sampled_train
from syltrain_punct import load_split, make_batch, segments
from sylmodel_punct import NEG, nll, viterbi

EXPECTED_SHA = 'a9364088b5aa867099753f498302ed0cdc554dddf522f2336d60ba50dde8d520'
NAMES = ('flat', 'letter_tone')


class Arm(nn.Module):
    def __init__(self, base, structured):
        super().__init__()
        self.base = base
        self.head = MatchedHead(base.d)
        self.structured = structured

    def forward(self, x, lengths, features, gaps, spec, skip):
        original, trans, mask, ctx, ce, cid = self.base(x, lengths, features, gaps, return_unary_features=True)
        family, tone, flat = spec.batch(cid)
        feat, bucket = (group_mean(ce, family, cid), tone) if self.structured else (ce, flat)
        e = (skip * original + self.head(ctx, feat, bucket)).masked_fill(~mask, NEG)
        return e, trans, mask, cid


# Hold the pretrained unary at full scale in both arms. It remains trainable.
# V1 can be reproduced from the archived Kaggle package; this is a new pilot.
UNARY_SCALE = 1.0


def make_order(lengths, steps, batch, rng):
    picked = rng.permutation(len(lengths))[:steps * batch]
    if len(picked) != steps * batch:
        raise ValueError('insufficient distinct training segments')
    # Local length bucketing; same batches and dropout seeds across arms.
    out = []
    for chunk in np.array_split(picked, math.ceil(steps / 16)):
        out.extend(chunk[np.argsort(lengths[chunk], kind='stable')].tolist())
    return np.asarray(out).reshape(steps, batch)


@torch.no_grad()
def evaluate(ref, arms, specs, data, rows, device, skip, batch):
    ref.eval()
    for arm in arms:
        arm.eval()
    _, _, lengths, starts, *_ = data
    order = rows[np.argsort(lengths[rows], kind='stable')]
    totals = [Counter() for _ in range(3)]
    changes = [Counter() for _ in range(3)]
    per_row = []
    vocab = specs[0].vocab
    for pos in range(0, len(order), batch):
        ix = order[pos:pos+batch]
        x, y, ln, f, g = make_batch(data, starts, lengths, ix, device)
        e, t, m, _, _, cid = ref(x, ln, f, g, return_unary_features=True)
        outputs = [(e,t,m,cid)] + [a(x,ln,f,g,s,skip) for a,s in zip(arms,specs)]
        real = m.any(-1)
        assert m.gather(-1, y.unsqueeze(-1)).squeeze(-1)[real].all()
        preds, oks = [], []
        gid = cid.gather(-1,y.unsqueeze(-1)).squeeze(-1)
        for c,(ae,at,am,ac) in zip(totals,outputs):
            assert torch.equal(am,m) and torch.equal(ac,cid)
            p = viterbi(ae,at,am)
            ok = p.eq(y) & real
            err = real & ~ok
            pid = cid.gather(-1,p.unsqueeze(-1)).squeeze(-1)
            c.update(tokens=int(real.sum()), correct=int(ok.sum()), paragraphs=len(ix),
                     exact=int((ok | ~real).all(-1).sum()), ambiguous=int((m.sum(-1)>1).sum()),
                     ambiguous_correct=int((ok & (m.sum(-1)>1)).sum()),
                     tone_only_errors=sum(analyze_spelling(vocab[a])[0] == analyze_spelling(vocab[b])[0]
                                          for a,b in zip(gid[err].tolist(),pid[err].tolist())))
            preds.append(p)
            oks.append(ok)
        for c,a,b in zip(changes, (0,0,1), (1,2,2)):
            c.update(fixed=int((~oks[a]&oks[b]).sum()), harmed=int((oks[a]&~oks[b]).sum()),
                     changed=int(((preds[a]!=preds[b])&real).sum()))
        per_row.extend(zip(ix.tolist(),ln.tolist(),(oks[2].sum(-1)-oks[1].sum(-1)).tolist()))
    return dict(scores=dict(zip(('frozen',*NAMES),totals)), changes=dict(zip(('flat_vs_frozen','letter_tone_vs_frozen','letter_tone_vs_flat'),changes)), paired_paragraph_deltas=per_row)


def write_json(path, data):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n', encoding='utf-8')
    tmp.replace(path)


def save(path, arm, optimizer, meta, step):
    tmp = path.with_suffix('.tmp')
    # Lattice is loaded from the pinned parent; save every learned parameter.
    torch.save(dict(parameters={n:p.detach().cpu() for n,p in arm.named_parameters()},
                    optimizer=optimizer.state_dict(), step=step, config=meta),tmp)
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('inventory-dir','checkpoint','out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    for n, default in [('steps',480),('batch',16),('train-rows',9500),('dev-rows',1000),('seed',7311),('threads',2)]:
        p.add_argument('--'+n,type=int,default=default)
    p.add_argument('--extend-rows', type=int, default=0, help='sample this many extra paragraphs, disjoint from the original pool')
    p.add_argument('--milestones', type=int, nargs='+', default=None)
    a = p.parse_args()
    milestones = sorted(set(a.milestones or [a.steps]) | {a.steps})
    if min(a.steps, a.batch, a.train_rows, a.dev_rows, a.threads, *milestones) < 1 or milestones[-1] > a.steps:
        p.error('invalid sizes or milestones')
    if (a.extend_rows and a.steps <= 480) or a.extend_rows < 0:
        p.error('extension requires >480 total steps and a positive extra pool')
    torch.set_num_threads(a.threads)
    dev = torch.device(a.device)
    if a.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Kaggle GPU not visible')
    with a.checkpoint.open('rb') as f:
        sha = hashlib.file_digest(f,'sha256').hexdigest()
    if sha != EXPECTED_SHA:
        raise ValueError(f'Unexpected base checkpoint: {sha}')
    a.out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    rest = FrozenRestorer('punct',a.checkpoint,a.inventory_dir/'train')
    ref = rest.model.requires_grad_(False).eval().to(dev)
    import copy
    arms = []
    for structured in (False, True):
        base = copy.deepcopy(ref).requires_grad_(True)
        # Share immutable inventory, saving GPU space; model parameters remain distinct.
        for name, buf in ref.named_buffers():
            setattr(base,name,buf)
        arms.append(Arm(base,structured).to(dev))
    arms[1].head.load_state_dict(arms[0].head.state_dict())
    specs = [Spec(rest.vocab),Spec(rest.vocab)]
    opts = [torch.optim.AdamW([dict(params=m.base.parameters(),lr=5e-5),dict(params=m.head.parameters(),lr=1e-3)], weight_decay=.01) for m in arms]
    counts = [sum(p.numel() for p in arm.parameters() if p.requires_grad) for arm in arms]
    assert counts[0] == counts[1]
    tr, starts, lens, train_rows = sampled_train(a.inventory_dir/'train',a.train_rows,rng)
    val = load_split(a.inventory_dir/'validation')
    rows = np.sort(rng.choice(len(val[2]),a.dev_rows,replace=False))
    prefix_steps = 480 if a.extend_rows else a.steps
    order = make_order(lens,prefix_steps,a.batch,rng)
    if a.extend_rows:
        lengths, offsets = tr[2:4]
        eligible = np.ones(len(lengths), dtype=bool)
        eligible[train_rows] = False
        extra_rows = np.sort(rng.choice(np.flatnonzero(eligible), a.extend_rows, replace=False))
        extra_starts, extra_lens = segments(lengths[extra_rows], offsets[extra_rows])
        extra_order = make_order(extra_lens,a.steps-prefix_steps,a.batch,rng)
        assert np.unique(order).size == order.size and np.unique(extra_order).size == extra_order.size
        order = np.concatenate([order, extra_order+len(lens)])
        starts, lens = np.r_[starts,extra_starts], np.r_[lens,extra_lens]
    else:
        extra_rows = np.array([], dtype=np.int64)
    # Record exact token exposure, with no padded tokens.
    tokens_by_step = np.cumsum(lens[order].sum(axis=1))
    meta = dict(base_sha256=sha,torch=str(torch.__version__), device=torch.cuda.get_device_name() if a.device=='cuda' else 'cpu',
                seed=a.seed, steps=a.steps, batch=a.batch, trainable_params=counts,
                head_parameters=sum(p.numel() for p in arms[0].head.parameters()),
                train_rows=train_rows.tolist(), dev_rows=rows.tolist(), order=order.tolist(),
                original_unary_scale=UNARY_SCALE, learning_rates=[5e-5,1e-3],
                protocol='v3_data_extension' if a.extend_rows else 'v2_constant_pretrained_unary',
                extra_rows=extra_rows.tolist(), milestones=milestones,
                tokens_at_milestones={str(i):int(tokens_by_step[i-1]) for i in milestones})
    write_json(a.out/'manifest.json',meta)
    print('Initial dev replay',flush=True)
    initial = evaluate(ref,arms,specs,val,rows,dev,1.,a.batch)
    assert initial['changes']['letter_tone_vs_flat']['changed'] == 0
    assert all(initial['changes'][key]['changed']==0 for key in ('flat_vs_frozen','letter_tone_vs_frozen'))
    write_json(a.out/'initial.json',initial)
    t0 = time.monotonic()
    losses = []
    best = dict.fromkeys(NAMES, (-1, 0))
    for i, idx in enumerate(order,1):
        x,y,ln,f,g = make_batch(tr,starts,lens,idx,dev)
        pair_loss = []
        for name, arm, spec, opt in zip(NAMES,arms,specs,opts):
            torch.manual_seed(a.seed+i)
            arm.train()
            opt.zero_grad(set_to_none=True)
            e,t,m,_ = arm(x,ln,f,g,spec,UNARY_SCALE)
            loss = nll(e,t,m,y,ln)
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f'{name}: non-finite at {i}')
            loss.backward()
            if i == 1:
                assert arm.base.emit_w.weight.grad.abs().sum() > 0
                assert arm.base.lstm.weight_ih_l0.grad.abs().sum() > 0
                assert arm.base.trans_p.weight.grad.abs().sum() > 0
                assert arm.head.gain.grad.abs().sum() > 0
            torch.nn.utils.clip_grad_norm_(arm.parameters(),1.0,error_if_nonfinite=True)
            opt.step()
            pair_loss.append(loss.item())
        losses.append(pair_loss)
        if i % 40 == 0 or i == a.steps:
            print(f'{i}/{a.steps} skip={UNARY_SCALE:.3f} mean_nll={np.mean(losses[-40:],axis=0)} elapsed={time.monotonic()-t0:.0f}s',flush=True)
        if i % 120 == 0 or i == a.steps:
            for name,m,o in zip(NAMES,arms,opts):
                save(a.out/f'{name}_latest.pt',m,o,meta,i)
        if i in milestones:
            result = evaluate(ref,arms,specs,val,rows,dev,UNARY_SCALE,a.batch)
            for name,m,o in zip(NAMES,arms,opts):
                score = result['scores'][name]['correct']
                if score > best[name][0]:
                    best[name] = (score,i)
                    save(a.out/f'{name}_best.pt',m,o,meta,i)
            write_json(a.out/f'result_step_{i}.json',result)
            print(f'Dev at {i} after {tokens_by_step[i-1]} tokens: {result["scores"]}',flush=True)
    result['train_losses'] = losses
    result['best_steps'] = {name:step for name,(_,step) in best.items()}
    # Verify the exported inference parameters and replay dev from the saved weights.
    checkpoint_hashes = {}
    for name, arm in zip(NAMES, arms):
        for label in ('best', 'latest'):
            path = a.out/f'{name}_{label}.pt'
            ck = torch.load(path, map_location='cpu', weights_only=True)
            expected = dict(arm.named_parameters())
            step = a.steps if label == 'latest' else best[name][1]
            assert ck['step'] == step and ck['config'] == meta
            assert ck['parameters'].keys() == expected.keys()
            for key, tensor in ck['parameters'].items():
                assert torch.isfinite(tensor).all() and tensor.shape == expected[key].shape, key
                if step == a.steps:
                    assert torch.equal(tensor, expected[key].detach().cpu()), key
            with path.open('rb') as handle:
                checkpoint_hashes[path.name] = hashlib.file_digest(handle,'sha256').hexdigest()
        with torch.no_grad():
            for key, param in expected.items():
                param.copy_(ck['parameters'][key])
    replay = evaluate(ref,arms,specs,val,rows,dev,UNARY_SCALE,a.batch)
    assert all(result[k] == v for k,v in replay.items())
    write_json(a.out/'verification.json',dict(status='PASS', checkpoint_sha256=checkpoint_hashes,
                                               initial_predictions_identical=True, final_dev_replay_exact=True))
    write_json(a.out/'result.json',result)
    print(json.dumps({k:result[k] for k in ('scores','changes')},indent=2),flush=True)


if __name__ == '__main__':
    main()
