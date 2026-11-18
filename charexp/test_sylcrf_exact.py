"""Exact enumeration and padding checks for the experimental lattice CRF."""
import itertools
import torch
import sylmodel as sm

torch.set_num_threads(2)
torch.manual_seed(17)
for lengths in ([1, 3, 4], [4, 2, 1]):
    b, length, k = 3, 4, 3
    emit = torch.randn(b, length, k, dtype=torch.float64, requires_grad=True)
    trans = torch.randn(b, length-1, k, k, dtype=torch.float64, requires_grad=True)
    valid = torch.rand(b, length, k) > .4
    valid[..., 0] = True
    mask = valid & (torch.arange(length)[None, :, None] < torch.tensor(lengths)[:, None, None])
    scores = emit.masked_fill(~valid, sm.NEG)
    z = sm.log_partition(scores, trans, mask)
    pred = sm.viterbi(scores, trans, mask)
    gold = torch.zeros(b, length, dtype=torch.long)
    gs = sm.gold_score(scores, trans, mask, gold)
    for i, n in enumerate(lengths):
        paths = list(itertools.product(range(k), repeat=n))
        values = torch.stack([sum(scores[i,t,p[t]] for t in range(n)) +
                              sum(trans[i,t,p[t],p[t+1]] for t in range(n-1)) for p in paths])
        assert torch.allclose(z[i], values.logsumexp(0), atol=1e-10)
        best = paths[int(values.argmax())]
        assert tuple(pred[i,:n].tolist()) == best
        assert torch.allclose(gs[i], sum(scores[i,t,0] for t in range(n)) +
                              sum(trans[i,t,0,0] for t in range(n-1)))
    loss = sm.nll(scores, trans, mask, gold, torch.tensor(lengths))
    loss.backward()
    assert torch.isfinite(emit.grad).all() and torch.isfinite(trans.grad).all()

base_ch = torch.tensor([[0,0,0],[1,2,0],[2,1,0],[3,2,1]])
cand_ch = torch.tensor([[1,2,0],[2,1,0],[3,2,1]])
cid = torch.tensor([[-1,-1],[0,1],[1,2],[0,2]], dtype=torch.int32)
model = sm.SylCRF(4,base_ch,cand_ch,cid,d=16,r=4,hidden=12,layers=1,drop=0)
model.eval()
x = torch.tensor([[1,2,3],[2,0,0]])
lens = torch.tensor([3,1])
e,t,m = model(x,lens)
e1,t1,m1 = model(x[1:,:1],lens[1:])
assert torch.allclose(e[1,:1], e1[0], atol=1e-6), 'padding changed predictions'
assert not torch.allclose(model.encode_chars(cand_ch[:1]),model.encode_chars(cand_ch[1:2])), 'order lost'
loss = sm.nll(e,t,m,torch.zeros(2,3,dtype=torch.long),lens)
loss.backward()
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
print('PASS: exact partition, gold score, Viterbi, variable-length padding, order, finite gradients')
