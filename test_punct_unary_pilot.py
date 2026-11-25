import itertools
import unittest

import torch

from punct_unary_pilot import MatchedHead, Spec, analyze_spelling, group_mean, paired_scores
from sylmodel_punct import NEG, nll, viterbi


class UnaryPilotTest(unittest.TestCase):
    def test_spelling_and_fallback(self):
        self.assertEqual(analyze_spelling('rối'), ('rôi', 2))
        self.assertEqual(analyze_spelling('rồi'), ('rôi', 1))
        self.assertEqual(analyze_spelling('đồ'), ('đô', 1))
        self.assertEqual(analyze_spelling('rối'), ('rôi', 2))
        self.assertIsNone(analyze_spelling('123')[1])
        self.assertIsNone(analyze_spelling('\u0301')[1])
        spec = Spec(['rối', 'rồi', '123', '456'])
        self.assertEqual(spec.get(0)[0], spec.get(1)[0])
        self.assertNotEqual(spec.get(2)[0], spec.get(3)[0])
        self.assertEqual(spec.get(2)[1], spec.get(2)[2])

    def test_initial_identity_and_padding_invariance(self):
        torch.manual_seed(12)
        spec = Spec(['rối', 'rồi'])
        cid = torch.tensor([[[0, 1, -1], [0, 1, -1]]])
        mask = cid >= 0
        ce = torch.randn(1, 2, 3, 8)
        ctx = torch.randn(1, 2, 8)
        group, _, _ = spec.batch(cid)
        feat = group_mean(ce, group, cid)
        self.assertTrue(torch.equal(feat[:, :, 0], feat[:, :, 1]))
        ce[:, :, -1] *= 1000
        self.assertTrue(torch.equal(feat, group_mean(ce, group, cid)))
        original = torch.randn(1, 2, 3).masked_fill(~mask, NEG)
        t = torch.randn(1, 1, 3, 3)
        heads = [MatchedHead(8), MatchedHead(8)]
        for head, e in zip(heads, paired_scores(original, mask, ctx, ce, cid, heads, spec)):
            self.assertTrue(torch.equal(original, e))
            self.assertTrue(torch.equal(viterbi(e, t, mask), viterbi(original, t, mask)))
            nll(e, t, mask, torch.tensor([[0, 1]]), torch.tensor([2])).backward()
            self.assertGreater(head.gain.grad.abs().sum().item(), 0.0)
        with torch.no_grad():
            heads[1].gain.fill_(1.0)
        e = paired_scores(original, mask, ctx, ce, cid, heads, spec)[1]
        self.assertTrue(torch.all(e[~mask] == NEG))
        best = max(itertools.product(range(2), repeat=2), key=lambda path:
                   float((e[0, 0, path[0]] + e[0, 1, path[1]] + t[0, 0, *path]).detach()))
        self.assertEqual(tuple(viterbi(e, t, mask)[0].tolist()), best)


if __name__ == '__main__':
    unittest.main()
