"""CPU tests for the loss-only pilot: no dataset or checkpoint needed."""
import unittest

import torch
from torch.nn import functional as F

import sentence_loss_pilot as sl


class SentenceLossTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(0)

    def test_finite_gradients_both_arms(self):
        for arm in sl.ARMS:
            logits = torch.randn(3, 7, 4, requires_grad=True)
            target = torch.rand(3, 7) > 0.25
            y = torch.randint(0, 4, (3, 7))
            loss, parts = sl.loss_components(logits, y, target, arm)
            self.assertTrue(all(torch.isfinite(v) for v in parts.values()))
            loss.backward()
            self.assertTrue(torch.isfinite(logits.grad).all())
            self.assertGreater(logits.grad[target].abs().sum().item(), 0)
            self.assertEqual(logits.grad[~target].abs().sum().item(), 0)

    def test_masks_invalid_labels_and_empty_sentences_safe(self):
        logits = torch.randn(3, 4, 3, requires_grad=True)
        mask = torch.tensor([[True, True, True, False], [True, True, False, False], [False]*4])
        ok = torch.tensor([[True, False, True, False], [True, False, False, False], [False]*4])
        ln = torch.tensor([[2, 3, 1, 0], [2, 2, 0, 0], [0]*4])
        y = torch.tensor([[1, -1, 0, -1], [2, -1, -1, -1], [-1]*4])
        target = sl.training_target(mask, ok, ln)
        self.assertEqual(target.tolist(), [[True, False, False, False], [True, False, False, False], [False]*4])
        loss, _ = sl.loss_components(logits, y, target, 'sentence_hard')
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(logits.grad[~target].abs().sum().item(), 0)
        # Arbitrary impossible labels and NaN logits OUTSIDE the target remain irrelevant.
        changed = logits.detach().clone()
        changed[~target] = float('nan')
        y[~target] = -100000
        other, _ = sl.loss_components(changed, y, target, 'sentence_hard')
        torch.testing.assert_close(loss, other)

    def test_original_singleton_fallback_and_empty_rejection(self):
        mask = torch.tensor([[True, True, False]])
        ok = torch.tensor([[True, False, False]])
        self.assertTrue(torch.equal(sl.training_target(mask, ok, torch.ones_like(mask)), mask & ok))
        with self.assertRaises(RuntimeError):
            sl.training_target(mask, torch.zeros_like(mask), torch.ones_like(mask))

    def test_sentence_balancing_and_formula(self):
        # One hard token in sentence 0, four easy tokens in sentence 1.
        logits = torch.tensor([[[0., 3.]]*4, [[3., 0.]]*4])
        y = torch.zeros((2, 4), dtype=torch.long)
        target = torch.tensor([[True, False, False, False], [True]*4])
        loss, p = sl.loss_components(logits, y, target, 'sentence_hard')
        hard = F.cross_entropy(logits[0, :1], y[0, :1])
        easy = F.cross_entropy(logits[1], y[1])
        torch.testing.assert_close(p['token_mean'], (hard + 4*easy)/5)
        torch.testing.assert_close(p['sentence_mean'], (hard + easy)/2)
        torch.testing.assert_close(p['sentence_surrogate'], (hard + easy)/2)
        torch.testing.assert_close(loss, 0.5*(hard + 4*easy)/5 + 0.5*(hard + easy)/2)

    def test_topk_focus_ceil_and_differentiability(self):
        # n=5 -> ceil(1.25)=2, so precisely the two highest CE are selected.
        logits = torch.tensor([[[4., 0.], [3., 0.], [2., 0.], [0., 2.], [0., 4.]]], requires_grad=True)
        y = torch.zeros((1, 5), dtype=torch.long)
        target = torch.ones_like(y, dtype=torch.bool)
        _, parts = sl.loss_components(logits, y, target, 'sentence_hard')
        ce = F.cross_entropy(logits[0], y[0], reduction='none')
        torch.testing.assert_close(parts['sentence_top_quartile_mean'], ce[-2:].mean())
        self.assertGreater(parts['sentence_top_quartile_mean'].item(), parts['sentence_mean'].item())
        parts['sentence_top_quartile_mean'].backward()
        self.assertEqual(logits.grad[0, :3].abs().sum().item(), 0)
        self.assertGreater(logits.grad[0, -2:].abs().sum().item(), 0)

    def test_baseline_loss_and_gradients_exactly_original(self):
        x = torch.randn(3, 5, 7, requires_grad=True)
        z = x.detach().clone().requires_grad_()
        y = torch.randint(0, 7, (3, 5))
        mask = torch.rand(3, 5) > 0.2
        ok = torch.rand(3, 5) > 0.2
        ln = torch.randint(1, 5, (3, 5))
        target = sl.training_target(mask, ok, ln)
        original = F.cross_entropy(x[target], y[target])
        control, _ = sl.loss_components(z, y, target, 'token_ce')
        self.assertTrue(torch.equal(original, control))
        original.backward()
        control.backward()
        self.assertTrue(torch.equal(x.grad, z.grad))

    def test_defaults_and_exact_smoke_sizes(self):
        args = sl.parse_args([])
        self.assertEqual((args.steps, args.train_sents, args.val_sents, args.seed, args.lr),
                         (3000, 100000, 3000, 0, 1e-4))
        args = sl.parse_args(['--smoke', '--steps', '1', '--train-sents', '1'])
        self.assertEqual((args.steps, args.train_sents, args.val_sents), (2, 128, 20))


if __name__ == '__main__':
    unittest.main()
