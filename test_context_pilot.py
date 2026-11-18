"""CPU regression tests: identity, packed padding invariance and raw scoring."""
import unittest
from unittest.mock import patch

import torch

import context_pilot as cp
import pilot_arch as pa

CONFIG = dict(syl=192, dim=384, blocks=1, heads=8, ch_dim=16,
              window=8, out_std=0.1, drop=0.0, ff_mult=1, packed_chars=True)


class ContextTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        torch.manual_seed(0)
        self.base = pa.FactorizedDiac(10, 3, 4, 5, **CONFIG).eval()
        self.context = cp.ContextDiac(10, 3, 4, 5, base_state=self.base.state_dict(), **CONFIG).eval()
        self.x = torch.tensor([[[1, 2], [3, 9], [4, 5]]])
        self.cl = torch.tensor([[2, 1, 2]])
        self.mask = torch.ones((1, 3), dtype=torch.bool)

    def test_zero_initial_exact_identity_and_strict_load(self):
        with torch.no_grad():
            a = self.base(self.x, self.cl, self.mask)
            b = self.context(self.x, self.cl, self.mask)
        for x, y in zip(a, b):
            self.assertTrue(torch.equal(x, y))
        self.assertTrue(all(p.requires_grad for p in self.context.parameters()))
        broken = dict(self.base.state_dict())
        broken.pop('norm.weight')
        with self.assertRaises(RuntimeError):
            cp.ContextDiac(10, 3, 4, 5, base_state=broken, **CONFIG)

    def test_padding_invariant_after_nonzero_residual(self):
        # Nonzero branch must remain invariant too, not merely its zero-init bypass.
        torch.nn.init.normal_(self.context.context_proj.weight, std=0.03)
        torch.nn.init.normal_(self.context.context_proj.bias, std=0.03)
        x = torch.full((3, 7, 6), 9, dtype=torch.long)
        cl = torch.ones((3, 7), dtype=torch.long)
        mask = torch.zeros((3, 7), dtype=torch.bool)
        x[0, :3, :2] = self.x[0]
        cl[0, :3] = self.cl[0]
        mask[0, :3] = True
        x[1, :, :2] = torch.tensor([1, 2])
        cl[1] = 2
        mask[1] = True
        # Third row has zero real tokens: clamped packing must still be finite.
        with torch.no_grad():
            alone = self.context(self.x, self.cl, self.mask)
            padded = self.context(x, cl, mask)
            x[0, 3:] = 7  # Changing masked padding cannot affect real outputs.
            changed = self.context(x, cl, mask)
        for a, b, c in zip(alone, padded, changed):
            torch.testing.assert_close(a[0], b[0, :3], atol=3e-6, rtol=3e-6)
            torch.testing.assert_close(b[0, :3], c[0, :3], atol=3e-6, rtol=3e-6)
            self.assertTrue(torch.isfinite(b).all())
            self.assertEqual(int(torch.count_nonzero(b[~mask])), 0)

    def test_raw_validation_unknowns_and_singletons_not_omitted(self):
        payload = dict(cand={'a': ['á', 'a']}, forms=['a'], char2id={'a': 1}, nchar=3)
        rows = [['Á,', 'zzz', 'É!', '.'], ['É'], ['A'], ['!!!']]
        tokens = cp.ev.Tokens(rows, payload['cand'], payload['char2id'])
        class Dummy(torch.nn.Module):
            pass
        # Always choose accented a. Unknown accented é must remain e, hence wrong.
        def choose(model, arm, xc, cl, f, ln, mask, tables):
            return torch.zeros((*f.shape, 2), device=f.device)
        with patch.object(pa, 'candidate_logits', side_effect=choose):
            report = cp.evaluate_model(Dummy(), rows, payload, tokens, None)
        self.assertEqual(report['n_sentences'], 4)
        self.assertEqual(report['sentence_lengths'], [3, 1, 1, 0])
        self.assertEqual(report['sentence_errors'], [1, 1, 1, 0])
        self.assertEqual(report['unknown_letter_tokens'], 3)
        self.assertEqual(report['oracle_dictionary_ceiling']['n_errors'], 2)
        self.assertAlmostEqual(report['token_accuracy'], 0.4)

    def test_reject_holey_sentence_mask(self):
        with self.assertRaises(ValueError):
            self.context(self.x, self.cl, torch.tensor([[True, False, True]]))


if __name__ == '__main__':
    unittest.main()
