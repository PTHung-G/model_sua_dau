"""CPU, synthetic-checkpoint regression tests. No training/data/network required."""
import copy
import tempfile
from pathlib import Path
import unittest

import torch
import evaluate as ev
import pilot_arch as pa
import hybrid_rerank_model as hm


CFG = dict(dim=16, heads=4, layers=1, feedforward=32, dropout=0.0,
           char_embedding=8, query_chunk=2)


def payload_fixture():
    config = dict(dim=16, heads=4, syl=8, ch_dim=4, blocks=1, window=8,
                  drop=0.0, out_std=0.01, packed_chars=True)
    base = pa.FactorizedDiac(9, 3, 2, 4, **config)
    return dict(config=config, model=base.state_dict(), nchar=9, max_c=3,
                signatures=dict(modifier=[(), ('x',)], tone=[(), ('a',), ('b',), ('c',)]),
                cand={'abcd': ['abcd', 'abcđ', 'abcá'], 'ab': ['ab', 'áb']},
                forms=['abcd', 'ab'], char2id={'a': 1, 'b': 2, 'c': 3, 'd': 4},
                tone_table=torch.tensor([[0, 1, 2], [0, 3, 0]]),
                mod_table=torch.tensor([[0, 1, 0], [0, 0, 0]]))


def inputs():
    xc = torch.tensor([[[1, 2, 3, 4], [1, 2, 0, 0]],
                       [[1, 2, 0, 0], [8, 8, 8, 8]]])
    cl = torch.tensor([[4, 2], [2, 1]])
    f = torch.tensor([[0, 1], [1, 0]])
    ln = torch.tensor([[3, 2], [2, 0]])
    mask = torch.tensor([[True, True], [True, False]])
    return xc, cl, f, ln, mask


class HybridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(91)
        self.payload = payload_fixture()

    def model(self, arm):
        return hm.HybridReranker(self.payload, arm, CFG).eval()

    @staticmethod
    def nonzero_head(model):
        with torch.no_grad():
            model.scorer[-1].weight.normal_(std=0.2)

    def test_exact_identity_both_arms_and_frozen_eval(self):
        for arm in hm.CONFIGS:
            m = self.model(arm).train()
            self.assertFalse(m.base.training)
            xs = inputs()
            expected = m.frozen(*xs)[1]
            self.assertTrue(torch.equal(m.logits(*xs), expected))
            self.assertTrue(all(not p.requires_grad for p in m.base.parameters()))
            m.logits(*xs)[xs[-1]].sum().backward()
            self.assertTrue(all(p.grad is None for p in m.base.parameters()))

    def test_all_modules_gradient_after_nonzero_head(self):
        for arm in hm.CONFIGS:
            m = self.model(arm)
            self.nonzero_head(m)
            result = m.logits(*inputs())
            result[result != ev.NEG].square().sum().backward()
            for name, p in m.named_parameters():
                if p.requires_grad:
                    self.assertIsNotNone(p.grad, f'{arm}: {name} not used')
                    self.assertGreater(p.grad.abs().sum().item(), 0, f'{arm}: {name} zero gradient')

    def test_token_ce_gradients_after_nonzero_head(self):
        for arm in hm.CONFIGS:
            m = self.model(arm)
            self.nonzero_head(m)
            xs = inputs()
            scores = m.logits(*xs)
            target = torch.tensor([[2, 1], [0, 0]])
            torch.nn.functional.cross_entropy(scores[xs[-1]], target[xs[-1]]).backward()
            self.assertIsNone(m.key.bias)
            self.assertIsNone(m.scorer[-1].bias)
            # Check actual token-CE signal, not a loss on absolute logit offsets.
            for name, p in m.named_parameters():
                if p.requires_grad:
                    self.assertIsNotNone(p.grad, f'{arm}: {name} not used by CE')
                    self.assertGreater(p.grad.abs().sum().item(), 1e-12,
                                       f'{arm}: {name} lacks substantive CE gradient')
            self.assertTrue(all(p.grad is None for p in m.base.parameters()))

    def test_full_char_distinctions_and_unique_live_encoding(self):
        m = self.model('HYBRID')
        self.assertEqual(m.char_ids.shape[1], 4)
        f = torch.tensor([[0, 0, 1]])
        ln = torch.tensor([[3, 3, 2]])
        mask = torch.tensor([[True, True, False]])
        seen = []
        hook = m.candidate.register_forward_pre_hook(lambda module, args: seen.append(args[0].shape[0]))
        pos, embeddings = m.candidate_embeddings(f, ln, mask)
        hook.remove()
        self.assertEqual(seen, [3])
        self.assertEqual(len(pos), 6)
        self.assertFalse(torch.equal(embeddings[0], embeddings[1]))
        self.assertFalse(torch.equal(embeddings[0], embeddings[2]))
        self.assertTrue(torch.equal(embeddings[:3], embeddings[3:]))
        # Base alphabet deliberately omitted every accented character.
        self.assertFalse(torch.equal(m.char_ids[0], m.char_ids[1]))

    def test_padding_invariance(self):
        for arm in hm.CONFIGS:
            m = self.model(arm)
            self.nonzero_head(m)
            xs = inputs()
            original = m.logits(*xs)
            padded = []
            for value in xs:
                shape = list(value.shape)
                shape[1] = 3
                extra = torch.zeros(shape, dtype=value.dtype)
                if value is xs[0]:
                    extra.fill_(8)
                elif value is xs[1]:
                    extra.fill_(1)
                elif value is xs[2]:
                    extra.fill_(100000)  # masked form ids must not be dereferenced
                padded.append(torch.cat((value, extra), 1))
            result = m.logits(*padded)
            torch.testing.assert_close(result[:, :2], original, rtol=1e-5, atol=1e-6)
            self.assertTrue((result[:, 2:] == ev.NEG).all())

    def test_empty_masks_and_zero_dimensions(self):
        for arm in hm.CONFIGS:
            m = self.model(arm)
            xs = list(inputs())
            xs[-1] = torch.zeros_like(xs[-1])
            out = m.logits(*xs)
            self.assertTrue(torch.isfinite(out).all())
            self.assertTrue((out == ev.NEG).all())
            out.sum().backward()
            for axis in (0, 1):
                empty = [x[:0] if axis == 0 else x[:, :0] for x in xs]
                result = m.logits(*empty)
                self.assertEqual(result.shape, (*empty[2].shape, 3))
            # Mixed batches with an entirely empty row must be finite as well.
            xs = list(inputs())
            xs[-1][1] = False
            self.assertTrue(torch.isfinite(m.logits(*xs)).all())

    def test_strict_reload(self):
        for arm in hm.CONFIGS:
            m = self.model(arm)
            self.nonzero_head(m)
            checkpoint = dict(self.payload, model=m.state_dict(), arm=hm.arm_name(arm),
                              config_name=arm, reranker_config=m.cfg)
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / 'model.pt'
                torch.save(checkpoint, path)
                loaded, _ = hm.load_reranker_checkpoint(path)
                self.assertTrue(torch.equal(loaded.logits(*inputs()), m.logits(*inputs())))
                broken = copy.deepcopy(checkpoint)
                del broken['model']['query.weight']
                torch.save(broken, path)
                with self.assertRaises(RuntimeError):
                    hm.load_reranker_checkpoint(path)

    def test_candidate_permutation_equivariance(self):
        for arm in hm.CONFIGS:
            m = self.model(arm)
            self.nonzero_head(m)
            original = m.logits(*inputs())
            perm = torch.tensor([2, 0, 1])
            with torch.no_grad():
                for table in (m.candidate_index, m.tone_table, m.mod_table):
                    table[0] = table[0, perm].clone()
            changed = m.logits(*inputs())
            torch.testing.assert_close(changed[0, 0], original[0, 0, perm], rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(changed[0, 1], original[0, 1], rtol=1e-5, atol=1e-6)

    def test_capacity_exactly_matched(self):
        control, hybrid = [self.model(arm) for arm in hm.CONFIGS]
        self.assertEqual(control.parameter_counts(), hybrid.parameter_counts())
        self.assertGreater(control.parameter_counts()['trainable'], 0)

    def test_non_prefix_mask_rejected(self):
        xs = list(inputs())
        xs[-1][1] = torch.tensor([False, True])
        with self.assertRaisesRegex(ValueError, 'right-padded'):
            self.model('HYBRID').logits(*xs)


if __name__ == '__main__':
    unittest.main(verbosity=2)
