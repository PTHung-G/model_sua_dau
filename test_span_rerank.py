"""CPU tests for the rank-aware span reranker (configurations A/B/C)."""
import hashlib
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

import span_rerank_pilot as sr


def fixture():
    config = dict(ch_dim=8, syl=16, dim=384, heads=4, window=2,
                  blocks=1, drop=0.1, ff_mult=1, out_std=0.01, packed_chars=True)
    base = sr.pa.FactorizedDiac(5, 3, 2, 3, **config)
    return dict(config=config, model=base.state_dict(), nchar=5, max_c=3,
                signatures={'modifier': [(), ((0, 'â'),)], 'tone': [(), ((0, '́'),), ((0, '̀'),)]},
                tone_table=torch.tensor([[0, 1, 2], [0, 0, 0]]),
                mod_table=torch.tensor([[0, 0, 1], [0, 0, 0]]),
                cand={'a': ['a', 'á', 'ầ'], 'b': ['b']}, forms=['a', 'b'],
                char2id={'a': 1, 'b': 2}, arm='factorized', step=120000)


class SpanRerankTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        sr.cp.reset_seed(0)
        self.payload = fixture()
        self.models = {name: sr.SpanReranker(self.payload, name).eval() for name in 'ABC'}
        self.batch, _ = sr.se.collate([['á', 'b', 'a'], ['a']], self.payload, 'cpu')

    def inputs(self):
        xc, cl, f, y, ln, mask, ok = self.batch
        return xc, cl, f, ln, mask

    def test_initial_identity_matches_frozen_generator(self):
        baseline = sr.cp.model_from_checkpoint(self.payload, 'baseline').cpu().eval()
        expected = sr.pa.candidate_logits(baseline, 'factorized', *self.inputs(),
                                          (self.payload['tone_table'], self.payload['mod_table']))
        for name, model in self.models.items():
            for training in (False, True):
                model.train(training)
                self.assertTrue(torch.equal(model.logits(*self.inputs()), expected), name)
                self.assertFalse(model.base.training, name)

    def test_padding_invariance_with_nonzero_correction(self):
        for name, model in self.models.items():
            torch.nn.init.normal_(model.slot.weight, std=0.05)
            torch.nn.init.normal_(model.slot.bias, std=0.05)
            padded, _ = sr.se.collate([['á', 'b', 'a'], ['a'], ['a', 'b']],
                                      self.payload, 'cpu')
            with torch.inference_mode():
                a = model.logits(*self.inputs())
                b = model.logits(padded[0], padded[1], padded[2], padded[4], padded[5])[:2]
            self.assertTrue(torch.allclose(a, b, atol=1e-4, rtol=1e-4), name)

    def test_frozen_base_untouched_by_optimizer(self):
        model = self.models['A']
        hashed = {key: value.clone() for key, value in model.base.state_dict().items()}
        model.train()
        xc, cl, f, ln, mask = self.inputs()
        logits = model.logits(xc, cl, f, ln, mask)
        logits.reshape(-1, logits.shape[-1]).sum().backward()
        opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=1e-2)
        opt.step()
        for key, value in model.base.state_dict().items():
            self.assertTrue(torch.equal(value, hashed[key]), key)

    def test_span_configs_receive_gradient_through_span_path(self):
        for name in 'BC':
            model = self.models[name]
            model.train()
            xc, cl, f, ln, mask = self.inputs()
            model.logits(xc, cl, f, ln, mask).sum().backward()
            self.assertIsNotNone(model.span_out.weight.grad, name)
            self.assertGreater(float(model.span_out.weight.grad.abs().sum()), 0, name)

    def test_ranking_loss_prefers_correct_candidate(self):
        logits = torch.tensor([[[2.0, 1.0, 0.0], [0.0, 3.0, 1.0]]])
        y = torch.tensor([[0, 1]])
        mask = torch.ones(1, 2, dtype=torch.bool)
        ok = torch.ones(1, 2, dtype=torch.bool)
        ln = torch.tensor([[3, 3]])
        good = sr.ranking_loss(logits, y, mask, ok, ln)
        bad = sr.ranking_loss(logits.flip(-1), y, mask, ok, ln)
        self.assertEqual(float(good), 0.0)
        self.assertGreater(float(bad), 0.0)
        single = torch.tensor([[[1.0, 0.0, 0.0]]])
        self.assertIsNone(sr.ranking_loss(single, torch.zeros(1, 1, dtype=torch.long),
                                          torch.ones(1, 1, dtype=torch.bool),
                                          torch.ones(1, 1, dtype=torch.bool),
                                          torch.ones(1, 1, dtype=torch.long)))

    def test_all_empty_rows_and_empty_batches_are_finite(self):
        for name in 'BC':
            model = self.models[name]
            model.train()
            xc, cl, f, ln, mask = self.inputs()
            zeros = [torch.zeros(2, 1, xc.shape[-1], dtype=xc.dtype), torch.ones(2, 1, dtype=cl.dtype),
                     torch.zeros(2, 1, dtype=f.dtype), torch.ones(2, 1, dtype=ln.dtype),
                     torch.zeros(2, 1, dtype=torch.bool)]
            out = model.logits(*zeros)
            self.assertTrue(torch.isfinite(out).all(), name)
        xc, cl, f, ln, mask = self.inputs()
        empty = torch.zeros(0, xc.shape[1], xc.shape[2], dtype=xc.dtype)
        out = self.models['C'].logits(empty, cl[:0], f[:0], ln[:0], mask[:0])
        self.assertEqual(out.shape[0], 0)

    def test_strict_reload_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            model = self.models['A']
            model.train()
            xc, cl, f, ln, mask = self.inputs()
            model.logits(xc, cl, f, ln, mask).sum().backward()
            opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=1e-3)
            opt.step()
            meta = dict(arm=sr.arm_name('A'), config_name='A', reranker_config=model.cfg,
                        cand=self.payload['cand'], forms=self.payload['forms'],
                        char2id=self.payload['char2id'], config=self.payload['config'],
                        nchar=self.payload['nchar'], max_c=self.payload['max_c'],
                        signatures=self.payload['signatures'],
                        tone_table=self.payload['tone_table'],
                        mod_table=self.payload['mod_table'])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
            rng = __import__('numpy').random.default_rng(0)
            sr.pa.checkpoint(out / 'ckpt.pt', model, opt, scheduler, rng, 1, meta, {})
            reloaded, payload = sr.load_reranker_checkpoint(out / 'ckpt.pt', device='cpu')
            self.assertEqual(payload['config_name'], 'A')
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(reloaded.state_dict()[key], value), key)
            model.eval()
            with torch.inference_mode():
                self.assertTrue(torch.allclose(reloaded.logits(*self.inputs()),
                                               model.logits(*self.inputs())))


if __name__ == '__main__':
    unittest.main()
