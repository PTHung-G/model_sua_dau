"""CPU tests for frozen-generator candidate-aware two-pass experiment."""
import hashlib
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

import two_pass_pilot as tp


def fixture():
    config = dict(ch_dim=8, syl=16, dim=384, heads=4, window=2,
                  blocks=1, drop=0.1, ff_mult=1, out_std=0.01, packed_chars=True)
    base = tp.pa.FactorizedDiac(5, 3, 2, 3, **config)
    return dict(config=config, model=base.state_dict(), nchar=5, max_c=3,
                signatures={'modifier': [(), ((0, 'â'),)], 'tone': [(), ((0, '́'),), ((0, '̀'),)]},
                tone_table=torch.tensor([[0, 1, 2], [0, 0, 0]]),
                mod_table=torch.tensor([[0, 0, 1], [0, 0, 0]]),
                cand={'a': ['a', 'á', 'ầ'], 'b': ['b']}, forms=['a', 'b'],
                char2id={'a': 1, 'b': 2}, arm='factorized', step=120000)


def state_hash(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(tensor.cpu().numpy().tobytes())
    return h.hexdigest()


class TwoPassTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        tp.cp.reset_seed(0)
        self.payload = fixture()
        self.model = tp.TwoPassDiac(self.payload).eval()
        self.batch, _ = tp.se.collate([['á', 'b', 'a'], ['a']], self.payload, 'cpu')

    def inputs(self, batch=None):
        xc, cl, f, y, ln, mask, ok = self.batch if batch is None else batch
        return xc, cl, f, ln, mask

    def test_initial_exact_identity_train_and_eval(self):
        args = self.inputs()
        baseline = tp.cp.model_from_checkpoint(self.payload, 'baseline').cpu().eval()
        expected = tp.pa.candidate_logits(baseline, 'factorized', *args,
                                          (self.payload['tone_table'], self.payload['mod_table']))
        for train in (False, True):
            self.model.train(train)
            actual = self.model.logits(*args)
            self.assertTrue(torch.equal(actual, expected))
            self.assertFalse(self.model.base.training)
        self.assertGreater(float(self.model.tone_embedding.weight.abs().sum()), 0)

    def test_nonzero_correction_padding_invariance(self):
        torch.nn.init.normal_(self.model.query.weight, std=0.02)
        torch.nn.init.normal_(self.model.query.bias, std=0.02)
        args = self.inputs()
        original = self.model.logits(*args)
        _, base = self.model.frozen_scores(*args)
        valid = torch.arange(3) < args[3][..., None]
        self.assertGreater(float((original - base)[valid & args[4][..., None]].abs().sum()), 0)
        xc, cl, f, ln, mask = args
        def pad(x, value):
            shape = list(x.shape)
            shape[1] = 4
            return torch.cat((x, torch.full(shape, value, dtype=x.dtype)), dim=1)
        padded = self.model.logits(pad(xc, 4), pad(cl, 1), pad(f, 0), pad(ln, 1), pad(mask, False))
        torch.testing.assert_close(original[mask], padded[:, :mask.shape[1]][mask], rtol=1e-5, atol=1e-6)

    def test_frozen_base_and_corrector_gradients(self):
        self.model.train()
        before = state_hash(self.model.base)
        optimizer = torch.optim.AdamW((p for p in self.model.parameters() if p.requires_grad), lr=0.001)
        xc, cl, f, y, ln, mask, ok = self.batch
        target = tp.sl.training_target(mask, ok, ln)
        # Zero projection gates upstream gradients on first step; after update all
        # corrector components must receive real gradients, without gold context.
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            logits = self.model.logits(xc, cl, f, ln, mask)
            F.cross_entropy(logits[target], y[target]).backward()
            self.assertTrue(all(p.grad is None for p in self.model.base.parameters()))
            self.assertFalse(self.model.base.training)
            self.assertGreater(float(self.model.query.weight.grad.abs().sum()), 0)
            if step:
                for module in (self.model.tone_embedding, self.model.mod_embedding,
                               self.model.hidden_projection, self.model.context):
                    self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None), 0)
            optimizer.step()
        self.assertEqual(before, state_hash(self.model.base))

    def test_strict_self_contained_reload(self):
        torch.nn.init.normal_(self.model.query.weight, std=0.02)
        expected = self.model.logits(*self.inputs())
        saved = {k: v for k, v in self.payload.items() if k != 'model'}
        saved.update(arm=tp.ARM, corrector_config=self.model.corrector_config, model=self.model.state_dict())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'two_pass.pt'
            torch.save(saved, path)
            loaded, payload = tp.load_two_pass_checkpoint(path, 'cpu')
            self.assertTrue(torch.equal(expected, loaded.logits(*self.inputs())))
            payload['model'].pop('query.bias')
            torch.save(payload, path)
            with self.assertRaises(RuntimeError):
                tp.load_two_pass_checkpoint(path, 'cpu')

    def test_masked_candidates_and_empty_batches_finite(self):
        torch.nn.init.normal_(self.model.query.weight, std=0.02)
        args = list(self.inputs())
        for entirely_empty in (False, True):
            mask = args[-1].clone()
            mask[1] = False
            if entirely_empty:
                mask[:] = False
            args[-1] = mask
            logits = self.model.logits(*args)
            self.assertTrue(torch.isfinite(logits).all())
            invalid = torch.arange(3) >= args[3][..., None]
            self.assertTrue((logits[invalid] == tp.ev.NEG).all())
        args = self.inputs()
        for axis in (0, 1):
            empty = [x[:0] if axis == 0 else x[:, :0] for x in args]
            scores = self.model.logits(*empty)
            self.assertEqual(scores.shape, (*empty[2].shape, 3))
            self.assertTrue(torch.isfinite(scores).all())

    def test_raw_evaluation_topk_unknowns_and_missing(self):
        # Candidate ầ deliberately has a different stripped stem; fixture does
        # not build tables and this only exercises evaluation label coverage.
        rows = [['á', 'b'], ['à', 'xyz'], ['xyz']]
        metric = tp.evaluate_model(self.model, rows, self.payload)
        self.assertEqual(metric['unknown_letter_tokens'], 2)
        self.assertEqual(metric['candidate_gold_missing_count'], 1)
        self.assertEqual(metric['topk_oracle']['1']['n_errors'], metric['n_errors'])
        self.assertGreaterEqual(metric['topk_oracle']['3']['sentence_exact'], metric['sentence_exact'])
        self.assertEqual(metric['topk_oracle']['3'], metric['topk_oracle']['5'])

    def test_cli(self):
        args = tp.parse_args([])
        self.assertEqual((args.steps, args.val_sents, args.lr, args.train_limit), (10000, 3000, 0.0003, None))
        args = tp.parse_args(['--smoke'])
        self.assertEqual((args.train_limit, args.val_sents, args.steps), (128, 20, 2))


if __name__ == '__main__':
    unittest.main()
