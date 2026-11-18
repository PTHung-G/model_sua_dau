"""Unit coverage for deterministic no-replacement sentence training."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import sentence_loss_epoch as se


class SentenceEpochTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        self.payload = dict(cand={'a': ['a', 'á'], 'b': ['b']},
                            forms=['a', 'b'], char2id={'a': 1, 'b': 2}, nchar=4)

    def test_permutation_coverage_remainder_determinism(self):
        order = se.epoch_order(131)
        batches = list(se.source_batches(order))
        self.assertEqual([len(x) for x in batches], [64, 64, 3])
        np.testing.assert_array_equal(np.sort(np.concatenate(batches)), np.arange(131))
        np.testing.assert_array_equal(order, se.epoch_order(131))
        self.assertFalse(np.array_equal(order, se.epoch_order(131, 1)))
        self.assertEqual(len(set(np.concatenate(batches).tolist())), 131)
        partial = np.concatenate(list(se.source_batches(order, budget=2)))
        self.assertEqual(len(partial), 128)
        self.assertEqual(len(set(partial.tolist())), 128)
        np.testing.assert_array_equal(partial, order[:128])

    def test_zero_one_known_and_invalid_gold(self):
        batch, stats = se.collate([['unknown'], [], ['á'], ['b'], ['à']], self.payload, 'cpu')
        self.assertEqual(stats['zero_known_rows'], 2)
        self.assertEqual(stats['one_known_rows'], 3)
        self.assertEqual(stats['invalid_gold_tokens'], 1)
        xc, cl, f, y, ln, mask, ok = batch
        self.assertEqual(tuple(y.shape), (3, 1))
        self.assertEqual(ok[:, 0].tolist(), [True, True, False])
        self.assertEqual(y[:, 0].tolist(), [1, 0, 0])
        self.assertEqual(se.sl.training_target(mask, ok, ln)[:, 0].tolist(), [True, False, False])
        empty, stats = se.collate([[], ['unknown']], self.payload, 'cpu')
        self.assertIsNone(empty)
        self.assertEqual(stats['zero_known_rows'], 2)
        single, _ = se.collate([['b']], self.payload, 'cpu')
        self.assertTrue(se.sl.training_target(single[5], single[6], single[4]).all())

    def test_finite_loss_gradient(self):
        batch, _ = se.collate([['á'], ['a', 'á'], ['b'], ['à']], self.payload, 'cpu')
        y, ln, mask, ok = batch[3:]
        logits = torch.randn(*y.shape, 2, requires_grad=True)
        target = se.sl.training_target(mask, ok, ln)
        loss, parts = se.sl.loss_components(logits, y, target, 'sentence_hard')
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(x) for x in parts.values()))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0)
        self.assertEqual(float(logits.grad[~target].abs().sum()), 0)

    def test_collator_matches_original_known_only_encoding(self):
        rows = [['a', 'unknown', 'á'], ['b', 'a']]
        class FixedRng:
            def integers(self, low, high, size):
                return np.arange(size)
        expected = next(se.pb.make_batches(rows, self.payload['cand'], self.payload['char2id'],
                                          2, 2, FixedRng(), nchar=4, forms={'a': 0, 'b': 1}))
        actual, _ = se.collate(rows, self.payload)
        for a, b in zip(actual, expected):
            self.assertTrue(torch.equal(a, b))

    def test_prefix_loader_hashes_full_source_without_full_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'tr.txt'
            raw = b'a\nb\nc\n'
            path.write_bytes(raw)
            with patch.object(se.pb, 'load_sents', side_effect=AssertionError('full load')):
                rows, manifest = se.load_training(path, 2)
            self.assertEqual(rows, [['a'], ['b']])
            self.assertEqual(manifest['source_lines'], 3)
            self.assertEqual(manifest['prefix_count'], 2)
            self.assertEqual(manifest['source_sha256'], hashlib.sha256(raw).hexdigest())
            self.assertEqual(manifest['prefix_sha256'], hashlib.sha256(b'a\nb\n').hexdigest())

    def test_cli_budget(self):
        args = se.parse_args([])
        self.assertEqual(args.steps, 10000)
        self.assertIsNone(args.train_limit)
        self.assertEqual(args.val_sents, 3000)
        args = se.parse_args(['--smoke'])
        self.assertEqual((args.steps, args.train_limit, args.val_sents), (2, 128, 20))


if __name__ == '__main__':
    unittest.main()
