"""Sampler coverage and tiny CPU integration checks, without touching project data.

Run: /home/dsh/ML/.venv/bin/python -m unittest -v test_epoch_sampler
"""
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

# Reproduce Kaggle's embedded module aliases without changing source files.
import sylmodel_attention
with patch.dict(sys.modules, {'sylmodel': sylmodel_attention}):
    trainer = importlib.import_module('syltrain_epochs')


class EpochSamplerTests(unittest.TestCase):
    def test_segment_and_batch_coverage(self):
        sent = np.array([1, 64, 65, 150])
        off = np.array([0, 1, 65, 130])
        starts, lengths = trainer.segments(sent, off)
        self.assertEqual(lengths.tolist(), [1, 64, 64, 1, 64, 64, 22])
        visits = np.zeros(280, dtype=int)
        for start, length in zip(starts, lengths):
            visits[start:start + length] += 1
        np.testing.assert_array_equal(visits, np.ones(280))
        for seed in range(5):
            batches = list(trainer.epoch_batches(lengths, 3, np.random.default_rng(seed), 1))
            self.assertEqual([len(b) for b in batches], [3, 3, 1])
            np.testing.assert_array_equal(np.sort(np.concatenate(batches)), np.arange(7))
            covered = np.zeros(280, dtype=int)
            for batch in batches:
                for i in batch:
                    covered[starts[i]:starts[i] + lengths[i]] += 1
            np.testing.assert_array_equal(covered, np.ones(280))

    def test_random_epochs_and_local_sort(self):
        lengths = np.arange(1, 1001) % 64 + 1
        a = list(trainer.epoch_batches(lengths, 7, np.random.default_rng(5), 3))
        b = list(trainer.epoch_batches(lengths, 7, np.random.default_rng(5), 3))
        c = list(trainer.epoch_batches(lengths, 7, np.random.default_rng(6), 3))
        np.testing.assert_array_equal(np.concatenate(a), np.concatenate(b))
        self.assertFalse(np.array_equal(np.concatenate(a), np.concatenate(c)))
        self.assertEqual(len(a), 143)
        self.assertTrue(all(np.all(np.diff(lengths[batch]) >= 0) for batch in a))
        np.testing.assert_array_equal(np.sort(np.concatenate(a)), np.arange(1000))

    def test_batch_feature_alignment(self):
        sent = np.array([1, 64, 65, 150])
        off = np.array([0, 1, 65, 130])
        base = np.arange(280)
        features = np.stack((base % 5, base % 6), axis=1)
        data = base, base + 10, sent, off, features
        starts, lengths = trainer.segments(sent, off)
        x, y, ln, ff = trainer.make_batch(data, starts, lengths, np.array([6, 0, 2]), 'cpu')
        for row, i in enumerate([6, 0, 2]):
            n = int(lengths[i]); start = starts[i]
            np.testing.assert_array_equal(x[row, :n], base[start:start + n])
            np.testing.assert_array_equal(y[row, :n], base[start:start + n] + 10)
            np.testing.assert_array_equal(ff[row, :n], features[start:start + n])
            self.assertEqual(int(x[row, n:].sum()), 0)

    def test_scheduler_updates(self):
        p = torch.nn.Parameter(torch.tensor(0.))
        optimizer = torch.optim.SGD([p], lr=1.)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: trainer.lr_factor(i, 100))
        used = []
        for _ in range(100):
            used.append(optimizer.param_groups[0]['lr'])
            optimizer.step(); scheduler.step()
        self.assertEqual(scheduler.last_epoch, 100)
        self.assertAlmostEqual(used[0], .2)
        self.assertAlmostEqual(used[4], 1.)
        self.assertAlmostEqual(used[-1], .05)
        self.assertTrue(all(np.isfinite(used)))

    def test_invalid_lengths(self):
        with self.assertRaises(ValueError):
            trainer.segments(np.array([0]), np.array([0]))


class TinyIntegrationTests(unittest.TestCase):
    def make_data(self, root):
        prefix = root / 'train'
        Path(str(prefix) + '_vocab.json').write_text(json.dumps(['a', 'á']))
        Path(str(prefix) + '_bases.json').write_text(json.dumps(['a']))
        np.save(str(prefix) + '_ctab.npy', np.array([[-1, -1], [0, 1]], dtype=np.int32))
        for name in ('train', 'validation/val'):
            p = root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            lengths = np.array([1, 64, 65, 150], dtype=np.int32)
            n = int(lengths.sum())
            np.savez(str(p) + '.npz', base=np.ones(n, dtype=np.int32),
                     gold=np.arange(n, dtype=np.int32) % 2, sent=lengths)
            ff = np.ones((n, 2), dtype=np.uint8)
            ff[:, 0] = 2 if name == 'train' else 3
            np.save(str(p) + '_input_features.npy', ff)
        return prefix, root / 'validation/val'

    def run_main(self, root, cap):
        train, val = self.make_data(root)
        out = root / 'result.pt'
        argv = ['syltrain_epochs.py', '--prefix', str(train), '--val-prefix', str(val),
                '--epochs', '2', '--bs', '3', '--d', '16', '--hidden', '8',
                '--r', '4', '--layers', '1', '--eval-every', '2', '--max-minutes',
                str(cap), '--out', str(out), '--smoke']
        # Trainer itself uses the required embedded alias; patch only in test.
        with patch.object(trainer, 'sm', sylmodel_attention), patch.object(sys, 'argv', argv):
            trainer.main()
        return torch.load(out, map_location='cpu', weights_only=False), out

    def test_two_epochs_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            state, out = self.run_main(Path(tmp), 0)
            self.assertEqual(state['status'], 'completed')
            self.assertEqual(state['step'], 6)
            self.assertEqual(state['scheduler']['last_epoch'], 6)
            self.assertEqual(state['epochs_completed'], 2)
            self.assertEqual(state['tokens_seen'], 560)
            self.assertEqual(state['epoch_tokens_seen'], 280)
            self.assertEqual(state['validation_tokens'], 280)  # Includes full 150-token row.
            self.assertEqual(state['epoch_batches_completed'], 3)
            self.assertTrue(Path(str(out) + '.best.pt').is_file())
            self.assertFalse(out.with_suffix('.pt.tmp').exists())
            self.assertIn('cuda_rng', state)
            self.assertIn('numpy_rng', state)

    def test_time_cap_is_not_completed_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            state, _ = self.run_main(Path(tmp), 1e-12)
            self.assertEqual(state['status'], 'time_cap')
            self.assertEqual(state['epochs_completed'], 0)
            self.assertEqual(state['step'], 0)
            self.assertEqual(state['tokens_seen'], 0)
            self.assertEqual(state['validation_tokens'], 280)


if __name__ == '__main__':
    unittest.main()
