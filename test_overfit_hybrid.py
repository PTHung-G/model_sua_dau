"""CPU-only fixed-subset diagnostic tests using a synthetic packed factorized base."""
import argparse
import contextlib
import io
import json
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

import hybrid_rerank_model as hm
import overfit_hybrid as oh
from test_hybrid_rerank import CFG, payload_fixture


class OverfitHybridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(12)
        self.payload = payload_fixture()
        self.rows = [['AB,', 'áb!', '123'], ['ab'], ['áb'], ['áb', 'ab'], ['ab', 'áb']]

    def test_defaults_and_smoke(self):
        args = oh.parse_args([])
        self.assertEqual((args.subset_size, args.steps, args.batch_size, args.lr), (1024, 4000, 64, 3e-4))
        self.assertEqual(hm.DEFAULT_CONFIG['dropout'], 0.1)
        smoke = oh.parse_args(['--smoke'])
        self.assertEqual((smoke.subset_size, smoke.steps, smoke.eval_every), (128, 2, 1))
        for argv in (['--steps', '-1'], ['--lr', 'nan'], ['--batch-size', '0']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                oh.parse_args(argv)

    def test_selection_matches_first_covered_in_permutation(self):
        rows = self.rows + [['unknown'], ['àb'], ['!!!'], []]
        expected = [int(i) for i in oh.se.epoch_order(len(rows), 7) if i < len(self.rows)][:4]
        selected = oh.select_subset(rows, self.payload, 4, 7)
        self.assertEqual(selected.tolist(), expected)
        self.assertEqual(len(set(selected.tolist())), 4)
        np.testing.assert_array_equal(selected, oh.select_subset(rows, self.payload, 4, 7))
        with self.assertRaisesRegex(ValueError, 'found only 5'):
            oh.select_subset(rows, self.payload, 6)

    def test_eval_mode_raw_contract_and_weighted_ce(self):
        model = hm.HybridReranker(self.payload, 'HYBRID', CFG).train()
        metrics = oh.evaluate_model(model, self.rows, self.payload, batch_size=2)
        self.assertTrue(model.training)
        self.assertFalse(model.base.training)
        self.assertEqual(metrics['n_sentences'], 5)
        self.assertEqual(metrics['n_tokens'], 8)  # numbers never score, punctuation/case normalized
        self.assertEqual(metrics['ambiguous_tokens'], 8)
        self.assertEqual(metrics['supervised_tokens'], 8)
        self.assertGreater(metrics['supervised_ce'], 0)
        self.assertAlmostEqual(metrics['ambiguous_token_accuracy'], metrics['token_accuracy'])
        all_at_once = oh.evaluate_model(model, self.rows, self.payload, batch_size=5)
        self.assertAlmostEqual(metrics['supervised_ce'], all_at_once['supervised_ce'], places=6)

    def test_repeated_epochs_full_checkpoint_and_initial_preservation(self):
        with tempfile.TemporaryDirectory() as td:
            args = argparse.Namespace(out_dir=td, subset_size=5, seed=0, batch_size=2,
                                      lr=0.003, steps=5, eval_every=2)
            with contextlib.redirect_stdout(io.StringIO()):
                result = oh.run_experiment(args, self.payload, self.rows, {'source': 'toy'}, reranker_config=CFG)
            self.assertEqual(result['status'], 'complete')
            self.assertTrue(result['diagnostic_only'])
            self.assertFalse(result['generalization_evidence'])
            self.assertEqual([x['step'] for x in result['evaluation_log']], [0, 2, 4, 5])
            self.assertEqual(result['optimizer_steps'], 5)
            self.assertEqual(result['completed_epochs'], 1)
            self.assertEqual(result['epoch_cursor'], 4)
            self.assertTrue(all(row['lr'] == args.lr for row in result['loss_log']))
            checkpoints = {}
            for kind in ('initial', 'best', 'latest'):
                model, checkpoint = hm.load_reranker_checkpoint(Path(td) / f'overfit_hybrid_{kind}.pt')
                checkpoints[kind] = checkpoint
                self.assertTrue(all(torch.isfinite(t).all() for t in checkpoint['model'].values() if t.is_floating_point()))
                for key, tensor in self.payload['model'].items():
                    self.assertTrue(torch.equal(model.base.state_dict()[key], tensor), key)
                self.assertTrue(all(not p.requires_grad for p in model.base.parameters()))
                self.assertIn('optimizer', checkpoint)
                self.assertIn('scheduler', checkpoint)
                self.assertEqual(set(checkpoint['rng']), {'python', 'numpy', 'torch', 'cuda', 'batch_generator'})
                metric = oh.evaluate_model(model, self.rows, self.payload, 2)
                expected = result['initial'] if kind == 'initial' else (result['final'] if kind == 'latest' else result['best']['metrics'])
                self.assertAlmostEqual(metric['supervised_ce'], expected['supervised_ce'], places=6)
            self.assertEqual(checkpoints['initial']['step'], 0)
            self.assertEqual(checkpoints['latest']['step'], 5)
            self.assertTrue(checkpoints['latest']['optimizer']['state'])
            self.assertEqual(checkpoints['latest']['scheduler']['last_epoch'], 5)
            self.assertEqual(checkpoints['latest']['epoch_order'].shape, (5,))
            self.assertEqual(result['best']['metrics']['supervised_ce'], min(x['metrics']['supervised_ce'] for x in result['evaluation_log']))
            manifest = json.loads((Path(td) / 'sample_manifests.json').read_text())
            saved = np.load(Path(td) / 'sample_indices.npy', allow_pickle=False)
            self.assertEqual(manifest['sample_indices'], saved.tolist())
            self.assertEqual(len(manifest['indices_sha256']), 64)
            self.assertTrue((Path(td) / 'sources' / 'overfit_hybrid.py').exists())
            with self.assertRaises(FileExistsError):
                oh.run_experiment(args, self.payload, self.rows, {}, reranker_config=CFG)

    def test_zero_steps_saves_initial_best_and_latest(self):
        with tempfile.TemporaryDirectory() as td:
            args = argparse.Namespace(out_dir=td, subset_size=2, seed=0, batch_size=2,
                                      lr=3e-4, steps=0, eval_every=3)
            with contextlib.redirect_stdout(io.StringIO()):
                result = oh.run_experiment(args, self.payload, self.rows, {}, reranker_config=CFG)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['best']['step'], 0)
            for kind in ('initial', 'best', 'latest'):
                _, checkpoint = hm.load_reranker_checkpoint(Path(td) / f'overfit_hybrid_{kind}.pt')
                self.assertEqual(checkpoint['step'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
