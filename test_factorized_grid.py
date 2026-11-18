"""Offline synthetic tests: no trained checkpoint, corpus, dataset download or network.

Run: python3 -B -m unittest -v test_factorized_grid
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

import eval_factorized_grid as eg


def mapping():
    # Deliberately NOT sorted: inference must obey the persisted form order.
    return dict(cand={'ma': ['ma', 'má'], 'ban': ['bạn']}, forms=['ma', 'ban'],
                char2id={'a': 1, 'b': 2, 'm': 3, 'n': 4}, nchar=6, max_c=2,
                signatures={'tone': [(), ((1, '́'),), ((1, '̣'),)],
                            'modifier': [()]},
                tone_table=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
                mod_table=torch.zeros(2, 2, dtype=torch.long))


class DummyFactorized(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, xc, cl, mask=None):
        assert mask is not None and mask.dtype == torch.bool and mask.all()
        assert (cl >= 1).all() and cl.max() <= xc.shape[-1]
        self.calls.append((xc.clone(), cl.clone(), mask.clone()))
        # Illegal second slot for ban has huge score: ln must mask it.
        tone = torch.tensor([0., 4., 100.]).expand(*cl.shape, 3)
        return tone, torch.zeros(*cl.shape, 1)


class FactorizedGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_inference_form_ids_lengths_case_punctuation_and_uncovered(self):
        model, ckpt = DummyFactorized(), mapping()
        out = eg.predict(model, ['“MA,” (Ban)!  zó … má', '!!!', ''], ckpt)
        self.assertEqual(out, [['“MÁ,”', '(Bạn)!', 'zo', '…', 'má'], ['!!!'], []])
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0][1].tolist(), [[2, 3, 2]])
        self.assertTrue(model.calls[0][2].all())  # includes genuine form id zero

    def test_pick0_uses_identical_alignment_and_never_calls_model(self):
        model, ckpt = DummyFactorized(), mapping()
        out = eg.predict(model, ['MÁ, !!! (Ban) zó'], ckpt, pick0=True)
        self.assertEqual(out, [['MA,', '!!!', '(Bạn)', 'zo']])
        self.assertEqual(model.calls, [])

    def test_nfc_and_character_lengths_are_packed_safe(self):
        model, ckpt = DummyFactorized(), mapping()
        self.assertEqual(eg.predict(model, ['ma\u0301'], ckpt), [['má']])
        self.assertEqual(model.calls[0][1].tolist(), [[2]])
        ckpt['cand']['m' * 70] = ['m' * 70]
        ckpt['forms'].append('m' * 70)
        ckpt['tone_table'] = torch.cat((ckpt['tone_table'], torch.zeros(1, 2, dtype=torch.long)))
        ckpt['mod_table'] = torch.zeros(3, 2, dtype=torch.long)
        eg.predict(model, ['m' * 70], ckpt)
        self.assertEqual(model.calls[-1][1].tolist(), [[64]])

    def test_scorer_uses_checkpoint_candidates_and_restores_global(self):
        previous = {'other': ['other']}
        with patch.object(eg.sa, 'CAND', previous):
            result = eg.score_predictions(['Má, bạn zó !!!'],
                                          [['Má,', 'bạn', 'zo', '!!!']], mapping()['cand'])
            self.assertIs(eg.sa.CAND, previous)
        self.assertEqual(result['n_scoring'], 3)
        self.assertAlmostEqual(result['all_syll_acc'], 2 / 3)
        self.assertEqual(result['syll_acc'], 1)
        self.assertAlmostEqual(result['stem_cov'], 2 / 3)
        self.assertAlmostEqual(result['exact_cov'], 2 / 3)
        self.assertAlmostEqual(result['raw_acc'], 2 / 3)
        result = eg.score_predictions(['mà'], [['má']], mapping()['cand'])
        self.assertEqual(result['stem_cov'], 1)
        self.assertEqual(result['exact_cov'], 0)

    def test_grid_inputs_not_targets_and_unweighted_mean_with_limit(self):
        model, ckpt = DummyFactorized(), mapping()
        datasets = {
            'GRID:business_55': [('ma', 'má'), ('ma', 'ma')],
            'GRID:formal_72': [('ma ma ma', 'ma ma ma')],
            'GRID:conversational_300': [('bán', 'bạn')],
            'GRID:literary_800': [('zzz', 'má')],
        }
        with patch.object(eg.sa, 'load_pairs', side_effect=lambda p: datasets[p]) as load:
            report = eg.evaluate_grid(model, ckpt, limit=1)
        self.assertEqual(load.call_count, 4)
        self.assertEqual(report['registers']['business']['n_sentences'], 1)
        self.assertEqual(report['registers']['literary']['scores']['all_syll_acc'], 0)
        self.assertEqual(report['MEAN']['all_syll_acc'], 0.5)
        self.assertEqual(report['pick0_MEAN']['all_syll_acc'], 0.5)
        # Gold ma has a different tone than dummy's má; never used as model input.
        self.assertEqual(report['registers']['formal']['scores']['all_syll_acc'], 0)
        self.assertEqual(len(model.calls), 3)

    def synthetic_full_payload(self):
        payload = mapping()
        payload.update(arm='factorized', step=17,
                       config=dict(syl=8, dim=8, blocks=1, heads=2, ch_dim=4,
                                   window=2, out_std=.1, drop=0., ff_mult=1,
                                   packed_chars=True))
        model = eg.pa.FactorizedDiac(payload['nchar'], payload['max_c'],
                                     len(payload['signatures']['modifier']),
                                     len(payload['signatures']['tone']), **payload['config'])
        payload['model'] = model.state_dict()
        payload['params'] = sum(p.numel() for p in model.parameters())
        return payload

    def test_real_tiny_factorized_instantiation_strict_load_and_inference(self):
        payload = self.synthetic_full_payload()
        model = eg.model_from_checkpoint(payload)
        self.assertIsInstance(model, eg.pa.FactorizedDiac)
        self.assertTrue(model.packed_chars)
        self.assertFalse(model.training)
        self.assertEqual(len(eg.predict(model, ['ma ban zó'], payload)[0]), 3)
        bad = copy.deepcopy(payload)
        bad['model'].pop('tone_head.weight')
        with self.assertRaises(RuntimeError):
            eg.model_from_checkpoint(bad)

    def test_rejects_incomplete_or_legacy_metadata_no_mutable_fallback(self):
        payload = self.synthetic_full_payload()
        for key in ('cand', 'char2id', 'config', 'signatures', 'tone_table', 'mod_table'):
            bad = dict(payload)
            del bad[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                eg.model_from_checkpoint(bad)
        for mutate in (
            lambda p: p['config'].update(packed_chars=False),
            lambda p: p.update(forms=['ma', 'ma']),
            lambda p: p.update(params=-1),
            lambda p: p.update(tone_table=torch.zeros(1, 2, dtype=torch.long)),
        ):
            bad = copy.deepcopy(payload)
            mutate(bad)
            with self.assertRaises(ValueError):
                eg.model_from_checkpoint(bad)

    def test_cli_writes_hash_step_params_protocol_and_scores(self):
        payload = self.synthetic_full_payload()
        with tempfile.TemporaryDirectory() as tmp:
            # Opaque bytes stand in for a trusted full checkpoint; torch.load is mocked.
            ckpt, out = Path(tmp) / 'synthetic.pt', Path(tmp) / 'scores.json'
            ckpt.write_bytes(b'synthetic offline checkpoint')
            with patch.object(eg.torch, 'load', return_value=payload) as load, \
                    patch.object(eg.sa, 'load_pairs', return_value=[('ma', 'má')]):
                eg.main(['--ckpt', str(ckpt), '--out', str(out), '--limit', '1'])
            load.assert_called_once_with(ckpt.resolve(), map_location='cpu', weights_only=False)
            report = json.loads(out.read_text(encoding='utf-8'))
            self.assertEqual(report['checkpoint']['step'], 17)
            self.assertEqual(report['checkpoint']['params'], payload['params'])
            self.assertEqual(report['checkpoint']['sha256'], eg.file_sha256(ckpt))
            self.assertIn('NOT official', report['protocol']['status'])
            self.assertEqual(set(report['MEAN']), set(eg.METRICS))
            self.assertEqual(len(report['registers']), 4)
            self.assertEqual(report['limit'], 1)


if __name__ == '__main__':
    unittest.main()
