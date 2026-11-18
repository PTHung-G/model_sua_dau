"""Fast CPU-only evaluator contract tests; no production evaluation/training."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import eval_punct_vs_best as ev


class RawScoringTests(unittest.TestCase):
    def test_missing_and_oov_tokens_remain_in_denominator(self):
        r = ev.score_pairs([('toi xyz', 'tôi xyz')], ['tôi'])
        self.assertEqual(r['tokens'], 2)
        self.assertEqual(r['correct'], 1)
        self.assertEqual(r['missing_tokens'], 1)
        self.assertEqual(r['token_accuracy'], .5)
        self.assertEqual(r['sentence_exact'], 0)

    def test_extra_tokens_and_empty_targets(self):
        r = ev.score_pairs([('!', '!')], ['word!'])
        self.assertEqual(r['extra_tokens'], 1)
        self.assertIsNone(r['token_accuracy'])
        self.assertEqual(r['sentence_exact'], 0)

    def test_case_restore_and_punctuation(self):
        self.assertEqual(ev.restore_case('điện', 'dIeN'), 'đIệN')
        self.assertEqual(ev.replace_spans('  dIeN, (toi)!\n', ['đIệN', 'tôi']), '  đIệN, (tôi)!\n')
        with self.assertRaises(ValueError):
            ev.replace_spans('a b', ['a'])
        with self.assertRaises(ValueError):
            ev.score_pairs([('a', 'a')], [])

    def test_schema_rejects_historical_artifacts(self):
        for ck in ({}, {'pipeline_schema': 1}, {'pipeline_schema': True}):
            with self.assertRaisesRegex(ValueError, 'pipeline_schema'):
                ev.require_punct_schema(ck)
        ev.require_punct_schema({'pipeline_schema': 2})

    def test_gap_mapping_reserved_schema(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'punct.json'
            p.write_text(json.dumps({'<PAD>': 0, '<UNK>': 1, '': 2, ', ': 3}))
            self.assertEqual(ev.load_gap_mapping(p)[', '], 3)
            p.write_text(json.dumps({'': 0, ', ': 1}))
            with self.assertRaises(ValueError):
                ev.load_gap_mapping(p)

    def test_raw_rows_not_filtered_or_replaced_with_target(self):
        class Spy:
            def __init__(self): self.seen = []
            def predict(self, source):
                self.seen.append(source)
                return source, {'input_tokens': len(ev.TOK.findall(source)), 'known_base_tokens': 0,
                                'oov_copy_tokens': len(ev.TOK.findall(source))}
        spy = Spy()
        r = ev.evaluate(spy, [('xyz', 'xýz'), ('a b', 'a'), ('', 'missing')])
        self.assertEqual(spy.seen, ['xyz', 'a b', ''])
        self.assertEqual(r['rows'], 3)
        self.assertEqual(r['tokens'], 3)
        self.assertEqual(r['missing_tokens'], 1)
        self.assertEqual(r['coverage']['oov_copy_tokens'], 3)


class FrozenIntegrationTests(unittest.TestCase):
    def test_corrected_punct_exact_gap_alignment_and_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = str(Path(directory) / 'tiny')
            Path(prefix + '_bases.json').write_text(json.dumps(['toi']))
            Path(prefix + '_vocab.json').write_text(json.dumps(['tôi']))
            np.save(prefix + '_ctab.npy', np.array([[-1], [0]], np.int32))
            Path(prefix + '_punct.json').write_text(json.dumps({'<PAD>': 0, '<UNK>': 1, '': 2, ', ': 3, '!': 4}))
            st = ev.importlib.import_module('syltrain_punct')
            sm = ev.importlib.import_module('sylmodel_punct')
            nchar, bc, cc, cid, _ = st.build_tensors(prefix)
            config = {'d': 8, 'hidden': 4, 'r': 2, 'layers': 1}
            model = sm.PunctSylCRF(nchar, bc, cc, cid, 5, **config)
            checkpoint = Path(directory) / 'new.pt'
            torch.save({'pipeline_schema': 2, 'model': model.state_dict(), 'config': config}, checkpoint)
            restorer = ev.FrozenRestorer('punct', checkpoint, prefix)
            calls = []
            forward = restorer.model.forward
            def capture(x, ln, ff, gg):
                calls.append((ff.tolist(), gg.tolist()))
                return forward(x, ln, ff, gg)
            with patch.object(restorer.model, 'forward', side_effect=capture):
                output, _ = restorer.predict('TOI, zZ9!')
                again, _ = restorer.predict('toi?')
            self.assertEqual(output, 'TÔI, zZ9!')
            self.assertEqual(again, 'tôi?')
            self.assertEqual(calls, [([[3, 4]], [[2, 3, 4]]), ([[1]], [[2, 1]])])

    def test_native_old_features_oov_and_buffer_restoration(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = str(Path(directory) / 'tiny')
            Path(prefix + '_bases.json').write_text(json.dumps(['toi']))
            Path(prefix + '_vocab.json').write_text(json.dumps(['tôi']))
            np.save(prefix + '_ctab.npy', np.array([[-1], [0]], np.int32))
            st = ev.importlib.import_module('syltrain')
            sm = ev.importlib.import_module('sylmodel_attention')
            nchar, bc, cc, cid, _ = st.build_tensors(prefix)
            config = {'d': 8, 'hidden': 4, 'r': 2, 'layers': 1}
            model = sm.SylCRF(nchar, bc, cc, cid, **config)
            checkpoint = Path(directory) / 'old.pt'
            torch.save({'model': model.state_dict(), 'config': config}, checkpoint)
            restorer = ev.FrozenRestorer('attention', checkpoint, prefix)
            original = restorer.model.cand_ids
            calls = []
            forward = restorer.model.forward
            def capture(x, ln, ff):
                calls.append(ff.tolist())
                return forward(x, ln, ff)
            with patch.object(restorer.model, 'forward', side_effect=capture):
                output, coverage = restorer.predict('TOI, zZ9!')
                again, _ = restorer.predict('toi?')
            self.assertEqual(output, 'TÔI, zZ9!')
            self.assertEqual(again, 'tôi?')
            self.assertEqual(coverage['oov_copy_tokens'], 1)
            self.assertEqual(calls[0], [[[3, 5], [4, 2]]])
            self.assertIs(restorer.model.cand_ids, original)
            self.assertEqual(restorer.model.base_ch.shape[0], 2)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
