"""Offline scorer regression tests: no checkpoint, model inference, or network."""
import sys
import types
import unittest
from unittest.mock import patch

import score_all as sa
import eval_frontier as ef


class ScorerTests(unittest.TestCase):
    def setUp(self):
        self.old_cand = dict(sa.CAND)
        self.old_forms = dict(sa.FORMS_BY_STRIP)
        self.old_frontier = dict(ef.CAND)
        sa.CAND.clear()
        sa.CAND.update({'toi': ['tôi'], 'ban': ['bạn']})
        ef.CAND.clear()
        ef.CAND.update(sa.CAND)

    def tearDown(self):
        for target, saved in ((sa.CAND, self.old_cand),
                              (sa.FORMS_BY_STRIP, self.old_forms),
                              (ef.CAND, self.old_frontier)):
            target.clear()
            target.update(saved)

    def test_case_punctuation_nfc_and_letters(self):
        r = sa.score(['“TÔI,” 123 ...'], [['“tôi,”', '123', '...']])
        # Curly quotes are punctuation too; core normalization removes them.
        self.assertEqual(r['n_scoring'], 1)
        self.assertEqual(r['syll_acc'], 1)
        self.assertEqual(sa.normalized_core('To\u0302i,'), 'tôi')
        self.assertEqual(sa.restore_token('"TOI,"', 'tôi'), '"TÔI,"')

    def test_exact_coverage_separate_from_stem(self):
        r = sa.score(['tôi bán xa'], [['tôi', 'bạn', 'xa']])
        self.assertAlmostEqual(r['stem_cov'], 2 / 3)
        self.assertAlmostEqual(r['exact_cov'], 1 / 3)
        self.assertEqual(r['n_exact_covered'], 1)
        self.assertEqual(r['exact_acc'], 1)

    def test_uncovered_gold_is_not_copied(self):
        # Even accidental accented input is stripped before uncovered fallback.
        pred = sa.predict(None, ['Xứ,'], {}, 2, 1)
        self.assertEqual(pred, [['Xu,']])
        self.assertEqual(sa.score(['Xứ,'], pred)['all_syll_acc'], 0)
        self.assertEqual(ef.predict_sentences(None, ['Xứ,'], {}, {}, 2, {}, 1), ['Xu,'])

    def test_public_input_not_target(self):
        fake = types.SimpleNamespace(load_dataset=lambda *a, **kw:
                                     [{'input': 'toi, xu', 'target': 'tôi, xứ'}])
        with patch.dict(sys.modules, {'datasets': fake}):
            self.assertEqual(sa.load_pairs('GRID:test'), [('toi, xu', 'tôi, xứ')])
            self.assertEqual(sa.load_lines('GRID:test'), ['toi, xu'])

    def test_frontier_case_and_punctuation(self):
        hits, covered = ef.word_scores([('TOI, 123', 'Tôi, 123')], ['tôi, 123'])
        self.assertEqual(hits, [1])
        self.assertEqual(covered, [True])

    def test_sentence_count_rejected(self):
        with self.assertRaises(ValueError):
            sa.score(['tôi'], [])


if __name__ == '__main__':
    unittest.main()
