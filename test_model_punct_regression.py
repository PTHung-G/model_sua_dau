"""CPU-only regression tests; run python -m unittest test_model_punct_regression -v."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import build_punct_tensors_v1 as builder
from charexp.sylmodel_punct import PunctSylCRF


def model():
    torch.manual_seed(29)
    chars = torch.tensor([[0, 0, 0, 0], [1, 2, 0, 0], [2, 3, 4, 0]], dtype=torch.int16)
    ids = torch.tensor([[-1, -1], [0, 1], [1, 2]], dtype=torch.int32)
    return PunctSylCRF(12, chars, chars, ids, 8, d=16, hidden=8,
                       layers=1, r=4, drop=0).eval()


class ModelRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_gap_context_invariant_to_batch_padding(self):
        m = model()
        single = m.gap_context(torch.tensor([[1, 2, 3]]), torch.tensor([2]), 3)
        batch = m.gap_context(torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 1]]),
                              torch.tensor([2, 4]), 5)
        torch.testing.assert_close(single[0], batch[0, :3], atol=1e-6, rtol=1e-5)
        self.assertEqual(batch[0, 3:].count_nonzero().item(), 0)

    def test_padding_contents_do_not_change_real_gaps(self):
        m = model()
        a = m.gap_context(torch.tensor([[1, 2, 0, 0]]), torch.tensor([1]), 4)
        b = m.gap_context(torch.tensor([[1, 2, 6, 7]]), torch.tensor([1]), 4)
        torch.testing.assert_close(a, b)

    def test_gap_range_validation(self):
        m = model()
        for ids in ([1, 8], [1, -1]):
            with self.assertRaisesRegex(ValueError, 'gap id outside'):
                m.gap_context(torch.tensor([ids]), torch.tensor([1]), 2)
        with self.assertRaisesRegex(ValueError, 'shape'):
            m.gap_context(torch.tensor([[1, 2, 3]]), torch.tensor([1]), 2)

    def test_internal_unknown_does_not_truncate_suffix(self):
        m = model()
        ids = torch.tensor([[1, 0, 2, 3, 0], [0, 1, 2, 0, 0]])
        actual = m.encode_chars(ids)
        for i, length in enumerate([4, 3]):
            _, h = m.char_encoder(m.emb(ids[i:i+1, :length]))
            expected = torch.cat((h[-2], h[-1]), -1)[0]
            torch.testing.assert_close(actual[i], expected)
        self.assertEqual(m.encode_chars(torch.zeros(1, 5, dtype=torch.long)).count_nonzero().item(), 0)

    def test_full_forward_batch_padding_invariance(self):
        m = model()
        with torch.no_grad():
            m.gap_proj.weight.normal_()
            m.edge_gate.weight.normal_()
        a = m(torch.tensor([[1, 2]]), torch.tensor([2]), torch.tensor([[2, 1]]),
              torch.tensor([[1, 2, 3]]))
        b = m(torch.tensor([[1, 2, 0, 0], [2, 1, 2, 1]]), torch.tensor([2, 4]),
              torch.tensor([[2, 1, 0, 0], [1, 2, 1, 1]]),
              torch.tensor([[1, 2, 3, 0, 0], [2, 3, 4, 5, 6]]))
        torch.testing.assert_close(a[0][0], b[0][0, :2], atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(a[1][0], b[1][0, :1], atol=2e-5, rtol=2e-5)


class BuilderAudit(unittest.TestCase):
    """Characterize builder behavior without changing its implementation."""
    def test_fresh_gap_ids_are_dense(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'pairs.jsonl'
            row = {'input': 'Toi buon, ban vui?', 'target': 'Tôi buồn, bạn vui?'}
            path.write_text(json.dumps(row, ensure_ascii=False) + '\n')
            *_, pid, n = builder.build_table(path)
            self.assertEqual(sorted(pid.values()), list(range(len(pid))))
            self.assertEqual(n, 1)

    def test_256_boundary_is_inclusive(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            path = td / 'pairs.jsonl'
            path.write_text('\n'.join(json.dumps({'input': ' '.join(['toi'] * n),
                                                  'target': ' '.join(['tôi'] * n)})
                                       for n in [255, 256, 257]) + '\n')
            bases, vocab, table, pid, _ = builder.build_table(path)
            result = builder.encode(path, td / 'encoded', bases, vocab, table, pid)
            self.assertEqual(result['rows'], 2)
            self.assertEqual(result['skipped'], 1)
            self.assertEqual(np.load(td / 'encoded.npz')['sent'].tolist(), [255, 256])

    def test_case_feature_cannot_preserve_mixed_case_positions(self):
        self.assertEqual(builder.case_id('iPhone'), builder.case_id('IPhoNe'))
        self.assertEqual(builder.case_id('Toi'), 2)
        self.assertEqual(builder.case_id('TOI'), 3)
        # The model receives only category 4, not the original per-character case.
        self.assertEqual(builder.case_id('iPhone'), 4)


if __name__ == '__main__':
    unittest.main()
