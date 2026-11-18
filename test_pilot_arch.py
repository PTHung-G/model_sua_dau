"""Small offline architecture regression tests; no training/data/network needed."""
import tempfile
import unittest
from pathlib import Path
import torch
from attn_model import AttnDiac
from pilot_arch import factor_tables, candidate_logits, FactorizedDiac


class PilotTests(unittest.TestCase):
    def test_lossless_signatures_and_valid_candidates(self):
        cand = {'ai': ['ái', 'aí'], 'di': ['di', 'đi'],
                'duong': ['dương', 'đường', 'dưỡng']}
        forms = sorted(cand)
        with tempfile.TemporaryDirectory() as d:
            sigs, tone, mod = factor_tables(cand, forms, Path(d)/'collisions.json')
        for i, stem in enumerate(forms):
            pairs = [(int(tone[i,j]), int(mod[i,j])) for j in range(len(cand[stem]))]
            self.assertEqual(len(set(pairs)), len(pairs))
        m = FactorizedDiac(24, 3, len(sigs['modifier']), len(sigs['tone']),
                           syl=16, dim=32, heads=4, packed_chars=True,
                           window=8, out_std=.1).eval()
        xc = torch.ones((1,3,5),dtype=torch.long)
        cl = torch.tensor([[2,2,5]])
        f = torch.tensor([[0,1,2]])
        ln = torch.tensor([[2,2,3]])
        lg = candidate_logits(m,'factorized',xc,cl,f,ln,
                              torch.ones_like(f,dtype=torch.bool),(tone,mod))
        self.assertTrue(torch.isfinite(lg).all())
        self.assertTrue((lg.argmax(-1)<ln).all())
        self.assertEqual(float(lg[0,0,2].detach()), -1e9)

    def test_padding_invariance(self):
        for window in (8,'dual'):
            torch.manual_seed(13)
            m = AttnDiac(24,4,syl=16,dim=32,heads=4,
                         window=window,packed_chars=True).eval()
            x = torch.tensor([[[1,2,3],[4,5,23]]])
            cl = torch.tensor([[3,2]])
            big = torch.full((2,6,9),23,dtype=torch.long)
            big[0,:2,:3] = x[0]
            big[1] = 1
            lengths = torch.ones((2,6),dtype=torch.long)
            lengths[0,:2] = cl[0]
            lengths[1] = 9
            mask = torch.zeros((2,6),dtype=torch.bool)
            mask[0,:2] = True
            mask[1] = True
            with torch.no_grad():
                a = m(x,cl,torch.ones((1,2),dtype=torch.bool))[0]
                b = m(big,lengths,mask)[0,:2]
            self.assertTrue(torch.allclose(a,b,atol=2e-6),window)
            self.assertTrue(torch.equal(a.argmax(-1),b.argmax(-1)),window)


if __name__ == '__main__':
    unittest.main()
