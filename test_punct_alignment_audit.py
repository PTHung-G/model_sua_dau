"""Independent regression tests; run: python -m unittest test_punct_alignment_audit -v."""
import json
import tempfile
import unittest
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'charexp'))
import build_punct_tensors_v1 as b
import syltrain_punct as t

class Alignment(unittest.TestCase):
    def fixture(self, root):
        rows=[('“Hom nay toi buon, ban co yeu toi khong?”','“Hôm nay tôi buồn, bạn có yêu tôi không?”'),('Toi di!  Ban o lai; dung khong?','Tôi đi!  Bạn ở lại; đúng không?'),('Gia 3,14... (co le).','Giá 3,14... (có lẽ).')]
        p=root/'source.jsonl'
        p.write_text(''.join(json.dumps({'input':s,'target':g},ensure_ascii=False)+'\n' for s,g in rows))
        bases,vocab,cid,pid,_=b.build_table(p)
        prefix=root/'train'
        b.encode(p,prefix,bases,vocab,cid,pid)
        (root/'manifest.json').write_text(json.dumps({'pipeline_schema':2}))
        return rows,prefix,pid

    def test_actual_batch_every_gap_shuffled_and_segmented(self):
        with tempfile.TemporaryDirectory() as d:
            rows,prefix,pid=self.fixture(Path(d));data=t.load_data(prefix)
            starts,lengths=t.segments(data[2],data[3],3)
            order=np.arange(len(starts))[::-1]
            x,y,ln,ff,gg=t.make_batch(data,starts,lengths,order,'cpu')
            for row,k in enumerate(order):
                start=int(starts[k]);n=int(lengths[k]);doc=int(np.searchsorted(data[3],start,side='right')-1);delta=start-int(data[3][doc])
                ms,gs=b.gaps(rows[doc][0]);expected=[pid[g[0]] for g in gs]+[pid[gs[-1][1]]]
                self.assertEqual(gg[row,:n+1].tolist(),expected[delta:delta+n+1])
                self.assertEqual(ff[row,:n].tolist(),[b.case_id(m.group()) for m in ms][delta:delta+n])
                self.assertTrue((gg[row,n+1:]==0).all())
            self.assertEqual(sum(map(int,lengths)),len(data[0]))
            with self.assertRaises(ValueError):t.make_batch(data,np.array([int(data[3][1])-1]),np.array([2]),[0],'cpu')

    def test_dense_gap_vocab_and_pair_validation(self):
        with tempfile.TemporaryDirectory() as d:
            _,prefix,pid=self.fixture(Path(d))
            self.assertEqual(sorted(pid.values()),list(range(len(pid))))
            self.assertEqual(pid[b.PAD_GAP],0);self.assertEqual(pid[b.UNK_GAP],1)
            self.assertNotEqual(pid[''],0)
            with self.assertRaises(ValueError):b.validate_pair('toi, ban','tôi bạn')

    def test_256_words_retained_and_unknown_gap_is_not_padding(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.jsonl';target=' '.join(['tôi']*256)+'?!'
            p.write_text(json.dumps({'input':b.strip_diac(target),'target':target})+'\n')
            bases,vocab,cid,pid,_=b.build_table(p)
            stats=b.encode(p,Path(d)/'z',bases,vocab,cid,{b.PAD_GAP:0,b.UNK_GAP:1,'':2})
            self.assertEqual(stats['rows'],1)
            self.assertEqual(int(np.load(str(Path(d)/'z')+'_gaps.npy')[-1]),1)

    def test_unknown_character_does_not_truncate_later_characters(self):
        with tempfile.TemporaryDirectory() as d:
            _,prefix,_=self.fixture(Path(d))
            nc,bc,cc,cid,_=t.build_tensors(str(prefix),min_char_count=10**9)
            self.assertEqual(nc,2)
            self.assertTrue((bc[1:,0]==1).all())
            self.assertEqual(int(bc[0].sum()),0)

    def test_sampler_exact_coverage_and_schedule(self):
        lengths=np.array([1,5,3,10,4,2,6]);order=np.concatenate(list(t.epoch_batches(lengths,3,np.random.default_rng(4))))
        self.assertEqual(sorted(order.tolist()),list(range(len(lengths))))
        self.assertAlmostEqual(t.lr_factor(99,100),.05)
        with self.assertRaises(ValueError):list(t.epoch_batches(lengths,0,np.random.default_rng(4)))

if __name__=='__main__':unittest.main()
