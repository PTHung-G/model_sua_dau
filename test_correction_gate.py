import unittest
import numpy as np
import torch
from train_correction_gate import features,gate_predict

class GateTests(unittest.TestCase):
    def data(self):
        return dict(base=np.array([[2.,1.,-1e9],[1.,2.,-1e9],[3.,0.,-1e9]],dtype='float32'),
                    rerank=np.array([[1.,2.,-1e9],[1.,2.,-1e9],[0.,3.,-1e9]],dtype='float32'),
                    counts=np.array([2,2,2]),sentence=np.array([0,0,1]),
                    length=np.array([2,2,1]),position=np.array([0,1,0]))
    def test_features_finite_and_no_gold(self):
        d=self.data();x=features(d)
        self.assertEqual(x.shape,(3,11));self.assertTrue(np.isfinite(x).all())
        d['gold']=np.array([9,9,9]);np.testing.assert_array_equal(x,features(d))
    def test_gate_extremes_and_safe_epoch0(self):
        d=self.data();c=dict(epoch=1,mean=np.zeros(11),std=np.ones(11),
                            model={'weight':torch.zeros(1,11),'bias':torch.tensor([100.])})
        np.testing.assert_array_equal(gate_predict(d,c),d['rerank'].argmax(-1))
        c['model']['bias']=torch.tensor([-100.])
        np.testing.assert_array_equal(gate_predict(d,c),d['base'].argmax(-1))
        c['epoch']=0;np.testing.assert_array_equal(gate_predict(d,c),d['rerank'].argmax(-1))

if __name__=='__main__':unittest.main()
